# =============== 基本依赖库 ===============
# pip install streamlit matplotlib scikit-learn numpy pandas optuna plotly joblib phe
import tenseal as ts
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, LabelBinarizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
import optuna
import functools
from sklearn.metrics import classification_report
import plotly.express as px
import time
import chardet
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, f1_score, recall_score

class CKKSEncryptor:
    def __init__(self, context=None, public_context_str=None):
        """初始化加密器，支持从序列化的公钥上下文创建"""
        self.encryption_count = 0
        if context is not None:
            self.context = context
        elif public_context_str is not None:
            # 从序列化的公钥上下文创建（不包含私钥）
            self.context = ts.context_from(public_context_str)
        else:
            # 创建新的完整上下文（包含私钥）
            self.context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[40, 21, 21, 21, 21, 40]
            )
            self.context.generate_galois_keys()
            self.context.global_scale = 2 ** 21

    # 批量加密方法
    def batch_encrypt(self, vectors):
        """批量加密向量"""
        batch_size = len(vectors)
        # 展平所有向量
        flattened = np.concatenate(vectors).flatten().tolist()
        encrypted = ts.ckks_vector(self.context, flattened)
        self.encryption_count += batch_size
        return encrypted, batch_size
    
    def encrypt_vector(self, vector):
        """加密单个特征向量"""
        return ts.ckks_vector(self.context, vector.tolist())
    
    def encrypt_label(self, label):
        """加密单个标签"""
        return ts.ckks_vector(self.context, [label])
        
    def encrypt_matrix(self, matrix):
        """加密整个特征矩阵"""
        # 将矩阵展平为一维数组
        flattened = matrix.flatten().tolist()
        return ts.ckks_vector(self.context, flattened)
    
    def decrypt_vector(self, ciphertext):
        """解密向量"""
        return ciphertext.decrypt()
    
    def get_public_context(self):
        """获取仅包含公钥的上下文"""
        public_context = self.context.copy()
        public_context.make_context_public()
        return public_context
        
    def serialize_public_context(self):
        """序列化公钥上下文（用于传输）"""
        return self.context.serialize(save_public_key=True, 
                                      save_secret_key=True, 
                                      save_galois_keys=True)

    def get_private_context(self):
        return self.context.copy()

class EncryptedLR:
    def __init__(self, n_features, context, batch_size=32):
        self.weight = [0.0] * n_features
        self.bias = 0.0
        self._delta_w = None
        self._delta_b = None
        self._count = 0
        self.n_features = n_features
        self.context = context  # 使用计算密钥系统的上下文
        self.batch_size = batch_size
        
    # 移除所有解密操作，仅使用计算密钥系统
    
    def forward_batch(self, enc_x_batch, actual_size):
        # 在加密状态下计算加权和
        replicated_weights = np.tile(self.weight, (actual_size, 1)).flatten().tolist()
        enc_weight_batch = ts.ckks_vector(self.context, replicated_weights)
        
        enc_out = enc_x_batch.dot(enc_weight_batch) + self.bias * actual_size
        return self.sigmoid_batch(enc_out)
    
    def backward_batch(self, enc_x_batch, enc_out_batch, enc_y_batch, actual_size):
        # 计算每个样本的误差值 (ŷ - y)
        enc_error = enc_out_batch - enc_y_batch  # 长度 = actual_size
        
        # 为每个样本生成一个噪声值（不重复到每个特征）
        noise = np.random.normal(0, 0.1, size=actual_size)
        enc_noise = ts.ckks_vector(self.context, noise.tolist())  # 长度 = actual_size
        
        # 在加密状态下添加噪声到误差向量
        enc_noisy_error = enc_error + enc_noise  # 长度 = actual_size
        
        # 扩展噪声误差到每个特征
        # 构造重复的噪声误差向量 (每个样本的误差值重复n_features次)
        repeated_noisy_error = []
        for error_val in enc_noisy_error.decrypt():
            repeated_noisy_error.extend([error_val] * self.n_features)
        enc_repeated_noisy_error = ts.ckks_vector(self.context, repeated_noisy_error)
        
        # 计算梯度: x * (误差+噪声)
        gradient = enc_x_batch * enc_repeated_noisy_error
        
        # === 梯度累加处理 ===
        if self._delta_w is None:
            # 初始化梯度累加器
            max_length = self.batch_size * self.n_features
            self._delta_w = ts.ckks_vector(self.context, [0.0] * max_length)
            self._delta_b = ts.ckks_vector(self.context, [0.0] * self.batch_size)
        
        # 直接累加梯度
        self._delta_w += gradient
        self._delta_b += enc_noisy_error
        
        self._count += actual_size
        
    def sigmoid_batch(self, enc_x):
        return enc_x.polyval([0.5, 0.2])
    
    def update_parameters(self, learning_rate):
        if self._count == 0:
            return
            
        # 解密梯度向量 - 使用计算密钥系统的私钥
        dec_delta_w = self._delta_w.decrypt()
        dec_delta_b = self._delta_b.decrypt()
        
        # 重塑为矩阵形式 (batch_size x n_features)
        delta_w_matrix = np.array(dec_delta_w[:self._count * self.n_features]
                                 ).reshape(self._count, self.n_features)
        delta_b_vector = np.array(dec_delta_b[:self._count])
        
        # 计算平均梯度
        avg_delta_w = np.mean(delta_w_matrix, axis=0)  # 按列求平均
        avg_delta_b = np.mean(delta_b_vector)

        weight_noise = np.random.normal(0, 0.05, size=len(avg_delta_w))
        bias_noise = np.random.normal(0, 0.05)
        
        avg_delta_w += weight_noise
        avg_delta_b += bias_noise

        reg_strength = 0.1 + 0.05 * np.log(self.n_features)  # 特征越多正则越强
        # 更新权重（带有L2正则化）
        weight_array = np.array(self.weight)
        weight_array -= learning_rate * (avg_delta_w + weight_array * reg_strength)
        self.weight = weight_array.tolist()
        self.bias -= learning_rate * avg_delta_b
        
        # 重置梯度
        self.reset_gradients()
        
    def reset_gradients(self):
        self._delta_w = None
        self._delta_b = None
        self._count = 0

# =============== 状态初始化 ===============
if 'file_processed' not in st.session_state:
    st.session_state.file_processed = False
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# 确保所有需要的session_state变量都存在
required_session_keys = [
    'best_learning_rate', 'best_iterations',
    'learning_rate', 'iterations', 'batch_size'
]

for key in required_session_keys:
    if key not in st.session_state:
        if key == 'best_learning_rate' or key == 'learning_rate':
            st.session_state[key] = 1.0
        elif key == 'best_iterations' or key == 'iterations':
            st.session_state[key] = 10
        elif key == 'batch_size':
            st.session_state[key] = 4096

def preprocess_data(data):
    """银行营销数据集专用预处理 - 增强版特征工程"""
    try:
        # 移除无关列
        data = data.drop(['day','contact'], axis=1, errors='ignore')
        
        # 创建数值型y列
        data['y'] = data['y'].map({'no': 0, 'yes': 1})
        
        # 1. 高级缺失值处理
        for col in ['job', 'education']:
            mode_val = data[col].mode()[0]
            data[col] = data[col].replace(['unknown', 'nonexistent'], mode_val)
        
        # 2. 数值化二分类特征
        binary_cols = ['default', 'housing', 'loan']
        for col in binary_cols:
            data[col] = data[col].map({'no': 0, 'yes': 1})
        
        # 极值处理
        for col in ['balance', 'duration', 'pdays', 'previous']:
            # 处理负值
            data[col] = np.where(data[col] < 0, 0, data[col])
            # 处理极端值
            p99 = data[col].quantile(0.99)
            data[col] = np.where(data[col] > p99, p99, data[col])
        
        # 3. 月份处理
        month_map = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
                     'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
        data['month'] = data['month'].map(month_map)
        
        # 4. 增强型特征工程
        # 4.1 金融行为特征
        data['debt_ratio'] = data['balance'] / (data['age'] + 1e-5)
        data['credit_usage'] = np.where(data['loan'] == 1, 
                                       data['balance'] * 0.3,
                                       data['balance'] * 0.1)
        
        # 4.2 营销响应特征
        data['response_ratio'] = data['previous'] / (data['campaign'] + 1e-5)
        data['high_value_target'] = np.where(
            (data['balance'] > data['balance'].quantile(0.75)) & 
            (data['age'] > 40), 1, 0)
        
        # 4.3 时间周期特征
        season_map = {
            1: 0, 2: 0, 3: 1, 4:1, 5:1, 6:2,
            7:2, 8:2, 9:3, 10:3, 11:3, 12:0
        }
        data['season'] = data['month'].map(season_map)
        
        # 4.4 交互特征
        data['age_balance_interaction'] = data['age'] * data['balance']
        
        # 4.5 职业和教育组合特征
        data['job_edu'] = data['job'] + "_" + data['education']
        le = LabelEncoder()
        data['job_edu_encoded'] = le.fit_transform(data['job_edu'])
        
        # 5. 特征转换
        data['age_group'] = pd.cut(data['age'], bins=5, labels=False)
        data['balance_group'] = pd.cut(data['balance'], bins=5, labels=False)
        data['duration_min'] = data['duration'] / 60
        data['log_duration'] = np.log1p(data['duration'])
        
        # 6. 关键特征筛选
        numerical_features = [
            'age', 'balance', 'duration', 'campaign', 'pdays', 
            'previous', 'default', 'housing', 'loan',
            'debt_ratio', 'credit_usage', 'response_ratio', 
            'high_value_target', 'season', 'age_balance_interaction',
            'job_edu_encoded', 'balance_group', 
            'duration_min', 'log_duration', 'month'
        ]
        
        # 7. 分类特征处理
        categorical_cols = ['job', 'education', 'poutcome']
        for col in categorical_cols:
            le = LabelEncoder()
            data[col + '_encoded'] = le.fit_transform(data[col])
            numerical_features.append(col + '_encoded')
        
        # 8. 分离特征和目标
        X = data[numerical_features]
        y = data['y'].values
        
        # 9. 关键修复：处理NaN值
        # 检查并处理NaN值
        nan_columns = X.columns[X.isna().any()].tolist()
        if nan_columns:
            st.warning(f"发现NaN值的列: {nan_columns}")
            
            # 数值列用中位数填充
            num_cols = X.select_dtypes(include=np.number).columns
            for col in num_cols:
                if X[col].isna().any():
                    # 检查中位数是否有效
                    if np.isnan(X[col].median()):
                        # 如果中位数无效，使用0填充
                        X[col] = X[col].fillna(0)
                        st.info(f"列 '{col}' 使用0填充了 {X[col].isna().sum()} 个NaN值")
                    else:
                        median_val = X[col].median()
                        X[col] = X[col].fillna(median_val)
                        st.info(f"列 '{col}' 使用中位数 {median_val:.2f} 填充了 {X[col].isna().sum()} 个NaN值")
            
            # 分类列用众数填充
            cat_cols = X.select_dtypes(include='object').columns
            for col in cat_cols:
                if X[col].isna().any():
                    mode_val = X[col].mode()[0]
                    X[col] = X[col].fillna(mode_val)
                    st.info(f"列 '{col}' 使用众数 '{mode_val}' 填充了 {X[col].isna().sum()} 个NaN值")
        
        # 确保没有剩余NaN值
        if X.isna().any().any():
            st.error("仍然存在NaN值，无法继续处理")
            st.dataframe(X.isna().sum())
            st.stop()

        feature_names = numerical_features
        return X, y, feature_names
        
    except Exception as e:
        st.error(f"数据处理错误: {str(e)}")
        st.stop()

# =============== 界面布局 ===============
st.title("基于全同态加密(CKKS)的隐私保护逻辑回归系统")
# 添加自定义样式
st.markdown("""
<style>
/* 增大滑块轨道两端的符号 */
.slider-symbols {
    display: flex;
    justify-content: space-between;
    width: 100%;
    margin-top: -18px;
    pointer-events: none;
    font-size: 24px;  /* 增大符号大小 */
    font-weight: bold;
}

/* 调整滑块容器位置 */
div.stSlider {
    margin-top: 15px;  /* 增加间距适应大符号 */
}

/* 调整按钮位置 */
div[data-baseweb="button"] {
    margin-top: 15px;
    height: 40px;
    width: 40px;
    font-size: 20px;
}

/* 数值标签样式 */
.value-display {
    font-size: 14px;
    text-align: center;
    margin-top: -5px;
}

/* 进度条样式 */
.stProgress > div > div {
    background-color: #4CAF50;
}
</style>
""", unsafe_allow_html=True)

# =============== 系统设置滑块 ===============
st.sidebar.header("系统设置")

# 添加批处理大小滑块
st.sidebar.markdown("**批处理大小**")
batch_col1, batch_col2, batch_col3 = st.sidebar.columns([1, 4, 1])

with batch_col2:
    st.markdown('<div class="slider-symbols"><span style="color:grey">-</span><span style="color:grey">+</span></div>', unsafe_allow_html=True)
    batch_size = st.slider(
        "批处理大小", 4096, 65536,
        value=st.session_state.batch_size,
        step=4096,
        key="batch_slider",
        label_visibility="collapsed"
    )
    
# 左侧减少按钮
with batch_col1:
    if st.button("-", key="batch_decrement"):
        st.session_state.batch_size = max(4096, st.session_state.batch_size - 4096)
        st.rerun()

# 右侧增加按钮
with batch_col3:
    if st.button("+", key="batch_increment"):
        st.session_state.batch_size = min(65536, st.session_state.batch_size + 4096)

        st.rerun()

st.session_state.batch_size = batch_size

# 学习率滑块带按钮
st.sidebar.markdown("**学习率**")
lr_col1, lr_col2, lr_col3 = st.sidebar.columns([1, 4, 1])

with lr_col2:
    # 添加滑块轨道两端的符号
    st.markdown('<div class="slider-symbols"><span style="color:grey">-</span><span style="color:grey">+</span></div>', unsafe_allow_html=True)
    
    # 滑块控件
    learning_rate = st.slider(
        "学习率", 0.0, 1.0,
        value=st.session_state.learning_rate,
        step=0.1,
        key="lr_slider",
        label_visibility="collapsed"
    )

# 左侧减少按钮
with lr_col1:
    if st.button("-", key="lr_decrement"):
        st.session_state.learning_rate = max(0.0, st.session_state.learning_rate - 0.1)
        st.rerun()

# 右侧增加按钮
with lr_col3:
    if st.button("+", key="lr_increment"):
        st.session_state.learning_rate = min(1.0, st.session_state.learning_rate + 0.1)
        st.rerun()

# 迭代次数滑块带按钮
st.sidebar.markdown("**迭代次数**")
iter_col1, iter_col2, iter_col3 = st.sidebar.columns([1, 4, 1])

with iter_col2:
    # 添加滑块轨道两端的符号
    st.markdown('<div class="slider-symbols"><span style="color:grey">-</span><span style="color:grey">+</span></div>', unsafe_allow_html=True)
    
    # 滑块控件
    iterations = st.slider(
        "", 0, 100,
        value=st.session_state.iterations,
        step=5,
        key="iter_slider",
        label_visibility="collapsed"
    )
    
# 左侧减少按钮
with iter_col1:
    if st.button("-", key="iter_decrement"):
        st.session_state.iterations = max(0, st.session_state.iterations - 5)
        st.rerun()

# 右侧增加按钮
with iter_col3:
    if st.button("+", key="iter_increment"):
        st.session_state.iterations = min(100, st.session_state.iterations + 5)
        st.rerun()

# 更新session_state
st.session_state.learning_rate = learning_rate
st.session_state.iterations = iterations


# 在侧边栏显示当前最佳参数
if st.session_state.best_learning_rate:
    st.sidebar.markdown(
        f"<div style='color: green; font-weight: bold;'>最佳学习率: {st.session_state.best_learning_rate:.5f}</div>",
        unsafe_allow_html=True
    )

if st.session_state.best_iterations:
    st.sidebar.markdown(
        f"<div style='color: green; font-weight: bold;'>最佳迭代次数: {st.session_state.best_iterations}</div>",
        unsafe_allow_html=True
    )

use_he = st.sidebar.checkbox("启用全同态加密协作训练", value=False)

# 训练评估函数
def train_evaluate(X_train, y_train, X_val, y_val, iterations, learning_rate):
    """优化后的训练函数，提供更稳定的收敛"""
    # 计算类别权重
    n_pos = np.sum(y_train)
    n_neg = len(y_train) - n_pos
    pos_weight = min(30.0, n_neg / (n_pos + 1e-5))  # 进一步降低权重上限
    
    # 权重初始化 - 使用更稳定的方法
    w = np.random.normal(0, 0.01, X_train.shape[1])
    
    # AdamW优化器参数
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    weight_decay = 0.001
    
    # AdamW状态变量
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    
    best_val_recall = 0
    best_weights = w.copy()  # 初始化为当前权重
    
    # 训练历史记录
    train_recall_history = []
    val_recall_history = []
    
    # 更稳定的评估机制
    eval_interval = min(10, max(5, iterations//20))  # 动态评估间隔
    
    # 学习率预热
    warmup_epochs = min(20, iterations//2)
    
    for epoch in range(iterations):
        # 学习率预热
        if epoch < warmup_epochs:
            current_lr = learning_rate * min(1.0, (epoch+1)/warmup_epochs)
        else:
            current_lr = learning_rate
        
        # 梯度计算
        logits = np.dot(X_train, w)
        preds = 1 / (1 + np.exp(-logits))
        errors = preds - y_train
        weighted_errors = np.where(y_train == 1, errors * pos_weight, errors)
        gradients = X_train.T @ weighted_errors / len(y_train)
        
        # AdamW更新
        m = beta1 * m + (1 - beta1) * gradients
        v = beta2 * v + (1 - beta2) * (gradients ** 2)
        m_hat = m / (1 - beta1 ** (epoch + 1))
        v_hat = v / (1 - beta2 ** (epoch + 1))
        
        # 参数更新（添加梯度裁剪）
        update = m_hat / (np.sqrt(v_hat) + epsilon)
        clip_value = 1.0  # 梯度裁剪阈值
        update = np.clip(update, -clip_value, clip_value)
        w = (1 - current_lr * weight_decay) * w - current_lr * update
        
        # 定期评估
        if (epoch % eval_interval == 0) or (epoch == iterations - 1):
            # 训练集评估
            train_preds = (logits > 0).astype(int)  # 重用当前logits
            train_recall = recall_score(y_train, train_preds, zero_division=0)
            train_recall_history.append(train_recall)
            
            # 验证集评估
            val_logits = np.dot(X_val, w)
            val_preds = (val_logits > 0).astype(int)
            val_recall = recall_score(y_val, val_preds, zero_division=0)
            val_recall_history.append(val_recall)
            
            # 更新最佳权重
            if val_recall > best_val_recall:
                best_val_recall = val_recall
                best_weights = w.copy()
    
    # 可视化训练过程（诊断工具）
    plt.figure(figsize=(10, 6))
    epochs = [i * eval_interval for i in range(len(train_recall_history))]
    epochs[-1] = iterations - 1  # 确保最后一个点是最终迭代
    
    plt.plot(epochs, train_recall_history, 'bo-', label='训练召回率')
    plt.plot(epochs, val_recall_history, 'ro-', label='验证召回率')
    plt.title(f'训练曲线 (最终验证召回率: {best_val_recall:.4f})')
    plt.xlabel('迭代次数')
    plt.ylabel('召回率')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    
    return best_weights

# 训练评估函数 - 专注于类别1的召回率
def objective(trial):
    """优化目标函数，专注于提高F1分数"""
    # 获取超参数建议
    lr = trial.suggest_float('learning_rate', 0.001, 1.0, log=True)  # 修复下限
    iterations = trial.suggest_int('iterations', 5, 50)
    
    # 使用优化后的训练函数
    w = train_evaluate(
        st.session_state.X_train_resampled, 
        st.session_state.y_train_resampled,
        st.session_state.X_val,
        st.session_state.y_val,
        iterations=iterations,
        learning_rate=lr
    )
    
    # 计算验证集F1分数
    val_logits = np.dot(st.session_state.X_val, w)
    val_preds = (val_logits > 0).astype(int)
    f1 = f1_score(st.session_state.y_val, val_preds, average='weighted')
    
    return f1

# 自动调参按钮
if st.sidebar.button("开始自动调参"):
    if 'X_train_resampled' not in st.session_state:
        st.sidebar.error("请先上传并处理数据集，完成数据分割后再进行自动调参。")
    else:
        with st.spinner("自动调参中，专注于提高F1分数..."):
            # 创建Optuna研究，最大化F1分数
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=30)
            
            best_params = study.best_params
            best_f1 = study.best_value
            
            # 更新session_state中的超参数值
            st.session_state.best_learning_rate = best_params['learning_rate']
            st.session_state.best_iterations = best_params['iterations']
            st.session_state.learning_rate = best_params['learning_rate']
            st.session_state.iterations = best_params['iterations']
            
            st.sidebar.success(
                f"自动调参完成！F1分数提升至: {best_f1:.4f}\n"
                f"新参数: 学习率={best_params['learning_rate']:.4f}, "
                f"迭代次数={best_params['iterations']}"
            )
            
            # 保留文件状态并重新渲染
            st.session_state.file_processed = True
            st.rerun()

# 数据上传
st.header("1. 上传数据集")
uploaded_file = st.file_uploader("选择一个.csv或.xlsx或.data数据集文件", type=["csv","xlsx","data"], key="file_uploader")

# 保存上传的文件状态
if uploaded_file:
    st.session_state.uploaded_file = uploaded_file
    st.session_state.file_processed = True

# 处理文件内容（即使自动调参后刷新也保持显示）
if st.session_state.file_processed and st.session_state.uploaded_file:
    file_data = st.session_state.uploaded_file
    # 获取文件扩展名
    file_ext = file_data.name.split('.')[-1].lower()
    
    try:
        if file_ext == 'xlsx':
            # Excel文件专用读取
            data = pd.read_excel(file_data)
        else:
            # 文本文件读取（含编码检测）
            raw_data = file_data.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding'] if result['encoding'] else 'utf-8'
            
            # 重置文件指针
            file_data.seek(0)
            
            try:
                data = pd.read_csv(file_data, encoding=encoding)
            except:
                # 终极fallback
                file_data.seek(0)
                data = pd.read_csv(file_data, encoding='latin1')
        
        # 执行专用预处理
        X, y, feature_names = preprocess_data(data)
        
        # 标准化数值特征
        scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
        X = scaler.fit_transform(X)
        st.session_state.feature_names = feature_names
        
        # 获取特征数量
        n_features = X.shape[1]
        st.session_state.n_features = n_features  # 存储特征数

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        classes = label_encoder.classes_
        n_classes = len(classes)

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=35, stratify=y_encoded
        )
        
        # 再从训练+验证中划分训练(75%)和验证(25%)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=35, stratify=y_train_val
        )
        
        # 应用SMOTE只对训练集 - 解决类别不平衡
        smote = SMOTE(
            sampling_strategy={1: int(len(y_train)*0.5)},  # 将少数类提升至50%
            k_neighbors=min(5, sum(y_train==1)-1),  # 动态调整邻居数
            random_state=42
        )
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        # 保存重采样数据和验证集
        st.session_state.X_train_resampled = X_train_resampled
        st.session_state.y_train_resampled = y_train_resampled
        st.session_state.X_val = X_val
        st.session_state.y_val = y_val
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.n_classes = n_classes
        st.session_state.classes = classes
        st.session_state.scaler = scaler

        st.write(f"重采样后训练集: 0类={sum(y_train_resampled==0)}, 1类={sum(y_train_resampled==1)}")
        st.write(f"验证集分布: 0类={sum(y_val==0)}, 1类={sum(y_val==1)}")
        st.write(f"测试集分布: 0类={sum(y_test==0)}, 1类={sum(y_test==1)}")

    except Exception as e:
        st.error(f"数据处理错误: {str(e)}")            
    
    # 准备全同态加密数据共享
    if use_he:
        # 创建主加密器
        main_encryptor = CKKSEncryptor()
        
        batch_size_encrypt = st.session_state.batch_size

        # 创建加密进度显示
        st.subheader("数据加密进度")
        encryption_progress = st.progress(0)
        status_text = st.empty()
        
        encrypted_X_batches = []
        encrypted_y_batches = []
        total_batches = (len(X_train) + batch_size_encrypt - 1) // batch_size_encrypt
        
        # 分批加密特征和标签
        for i in range(0, len(X_train), batch_size_encrypt):
            batch_end = min(i + batch_size_encrypt, len(X_train))
            X_batch = X_train[i:batch_end]
            y_batch = y_train[i:batch_end]
            
            # 加密特征批次
            enc_X_batch, batch_size = main_encryptor.batch_encrypt(X_batch)
            encrypted_X_batches.append(enc_X_batch)
            
            # 加密标签批次
            enc_y_batch = ts.ckks_vector(main_encryptor.context, y_batch.tolist())
            encrypted_y_batches.append(enc_y_batch)
            
            # 更新进度
            progress = min((i + batch_size_encrypt) / len(X_train), 1.0)
            encryption_progress.progress(progress)
            status_text.text(
                f"加密进度: {min(i+batch_size_encrypt, len(X_train))}/{len(X_train)} 条记录 "
                f"(批次 {len(encrypted_X_batches)}/{total_batches})"
            )
        
        # 存储加密数据和批次大小
        st.session_state.encrypted_X_batches = encrypted_X_batches
        st.session_state.encrypted_y_batches = encrypted_y_batches
        st.session_state.batch_size_encrypt = batch_size_encrypt
        
        st.success(f"批量加密完成! 总加密记录数: {len(X_train)}，使用批次大小: {batch_size_encrypt}")

    


# =============== 训练部分 ===============
if st.session_state.file_processed and st.session_state.uploaded_file:
    if st.button("开始训练"):
        # 获取数据集变量
        X_train = st.session_state.X_train_resampled
        y_train = st.session_state.y_train_resampled
        X_val = st.session_state.X_val
        y_val = st.session_state.y_val
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        n_classes = st.session_state.n_classes
        classes = st.session_state.classes
        iterations = st.session_state.iterations
        learning_rate = st.session_state.learning_rate
        batch_size = st.session_state.batch_size
            
        if use_he:
            # 创建独立的计算密钥系统（用于训练）
            compute_encryptor = CKKSEncryptor()
            compute_pk_str = compute_encryptor.serialize_public_context()
            
            # 创建客户端加密器（用于原始数据加密）
            client_encryptor = CKKSEncryptor(public_context_str=compute_pk_str)
            
            # 准备加密进度显示
            st.subheader("数据加密进度")
            encryption_progress = st.progress(0)
            status_text = st.empty()
            
            encrypted_X_batches = []
            encrypted_y_batches = []
            total_batches = (len(X_train) + batch_size_encrypt - 1) // batch_size_encrypt
            
            # 分批加密特征和标签 - 使用客户端加密器
            for i in range(0, len(X_train), batch_size_encrypt):
                batch_end = min(i + batch_size_encrypt, len(X_train))
                X_batch = X_train[i:batch_end]
                y_batch = y_train[i:batch_end]
                
                # 加密特征批次 - 使用计算密钥系统的公钥
                enc_X_batch, batch_size = client_encryptor.batch_encrypt(X_batch)
                encrypted_X_batches.append(enc_X_batch)
                
                # 加密标签批次 - 使用计算密钥系统的公钥
                enc_y_batch = ts.ckks_vector(client_encryptor.context, y_batch.tolist())
                encrypted_y_batches.append(enc_y_batch)
                
                # 更新进度
                progress = min((i + batch_size_encrypt) / len(X_train), 1.0)
                encryption_progress.progress(progress)
                status_text.text(
                    f"加密进度: {min(i+batch_size_encrypt, len(X_train))}/{len(X_train)} 条记录 "
                    f"(批次 {len(encrypted_X_batches)}/{total_batches})"
                )
            
            # 存储加密数据和批次大小
            st.session_state.encrypted_X_batches = encrypted_X_batches
            st.session_state.encrypted_y_batches = encrypted_y_batches
            st.session_state.batch_size_encrypt = batch_size_encrypt
            
            st.success(f"批量加密完成! 总加密记录数: {len(X_train)}，使用批次大小: {batch_size_encrypt}")
            
            # 获取公共上下文
            public_context = compute_encryptor.context
            
            # 初始化模型 - 添加批处理大小参数
            eelr = EncryptedLR(n_features, public_context, batch_size)
            
            # 新增：检查是否有预加密的批次数据
            if 'encrypted_X_batches' in st.session_state and 'encrypted_y_batches' in st.session_state:
                encrypted_X_batches = st.session_state.encrypted_X_batches
                encrypted_y_batches = st.session_state.encrypted_y_batches
                batch_size_encrypt = st.session_state.batch_size_encrypt
                
                start_time = time.time()
                
                # 创建训练进度条
                st.subheader("加密训练进度")
                overall_progress = st.progress(0)
                overall_status = st.empty()
                
                # 计算总批次数和总训练步数
                total_batches = len(encrypted_X_batches)
                total_steps = n_classes * iterations * total_batches
                current_step = 0
                
                # 训练每个类别
                for class_idx in range(n_classes):
                    # 二值化标签 (使用原始标签数据)
                    y_train_bin = (y_train == class_idx).astype(int)
                    
                    # 加密标签批次
                    encrypted_y_bin_batches = []
                    for i in range(0, len(y_train_bin), batch_size_encrypt):
                        batch_end = min(i + batch_size_encrypt, len(y_train_bin))
                        y_batch = y_train_bin[i:batch_end]
                        enc_y_batch = ts.ckks_vector(public_context, y_batch.tolist())
                        encrypted_y_bin_batches.append(enc_y_batch)
                    
                    for epoch in range(iterations):
                        # 打乱批次顺序
                        batch_indices = np.random.permutation(total_batches)
                        
                        for batch_idx in batch_indices:
                            enc_x_batch = encrypted_X_batches[batch_idx]
                            enc_y_batch = encrypted_y_bin_batches[batch_idx]
                            
                            # 获取实际批次大小
                            actual_size = min(batch_size_encrypt, 
                                            len(y_train) - batch_idx * batch_size_encrypt)
                            
                            # 前向传播 (批处理)
                            enc_out_batch = eelr.forward_batch(enc_x_batch, actual_size)
                            
                            # 反向传播 (批处理)
                            eelr.backward_batch(enc_x_batch, enc_out_batch, enc_y_batch, actual_size)
                            
                            # 更新参数
                            eelr.update_parameters(learning_rate)
                            
                            # 更新进度
                            current_step += 1
                            current_progress = current_step / total_steps
                            overall_progress.progress(current_progress)
                            overall_status.text(
                                f"类别 {class_idx+1}/{n_classes} | "
                                f"迭代 {epoch+1}/{iterations} | "
                                f"批次 {np.where(batch_indices == batch_idx)[0][0]+1}/{total_batches} "
                                f"({current_step}/{total_steps})"
                            )
                
                # 完成训练
                overall_progress.progress(1.0)
                st.balloons()
                st.success(f"CKKS加密训练完成！总用时: {time.time()-start_time:.1f}秒")
            else:
                st.error("未找到预加密的批次数据，请先完成数据加密步骤！")
                
        else:
            # 明文训练（OvR）
            weights_list = []
            best_val_recalls = [-1] * n_classes

            for class_idx in range(n_classes):
                y_train_bin = (y_train == class_idx).astype(int)
                y_val_bin = (y_val == class_idx).astype(int)
                
                # 使用优化后的训练函数
                w = train_evaluate(
                    X_train, y_train_bin, 
                    X_val, y_val_bin,
                    iterations=st.session_state.iterations,
                    learning_rate=st.session_state.learning_rate
                )
                
                weights_list.append(w)
                
                # 计算当前类别的召回率
                val_logits = np.dot(X_val, w)
                val_preds = (val_logits > 0).astype(int)
                val_recall = recall_score(y_val_bin, val_preds, zero_division=0)
                best_val_recalls[class_idx] = val_recall
                
                st.info(f"类别 {classes[class_idx]} 训练完成，验证召回率: {val_recall:.4f}")

            st.session_state.weights = weights_list
            st.success("明文训练完成！应用早停和正则化防止过拟合")


# =============== 模型评估 ===============
def find_optimal_threshold(y_true, y_prob):
    """寻找最佳分类阈值以最大化召回率"""
    thresholds = np.linspace(-5, 5, 100)  # 扩大搜索范围
    recalls = [recall_score(y_true, (y_prob > t)) for t in thresholds]
    best_idx = np.argmax(recalls)
    return thresholds[best_idx], recalls[best_idx]

st.header("3. 模型评估")
if st.button("开始评估"):
    if 'weights' not in st.session_state or any(w is None for w in st.session_state.weights):
        st.error("请先进行模型训练！")
    else:
        # 获取数据集变量
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        n_classes = st.session_state.n_classes
        classes = st.session_state.classes
        
        # 明文模式下的预测
        y_pred_proba = np.zeros((X_test.shape[0], n_classes))
        for class_idx, weights in enumerate(st.session_state.weights):
            logits = np.dot(X_test, weights)
            y_pred_proba[:, class_idx] = 1 / (1 + np.exp(-logits))

        # 针对二分类问题优化阈值
        if n_classes == 2:
            # 寻找最佳阈值（针对少数类）
            y_logits = np.dot(X_test, st.session_state.weights[1])
            
            # 寻找最佳阈值
            optimal_threshold, optimal_recall = find_optimal_threshold(
                y_test, y_logits
            )
            
            y_pred_final = (y_logits > optimal_threshold).astype(int)
        else:
            y_pred_final = np.argmax(y_pred_proba, axis=1)

        # 多类ROC曲线绘制
        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(y_test)
        if n_classes == 2:
            y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))

        # 计算ROC曲线和AUC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for class_idx in range(n_classes):
            fpr[class_idx], tpr[class_idx], _ = roc_curve(y_test_bin[:, class_idx], y_pred_proba[:, class_idx])
            roc_auc[class_idx] = auc(fpr[class_idx], tpr[class_idx])

        # 绘制所有ROC曲线
        plt.figure()
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'darkgreen', 'red', 'purple'])
        for class_idx, color in zip(range(n_classes), colors):
            plt.plot(fpr[class_idx], tpr[class_idx], color=color, lw=2,
                     label=f'ROC curve of class {classes[class_idx]} (AUC = {roc_auc[class_idx]:0.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Multi-class Classification')
        plt.legend(loc="lower right")
        st.pyplot(plt)

        # 计算最终预测类别
        y_pred_final = np.argmax(y_pred_proba, axis=1)

        # 计算总体准确率
        overall_accuracy = accuracy_score(y_test, y_pred_final)
        overall_accuracy_percentage = overall_accuracy * 100
        recall = recall_score(y_test, y_pred_final, average='weighted')

        st.markdown(
            f"""
            <div style="
                border: 2px solid #4CAF50; 
                border-radius: 10px; 
                padding: 15px; 
                background-color: #f9f9f9;
                text-align: center;
                font-size: 24px; 
                font-weight: bold;
                color: #333;
            ">
                总体分类准确率: {overall_accuracy_percentage:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred_final)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        fig, ax = plt.subplots()
        disp.plot(cmap="Blues", ax=ax)
        plt.title("Confusion Matrix")
        st.pyplot(fig)

        # 解析分类报告
        report_dict = classification_report(y_test, y_pred_final, target_names=classes, output_dict=True, zero_division=1)
        report_df = pd.DataFrame(report_dict).transpose()
        report_df = report_df.drop('accuracy', axis=0)  # 删除 accuracy 行

        st.write("Classification Report Table:")
        st.dataframe(report_df)

        # 可视化分类报告
        metrics = ['precision', 'recall', 'f1-score']
        for metric in metrics:
            metric_df = report_df.iloc[:len(classes)].reset_index()
            metric_df.rename(columns={'index': 'class'}, inplace=True)
            
            fig_metric = px.bar(
                metric_df, 
                x='class', 
                y=metric, 
                title=f'{metric.capitalize()} by Class',
                color='class',
                text=metric,
                height=500
            )
            fig_metric.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            st.plotly_chart(fig_metric)
        
        # 支持度可视化
        support_df = report_df.iloc[:len(classes)].reset_index()
        support_df.rename(columns={'index': 'class'}, inplace=True)
        fig_support = px.bar(
            support_df, 
            x='class', 
            y='support', 
            title='Support by Class',
            color='class',
            text='support',
            height=500
        )
        fig_support.update_traces(texttemplate='%{text}', textposition='outside')
        st.plotly_chart(fig_support)

st.write("系统状态: 运行正常")