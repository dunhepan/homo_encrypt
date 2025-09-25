# 同态加密逻辑回归 SDK（配置驱动 + 模块化编排）

本 SDK 将原先的 Streamlit 版逻辑回归（支持 CKKS 全同态加密）重构为“配置驱动 + 模块化编排”的工程形态，便于迁移、复用与扩展。支持两种训练模式：
- 明文训练（baseline）
- 同态加密训练（CKKS，TenSEAL）

## 目录结构
- sdk_he/
  - __init__.py                # SDK 出口：导出 load_config 与 SecureHEPipeline
  - config.py                  # dataclass 配置模型 + YAML 解析
  - encryption.py              # TenSEAL CKKSEncryptor
  - model.py                   # EncryptedLR（同态 LR）与明文训练逻辑
  - data.py                    # 数据读取、预处理、切分与 SMOTE
  - eval.py                    # 评估与阈值搜索、ROC/AUC 计算
  - pipeline.py                # 编排器：按配置顺序执行全流程
- examples/
  - config_he.yml              # 示例配置（可按需修改）
  - run_he.py                  # 入口脚本（可选择明文/HE 训练）

## 安装依赖
```bash
pip install tenseal numpy pandas scikit-learn imbalanced-learn matplotlib optuna plotly chardet
```

TenSEAL 依赖环境如 OpenMP、C++ 编译工具链，请参考官方安装指南。

## 运行示例
```bash
# 项目根目录下运行
python examples/run_he.py --config examples/config_he.yml
```

### 新增：加密训练进度条
启用同态加密 (`he.enabled: true`) 时现在会显示 `tqdm` 进度条：`HE Training`，统计所有类别 * 所有 epoch 的批次数。
如果本地尚未安装 tqdm：
```bash
pip install tqdm
```

### 训练稳定性改进
`EncryptedLR` 现在使用 AdamW + 多项式 sigmoid 近似 `0.5 + 0.197x - 0.004x^3`，并在每若干 epoch 对验证集做召回监控：
- 记录最佳召回对应的权重（防止后期漂移）
- 发现全 1 或全 0 预测时自动微调 bias 纠偏
- 逐批即时更新，避免之前累积梯度统计错误导致的“全部预测为 1”塌陷

你可以通过调整 YAML 中的：`train.iterations`, `train.learning_rate`, 以及 `he.add_noise_std(_update)` 控制收敛速度与隐私噪声强度。

## 为什么“改 YAML 就能换应用”
- 所有环境/超参/特征工程/训练/评估相关的变量都在 config_he.yml 中声明；
- config.py 负责把 YAML 解析成强类型 dataclass（SDKConfig），pipeline 与各功能模块“只消费配置”，不依赖硬编码；
- 迁移到不同数据、不同参数组合，只需改 YAML，无需改代码。

## 流程总览
1) run_he.py 读取 YAML -> load_config() -> SDKConfig
2) SecureHEPipeline(cfg).run_all():
   - load_and_preprocess()：读文件、预处理、缩放、分割、SMOTE（可选）
   - train_plain() 或 train_he()：按 cfg.he.enabled 决定路径
   - evaluate()：ROC/AUC、混淆矩阵、分类报告、阈值搜索（可选）
3) 打印/保存指标与模型参数
4) 释放资源并退出
