from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import yaml

@dataclass
class DataConfig:
    input_path: str
    target: str = "y"
    file_type: Optional[str] = None
    test_size: float = 0.2
    val_size: float = 0.25
    random_state: int = 35
    stratify: bool = True

@dataclass
class PreprocessConfig:
    drop_cols: List[str] = field(default_factory=lambda: ["day", "contact"])
    binary_map: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "default": {"no": 0, "yes": 1},
        "housing": {"no": 0, "yes": 1},
        "loan":    {"no": 0, "yes": 1},
    })
    target_map: Dict[str, int] = field(default_factory=lambda: {"no": 0, "yes": 1})
    unknown_fill_cols: List[str] = field(default_factory=lambda: ["job", "education"])
    extreme_clip_cols: List[str] = field(default_factory=lambda: ["balance", "duration", "pdays", "previous"])
    clip_p: float = 0.99
    month_col: str = "month"
    month_map: Dict[str, int] = field(default_factory=lambda: {
        "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12
    })
    scale_feature_range: List[float] = field(default_factory=lambda: [-0.5, 0.5])
    add_engineered_features: bool = True
    engineered: Dict[str, bool] = field(default_factory=lambda: {
        "debt_ratio": True,
        "credit_usage": True,
        "response_ratio": True,
        "high_value_target": True,
        "season": True,
        "age_balance_interaction": True,
        "job_edu_encoded": True,
        "age_group": True,
        "balance_group": True,
        "duration_min": True,
        "log_duration": True,
    })
    onehot_like_cols: List[str] = field(default_factory=lambda: ["job", "education", "poutcome"])

@dataclass
class SmoteConfig:
    enabled: bool = True
    minority_ratio: float = 0.5
    k_neighbors_cap: int = 5

@dataclass
class HEConfig:
    enabled: bool = False
    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: List[int] = field(default_factory=lambda: [40, 21, 21, 21, 21, 40])
    global_scale_bits: int = 21
    add_noise_std: float = 0.1
    add_noise_std_update: float = 0.05
    # 你配置的“期望加密批大小”，最终是否生效取决于 auto_capacity 与可用槽位
    batch_size_encrypt: int = 4096
    # 自动裁剪：capacity = min(batch_size_encrypt, slots // n_features)，slots = poly_modulus_degree // 2
    # 这是“唯一决定实际加密批大小”的逻辑（见 pipeline._choose_capacity 的打印）
    auto_capacity: bool = True
    # 控制台告警是否抑制（建议 False 便于排查）
    suppress_warnings: bool = False

@dataclass
class TrainConfig:
    iterations: int = 5             # 与 LR2 对齐
    learning_rate: float = 0.5      # 与 LR2 对齐
    batch_size: int = 512           # 明文用；HE 模式中由 capacity 决定（见上）
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.001
    grad_clip_value: float = 1.0
    ovr: bool = True
    # HE 专用：单类过采样的目标正类占比，用于提升少数类召回
    he_target_pos_ratio: float = 0.6
    # HE 专用：是否用样本先验初始化偏置 b=log(p/(1-p))，加快正类收敛
    he_bias_init: bool = True

@dataclass
class EvalConfig:
    # 是否用阈值法产生最终预测（默认 False 与 LR2 一致；建议开以提升 1 类召回）
    use_threshold_for_prediction: bool = False
    threshold_search: bool = True
    threshold_min: float = -5.0
    threshold_max: float = 5.0
    threshold_points: int = 200
    compute_roc: bool = True
    # 新增：阈值优化目标
    threshold_objective: str = "fbeta"  # ["f1","recall","fbeta","youden"]
    fbeta_beta: float = 2.0              # beta>1 偏向召回
    min_precision: float = 0.1           # 可设置一个最低精度约束
    positive_class: int = 1              # 针对哪一类做阈值搜索（默认 1 类）
    
@dataclass
class SDKConfig:
    role: str = "local"
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    smote: SmoteConfig = field(default_factory=SmoteConfig)
    he: HEConfig = field(default_factory=HEConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_config(path: str) -> SDKConfig:
    raw = _load_yaml(path)
    return SDKConfig(
        role=raw.get("role", "local"),
        data=DataConfig(**raw.get("data", {})),
        preprocess=PreprocessConfig(**raw.get("preprocess", {})),
        smote=SmoteConfig(**raw.get("smote", {})),
        he=HEConfig(**raw.get("he", {})),
        train=TrainConfig(**raw.get("train", {})),
        eval=EvalConfig(**raw.get("eval", {})),
    )