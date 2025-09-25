from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np
import tenseal as ts
from tqdm import tqdm  # 进度条

from .config import SDKConfig, HEConfig
from .encryption import CKKSEncryptor
from .model import EncryptedLR, train_plain_ovr, AdamWParams
from .data import _read_any, preprocess_dataframe, split_and_scale, maybe_smote
from .eval import evaluate_classifier

class SecureHEPipeline:
    def __init__(self, cfg: SDKConfig):
        self.cfg = cfg
        self.feature_names: List[str] = []
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None
        self.scaler = None
        self.weights_list: List[np.ndarray] = []
        self.encryptor: CKKSEncryptor | None = None

    def load_and_preprocess(self):
        df = _read_any(self.cfg.data.input_path, self.cfg.data.file_type)
        X, y, feature_names = preprocess_dataframe(
            df=df,
            target=self.cfg.data.target,
            drop_cols=self.cfg.preprocess.drop_cols,
            binary_map=self.cfg.preprocess.binary_map,
            target_map=self.cfg.preprocess.target_map,
            unknown_fill_cols=self.cfg.preprocess.unknown_fill_cols,
            extreme_clip_cols=self.cfg.preprocess.extreme_clip_cols,
            clip_p=self.cfg.preprocess.clip_p,
            month_col=self.cfg.preprocess.month_col,
            month_map=self.cfg.preprocess.month_map,
            engineered_flags=self.cfg.preprocess.engineered if self.cfg.preprocess.add_engineered_features else {},
            onehot_like_cols=self.cfg.preprocess.onehot_like_cols,
        )
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(
            X, y,
            test_size=self.cfg.data.test_size,
            val_size=self.cfg.data.val_size,
            random_state=self.cfg.data.random_state,
            stratify=self.cfg.data.stratify,
            feature_range=tuple(self.cfg.preprocess.scale_feature_range),
        )
        X_train, y_train = maybe_smote(
            X_train, y_train,
            enabled=self.cfg.smote.enabled,
            minority_ratio=self.cfg.smote.minority_ratio,
            k_neighbors_cap=self.cfg.smote.k_neighbors_cap,
            random_state=self.cfg.data.random_state,
        )
        self.feature_names = feature_names
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        self.scaler = scaler

    def train_plain(self):
        classes_ = np.unique(self.y_train)
        n_classes = len(classes_)
        adamw = AdamWParams(
            beta1=self.cfg.train.beta1,
            beta2=self.cfg.train.beta2,
            epsilon=self.cfg.train.epsilon,
            weight_decay=self.cfg.train.weight_decay,
            grad_clip_value=self.cfg.train.grad_clip_value,
        )
        self.weights_list = train_plain_ovr(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            n_classes=n_classes,
            iterations=self.cfg.train.iterations,
            learning_rate=self.cfg.train.learning_rate,
            adamw_params=adamw,
        )
        return classes_

    def _choose_capacity(self, he: HEConfig, n_features: int) -> int:
        slots = he.poly_modulus_degree // 2  # CKKS 可用槽位
        max_cap = max(1, slots // n_features)
        capacity = min(he.batch_size_encrypt, max_cap) if he.auto_capacity else he.batch_size_encrypt
        # 运行时打印：这是“实际生效的加密批大小”
        print(f"[HE] chosen capacity (effective encrypted batch size) = {capacity} "
              f"(requested={he.batch_size_encrypt}, slots={slots}, n_features={n_features}, auto_capacity={he.auto_capacity})")
        return capacity

    def _oversample_for_class(self, X: np.ndarray, y_bin: np.ndarray, target_pos_ratio=0.6) -> Tuple[np.ndarray, np.ndarray]:
        """明文维度上对正类过采样到目标比例，避免在密文中乘权。"""
        pos_idx = np.where(y_bin == 1)[0]
        neg_idx = np.where(y_bin == 0)[0]
        n_pos, n_neg = len(pos_idx), len(neg_idx)
        if n_pos == 0:
            return X[neg_idx], y_bin[neg_idx]
        cur_ratio = n_pos / (n_pos + n_neg)
        if cur_ratio >= target_pos_ratio:
            all_idx = np.concatenate([pos_idx, neg_idx])
            np.random.shuffle(all_idx)
            return X[all_idx], y_bin[all_idx]
        needed_pos = int(target_pos_ratio / (1 - target_pos_ratio) * n_neg)
        repeat = max(1, int(np.ceil(needed_pos / n_pos)))
        pos_rep = np.tile(pos_idx, repeat)[:needed_pos]
        new_idx = np.concatenate([pos_rep, neg_idx])
        np.random.shuffle(new_idx)
        return X[new_idx], y_bin[new_idx]

    def train_he(self):
        he: HEConfig = self.cfg.he
        classes_ = np.unique(self.y_train)
        n_classes = len(classes_)
        self.encryptor = CKKSEncryptor(
            poly_modulus_degree=he.poly_modulus_degree,
            coeff_mod_bit_sizes=he.coeff_mod_bit_sizes,
            global_scale_bits=he.global_scale_bits,
        )
        context = self.encryptor.context
        n_features = self.X_train.shape[1]
        capacity = self._choose_capacity(he, n_features)
        if capacity < 1:
            raise ValueError("capacity < 1, 请增大 poly_modulus_degree 或减少特征/批大小。")

        weights_list = []
        # 先预构建每类的密文批，计算总步数用于进度条
        total_batches_per_class: List[int] = []

        per_class_encrypted_data: List[Tuple[List[Tuple[ts.CKKSVector, int]], List[ts.CKKSVector], float]] = []
        # ^ 每项：(encrypted_X_batches, encrypted_y_batches, init_bias)
        for class_idx in range(n_classes):
            y_bin = (self.y_train == classes_[class_idx]).astype(int)
            X_cls, y_cls = self._oversample_for_class(self.X_train, y_bin, target_pos_ratio=self.cfg.train.he_target_pos_ratio)

            # 初始化偏置（先验）
            init_p = float(np.clip(np.mean(y_cls), 1e-6, 1 - 1e-6))
            init_bias = np.log(init_p / (1 - init_p)) if self.cfg.train.he_bias_init else 0.0

            encrypted_X_batches: List[Tuple[ts.CKKSVector, int]] = []
            encrypted_y_batches: List[ts.CKKSVector] = []
            for i in range(0, len(X_cls), capacity):
                Xb = X_cls[i: i + capacity]
                yb = y_cls[i: i + capacity]
                actual_size = Xb.shape[0]
                flat = Xb.reshape(-1).tolist()
                if actual_size < capacity:
                    pad = [0.0] * ((capacity - actual_size) * n_features)
                    flat += pad
                enc_X = ts.ckks_vector(context, flat)
                encrypted_X_batches.append((enc_X, actual_size))
                arr = np.zeros(capacity, dtype=float)
                arr[:actual_size] = yb
                enc_y = ts.ckks_vector(context, arr.tolist())
                encrypted_y_batches.append(enc_y)

            total_batches_per_class.append(len(encrypted_X_batches))
            per_class_encrypted_data.append((encrypted_X_batches, encrypted_y_batches, init_bias))

        total_steps = int(self.cfg.train.iterations * sum(total_batches_per_class))
        pbar = tqdm(total=total_steps, desc=f"HE training (classes={n_classes}, iters={self.cfg.train.iterations})", unit="batch")

        for class_idx in range(n_classes):
            encrypted_X_batches, encrypted_y_batches, init_bias = per_class_encrypted_data[class_idx]

            eelr = EncryptedLR(
                n_features=n_features,
                context=context,
                batch_size=capacity,
                noise_std=he.add_noise_std,
                noise_std_update=he.add_noise_std_update,
            )
            eelr.bias = init_bias  # 先验初始化，帮助正类召回

            for epoch in range(self.cfg.train.iterations):
                order = np.random.permutation(len(encrypted_X_batches))
                for b in order:
                    enc_x_batch, actual_size = encrypted_X_batches[b]
                    enc_y_batch = encrypted_y_batches[b]
                    enc_out = eelr.forward_batch(enc_x_batch, actual_size)
                    eelr.backward_batch(enc_x_batch, enc_out, enc_y_batch, actual_size)
                    eelr.update_parameters(self.cfg.train.learning_rate)
                    pbar.update(1)
                    pbar.set_postfix({"class": int(classes_[class_idx]), "epoch": epoch + 1, "batch": int(b) + 1})

            weights_list.append(np.array(eelr.weight, dtype=float))

        pbar.close()
        self.weights_list = weights_list
        return classes_

    def evaluate(self, classes_):
        thr_range = (self.cfg.eval.threshold_min, self.cfg.eval.threshold_max, self.cfg.eval.threshold_points)
        metrics = evaluate_classifier(
            X_test=self.X_test,
            y_test=self.y_test,
            weights_list=self.weights_list,
            classes_=classes_,
            threshold_search=self.cfg.eval.threshold_search,
            threshold_range=thr_range,
            compute_roc=self.cfg.eval.compute_roc,
            use_threshold_for_prediction=self.cfg.eval.use_threshold_for_prediction,
            threshold_objective=self.cfg.eval.threshold_objective,
            fbeta_beta=self.cfg.eval.fbeta_beta,
            min_precision=self.cfg.eval.min_precision,
            positive_class=self.cfg.eval.positive_class,
        )
        return metrics

    def run_all(self) -> Dict[str, Any]:
        self.load_and_preprocess()
        if self.cfg.he.enabled:
            classes_ = self.train_he()
        else:
            classes_ = self.train_plain()
        metrics = self.evaluate(classes_)
        return {
            "feature_names": self.feature_names,
            "classes": classes_.tolist(),
            "weights_list": [w.tolist() for w in self.weights_list],
            "metrics": metrics,
        }