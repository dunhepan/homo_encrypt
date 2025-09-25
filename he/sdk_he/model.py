from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List
import tenseal as ts
from sklearn.metrics import recall_score

@dataclass
class AdamWParams:
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.001
    grad_clip_value: float = 1.0

def train_plain_ovr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    iterations: int,
    learning_rate: float,
    adamw_params: AdamWParams,
) -> List[np.ndarray]:
    weights_list: List[np.ndarray] = []
    for class_idx in range(n_classes):
        y_train_bin = (y_train == class_idx).astype(int)
        y_val_bin = (y_val == class_idx).astype(int)
        n_pos = np.sum(y_train_bin); n_neg = len(y_train_bin) - n_pos
        pos_weight = min(30.0, n_neg / (n_pos + 1e-5))
        w = np.random.normal(0, 0.01, X_train.shape[1])
        m = np.zeros_like(w); v = np.zeros_like(w)
        best_val_recall, best_w = -1.0, w.copy()
        eval_interval = max(5, min(10, iterations // 20))
        warmup_epochs = min(20, iterations // 2)
        for epoch in range(iterations):
            current_lr = learning_rate * (min(1.0, (epoch + 1) / warmup_epochs) if epoch < warmup_epochs else 1.0)
            logits = X_train @ w
            preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
            errors = preds - y_train_bin
            weighted_errors = np.where(y_train_bin == 1, errors * pos_weight, errors)
            grad = (X_train.T @ weighted_errors) / len(y_train_bin)
            m = adamw_params.beta1 * m + (1 - adamw_params.beta1) * grad
            v = adamw_params.beta2 * v + (1 - adamw_params.beta2) * (grad ** 2)
            m_hat = m / (1 - adamw_params.beta1 ** (epoch + 1))
            v_hat = v / (1 - adamw_params.beta2 ** (epoch + 1))
            update = m_hat / (np.sqrt(v_hat) + adamw_params.epsilon)
            update = np.clip(update, -adamw_params.grad_clip_value, adamw_params.grad_clip_value)
            w = (1 - current_lr * adamw_params.weight_decay) * w - current_lr * update
            if (epoch % eval_interval == 0) or (epoch == iterations - 1):
                val_logits = X_val @ w
                val_preds = (val_logits > 0).astype(int)
                val_recall = recall_score(y_val_bin, val_preds, zero_division=0)
                if val_recall > best_val_recall:
                    best_val_recall, best_w = val_recall, w.copy()
        weights_list.append(best_w)
    return weights_list

class EncryptedLR:
    """
    HE 逻辑回归（批零填充 + 线性近似 sigmoid）。
    - 不在密文中做正类加权，避免乘法深度；正类权重通过“明文过采样”实现。
    - 每批 backward 后立即 update。
    """
    def __init__(self, n_features, context, batch_size=4096,
                 noise_std=0.1, noise_std_update=0.05):
        self.weight = [0.0] * n_features
        self.bias = 0.0
        self._delta_w = None
        self._delta_b = None
        self._count = 0
        self.n_features = n_features
        self.context = context
        self.batch_size = batch_size
        self.noise_std = noise_std
        self.noise_std_update = noise_std_update

    def sigmoid_batch(self, enc_x: ts.CKKSVector):
        # 线性近似，乘法深度低
        return enc_x.polyval([0.5, 0.2])

    def forward_batch(self, enc_x_batch: ts.CKKSVector, actual_size: int):
        replicated_weights = np.tile(self.weight, (self.batch_size, 1)).flatten().tolist()
        enc_weight_batch = ts.ckks_vector(self.context, replicated_weights)
        enc_out = enc_x_batch.dot(enc_weight_batch)
        bias_vec = [self.bias] * actual_size + [0.0] * (self.batch_size - actual_size)
        enc_bias = ts.ckks_vector(self.context, bias_vec)
        enc_out = enc_out + enc_bias
        return self.sigmoid_batch(enc_out)

    def backward_batch(self, enc_x_batch: ts.CKKSVector, enc_out_batch: ts.CKKSVector,
                       enc_y_batch: ts.CKKSVector, actual_size: int):
        enc_error = enc_out_batch - enc_y_batch
        noise = np.zeros(self.batch_size, dtype=float)
        if actual_size > 0:
            noise[:actual_size] = np.random.normal(0, self.noise_std, size=actual_size)
        enc_noise = ts.ckks_vector(self.context, noise.tolist())
        enc_noisy_error = enc_error + enc_noise
        repeated_noisy_error = []
        for err in enc_noisy_error.decrypt():
            repeated_noisy_error.extend([err] * self.n_features)
        enc_repeated_noisy_error = ts.ckks_vector(self.context, repeated_noisy_error)
        gradient = enc_x_batch * enc_repeated_noisy_error
        if self._delta_w is None:
            max_len = self.batch_size * self.n_features
            self._delta_w = ts.ckks_vector(self.context, [0.0] * max_len)
            self._delta_b = ts.ckks_vector(self.context, [0.0] * self.batch_size)
        self._delta_w += gradient
        self._delta_b += enc_noisy_error
        self._count += actual_size

    def update_parameters(self, learning_rate: float):
        if self._count == 0:
            return
        dec_delta_w = self._delta_w.decrypt()
        dec_delta_b = self._delta_b.decrypt()
        used_w = self._count * self.n_features
        used_b = self._count
        delta_w_matrix = np.array(dec_delta_w[:used_w]).reshape(self._count, self.n_features)
        delta_b_vector = np.array(dec_delta_b[:used_b])
        avg_delta_w = np.mean(delta_w_matrix, axis=0)
        avg_delta_b = np.mean(delta_b_vector)
        weight_noise = np.random.normal(0, self.noise_std_update, size=len(avg_delta_w))
        bias_noise = np.random.normal(0, self.noise_std_update)
        avg_delta_w += weight_noise
        avg_delta_b += bias_noise
        reg_strength = 0.1 + 0.05 * np.log(self.n_features + 1.0)
        weight_array = np.array(self.weight)
        weight_array -= learning_rate * (avg_delta_w + weight_array * reg_strength)
        self.weight = weight_array.tolist()
        self.bias -= learning_rate * avg_delta_b
        self._delta_w = None
        self._delta_b = None
        self._count = 0