from __future__ import annotations
import numpy as np
import tenseal as ts
import contextlib
import sys, os

class CKKSEncryptor:
    def __init__(self, context=None, public_context_bytes: bytes = None,
                 poly_modulus_degree=8192, coeff_mod_bit_sizes=None, global_scale_bits=21):
        self.encryption_count = 0
        if coeff_mod_bit_sizes is None:
            coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 40]
        if context is not None:
            self.context = context
        elif public_context_bytes is not None:
            self.context = ts.context_from(public_context_bytes)
        else:
            self.context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=poly_modulus_degree,
                coeff_mod_bit_sizes=coeff_mod_bit_sizes,
            )
            self.context.generate_galois_keys()
            self.context.global_scale = 2 ** global_scale_bits

    def batch_encrypt(self, vectors: np.ndarray):
        batch_size = len(vectors)
        flattened = np.asarray(vectors).reshape(-1).tolist()
        encrypted = ts.ckks_vector(self.context, flattened)
        self.encryption_count += batch_size
        return encrypted, batch_size

    def encrypt_vector(self, vector: np.ndarray):
        return ts.ckks_vector(self.context, np.asarray(vector).tolist())

    def encrypt_label(self, label):
        return ts.ckks_vector(self.context, [float(label)])

    def encrypt_matrix(self, matrix: np.ndarray):
        flattened = np.asarray(matrix).reshape(-1).tolist()
        return ts.ckks_vector(self.context, flattened)

    def decrypt_vector(self, ciphertext):
        return ciphertext.decrypt()

    def public_context_bytes(self, save_galois_keys=True):
        return self.context.serialize(
            save_public_key=True,
            save_secret_key=False,
            save_galois_keys=save_galois_keys
        )

@contextlib.contextmanager
def suppress_tenseal_warnings(enabled: bool):
    """
    简易 stderr 重定向，抑制 TenSEAL 在控制台的告警输出。
    注意：这会屏蔽被包裹代码块内的所有 stderr 输出，请谨慎使用。
    """
    if not enabled:
        yield
        return
    old_stderr = sys.stderr
    try:
        with open(os.devnull, "w") as devnull:
            sys.stderr = devnull
            yield
    finally:
        sys.stderr = old_stderr