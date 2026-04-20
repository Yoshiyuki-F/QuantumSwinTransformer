from pathlib import Path
import sys

import jax.numpy as jnp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quantum_transformers.quantum_layer import get_circuit

# テスト用データ
x = jnp.ones((128, 8))  # 入力データ (128サンプル, 8特徴量)
w = jnp.ones((1, 8))    # 重み (1重みベクトル, 8特徴量)

# 回路の取得
qpred_batch = get_circuit()

# 結果を取得
output = qpred_batch(x, w)

print("出力の形状:", output.shape)  # (128, 8) が期待される
print("Type", type(output))
