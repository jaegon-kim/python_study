import os
import pickle
import numpy as np
from numpy import genfromtxt

np.random.seed(220106)

# ====== 데이터 로드 ======
data = genfromtxt('IRIS_tiny_onehot.csv', delimiter=',', skip_header=1)

X = data[:, 0:4].astype(np.float32)   # input
Y = data[:, 4:7].astype(np.float32)   # one-hot target (3 classes)

# ====== 파라미터 초기화 ======
W = np.random.randn(3, 4).astype(np.float32)
B = np.random.randn(3, 1).astype(np.float32)

learning_rate = 0.001


def softmax_forward(X, Y, W, B):
    arr_pred = []
    arr_loss = []

    for i in range(len(X)):
        x = X[i].reshape(-1, 1)       # (4,1)
        y = Y[i].reshape(1, -1)       # (1,3)

        logits = (W @ x) + B          # (3,1)

        # 안정성 위해 max 빼기(overflow 방지)
        logits = logits - np.max(logits)

        exp_logits = np.exp(logits)   # (3,1)
        soft = exp_logits / np.sum(exp_logits)  # (3,1)
        soft = soft.reshape(1, -1)    # (1,3)

        arr_pred.append(soft)

        y_target = np.sum(soft * y, axis=1, keepdims=True)  # (1,1)
        loss = -np.log(y_target + 1e-12)  # log(0) 방지
        arr_loss.append(loss)

    preds = np.array(arr_pred)            # (N,1,3)
    losses = float(np.sum(np.array(arr_loss)))
    return preds, losses


def loss_gradient(X, Y, W, B):
    add_dL_dW = np.zeros((3, 4), dtype=np.float32)
    add_dL_dB = np.zeros((3, 1), dtype=np.float32)

    for i in range(len(X)):
        x = X[i].reshape(-1, 1)      # (4,1)
        y = Y[i].reshape(1, -1)      # (1,3)

        logits = (W @ x) + B         # (3,1)
        logits = logits - np.max(logits)

        exp_logits = np.exp(logits)
        s = exp_logits / np.sum(exp_logits)  # (3,1)

        # softmax를 (1,3)로
        softmax = s.reshape(1, -1)

        y_target = np.sum(softmax * y, axis=1, keepdims=True)  # (1,1)

        # dL/dsmax
        dL_dsmax = -1.0 / (y_target + 1e-12)  # (1,1)

        s1, s2, s3 = s[0, 0], s[1, 0], s[2, 0]

        dsmax_dg_matrix = np.array(
            [
                [s1 * (1 - s1), -(s1 * s2),    -(s1 * s3)],
                [-(s1 * s2),    s2 * (1 - s2), -(s2 * s3)],
                [-(s1 * s3),    -(s2 * s3),    s3 * (1 - s3)],
            ],
            dtype=np.float32
        )  # (3,3)

        # y: (1,3) -> (3,1)
        dsmax_dg = dsmax_dg_matrix @ y.T      # (3,1)

        dg_dW = x.T                            # (1,4)
        dsmax_dW = dsmax_dg @ dg_dW            # (3,4)

        dL_dW = float(dL_dsmax) * dsmax_dW     # (3,4)
        add_dL_dW += dL_dW

        dL_dB = float(dL_dsmax) * dsmax_dg     # (3,1)
        add_dL_dB += dL_dB

    return add_dL_dW, add_dL_dB


# ====== 학습 전 ======
pred, loss = softmax_forward(X, Y, W, B)
print('before loss:', loss)

# ====== 학습 ======
epochs = 100
for _ in range(epochs):
    dW, dB = loss_gradient(X, Y, W, B)
    W -= learning_rate * dW
    B -= learning_rate * dB

# ====== 학습 후 ======
pred, loss = softmax_forward(X, Y, W, B)
print('after loss:', loss)

# ====== 모델 저장 (PKL) ======
# KServe로 서빙할 파일 경로 (네가 쓰는 modelhub 디렉토리에 맞추면 됨)
MODEL_DIR = os.environ.get("MODEL_DIR", "./modelhub/iris-softmax")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

artifact = {
    "type": "softmax_linear_numpy",
    "input_dim": 4,
    "num_classes": 3,
    "W": W,   # (3,4)
    "B": B,   # (3,1)
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(artifact, f)

print(f"[OK] saved: {MODEL_PATH}")
