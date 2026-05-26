import os
import pickle
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.rl_high_level_policy import RLPolicyNet


DATASET_PATH = "datasets/high_level_policy_dataset.pkl"
SAVE_DIR = "checkpoints/rl_high_level"

STATE_DIM = 7
ACTION_DIM = 5

EPOCHS = 30
BATCH_SIZE = 256
LR = 1e-3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_dataset(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    states = np.array(
        [x[0] for x in data],
        dtype=np.float32,
    )

    actions = np.array(
        [x[1] for x in data],
        dtype=np.int64,
    )

    return states, actions


def train():
    states, actions = load_dataset(DATASET_PATH)

    print(f"dataset size: {len(states)}")

    counts = np.bincount(
        actions,
        minlength=ACTION_DIM,
    )

    print("action counts:", counts)

    x = torch.tensor(
        states,
        dtype=torch.float32,
        device=DEVICE,
    )

    y = torch.tensor(
        actions,
        dtype=torch.long,
        device=DEVICE,
    )

    model = RLPolicyNet(
        STATE_DIM,
        ACTION_DIM,
    ).to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LR,
    )

    loss_fn = nn.CrossEntropyLoss()

    n = len(x)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        perm = torch.randperm(
            n,
            device=DEVICE,
        )

        total_loss = 0.0
        correct = 0
        total = 0

        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]

            xb = x[idx]
            yb = y[idx]

            logits = model(xb)

            loss = loss_fn(
                logits,
                yb,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)

            pred = torch.argmax(
                logits,
                dim=1,
            )

            correct += (
                pred == yb
            ).sum().item()

            total += len(xb)

        avg_loss = total_loss / max(1, total)
        acc = correct / max(1, total)

        print(
            f"epoch {epoch + 1:03d} | "
            f"loss={avg_loss:.5f} | "
            f"acc={acc:.3f}"
        )

        if acc > best_acc:
            best_acc = acc

            os.makedirs(
                SAVE_DIR,
                exist_ok=True,
            )

            path = os.path.join(
                SAVE_DIR,
                "rl_high_level_imitation_best.pth",
            )

            torch.save(
                model.state_dict(),
                path,
            )

            print(
                f"✅ saved best: {path}"
            )

    ts = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    final_path = os.path.join(
        SAVE_DIR,
        f"rl_high_level_imitation_final_{ts}.pth",
    )

    torch.save(
        model.state_dict(),
        final_path,
    )

    print(
        f"\n✅ final saved: {final_path}"
    )
    print(
        f"best acc: {best_acc:.3f}"
    )


if __name__ == "__main__":
    train()