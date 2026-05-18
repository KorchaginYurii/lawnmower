import os
import torch
import shutil
from heapq import nsmallest

import os
import torch
import shutil

def detect_base_dir(project_name="Cab2"):
    """
    Автоматически выбирает папку для checkpoint'ов:
    - Kaggle: /kaggle/working/<project_name>/checkpoints
    - Local/PyCharm: <папка проекта>/checkpoints
    """

    # Kaggle
    if os.path.exists("/kaggle/working"):
        return os.path.join("/kaggle/working", project_name)

    # Local / PyCharm
    return os.getcwd()


class CheckpointManager:
    def __init__(self, k_best=3, folder=None, project_name="Cab2"):
        self.k_best = k_best

        base_dir = detect_base_dir(project_name)

        if folder is None:
            self.folder = os.path.join(base_dir, "checkpoints")
        else:
            self.folder = folder

        self.history_folder = os.path.join(self.folder, "history")
        self.last_path = os.path.join(self.folder, "last.pth")
        self.backup_path = os.path.join(self.folder, "backup.pth")

        self.best_models = []

        os.makedirs(self.folder, exist_ok=True)
        os.makedirs(self.history_folder, exist_ok=True)

        print(f"📦 Checkpoints folder: {self.folder}")

    def _save(self, path, data):
        torch.save(data, path)

    def _make_data(self, agent, ep, best_score):
        return {
            "model": agent.net.state_dict(),
            "optimizer": agent.opt.state_dict(),
            "scaler": agent.scaler.state_dict() if hasattr(agent, "scaler") else None,
            "episode": ep,
            "best": best_score,
        }

    def save_last(self, agent, ep, best_score):
        data = self._make_data(agent, ep, best_score)
        self._save(self.last_path, data)

    def save(self, agent, ep, best_score, eval_score):
        data = self._make_data(agent, ep, best_score)

        # backup предыдущего last
        if os.path.exists(self.last_path):
            shutil.copy(self.last_path, self.backup_path)

        # last
        self._save(self.last_path, data)

        # history
        if ep % 50 == 0:
            hist_path = os.path.join(self.history_folder, f"ep_{ep}.pth")
            self._save(hist_path, data)

        # best
        self._update_best(data, eval_score, ep)

    def _update_best(self, data, score, ep):
        path = os.path.join(
            self.folder,
            f"best_ep{ep}_score{score:.2f}.pth"
        )

        self.best_models.append((score, path))
        self.best_models.sort(reverse=True, key=lambda x: x[0])
        self.best_models = self.best_models[:self.k_best]

        keep = []

        for s, p in self.best_models:
            keep.append(os.path.basename(p))
            if not os.path.exists(p):
                torch.save(data, p)
                print(f"\n🔥 Saved best: {p}")

        # удалить лишние best-файлы
        for f in os.listdir(self.folder):
            if f.startswith("best_") and f not in keep:
                os.remove(os.path.join(self.folder, f))

    def load_checkpoint(self, agent):
        candidates = [
            os.path.join(self.folder, "last.pth"),
            os.path.join(self.folder, "backup.pth"),
        ]

        for path in candidates:
            if not os.path.exists(path):
                continue

            name = os.path.basename(path)
            print(f"🔄 Loading {name}...")

            try:
                checkpoint = torch.load(
                    path,
                    map_location=agent.device,
                    weights_only=False
                )

                agent.net.load_state_dict(checkpoint["model"])
                agent.opt.load_state_dict(checkpoint["optimizer"])

                if checkpoint.get("scaler") is not None and hasattr(agent, "scaler"):
                    agent.scaler.load_state_dict(checkpoint["scaler"])

                ep = checkpoint.get("episode", 0)
                best = checkpoint.get("best", -1e9)

                print(f"✅ Loaded {name} (ep {ep})")
                return ep + 1, best

            except Exception as e:
                print(f"❌ Failed to load {name}: {e}")

        print("⚠️ Starting from scratch")
        return 0, -1e9