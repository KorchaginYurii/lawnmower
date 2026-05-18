import csv
import os


class MetricsLogger:
    def __init__(self, path="logs/mission_log.csv"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

        self.header = [
            "episode",
            "success",
            "fail_reason",
            "reward",
            "collected",
            "total_cabbages",
            "steps",
            "energy_used",
            "energy_per_cabbage",
            "total_turns",
            "overlap_rate",
            "sector_switches",
        ]

        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.header)

    def log(self, **kwargs):
        row = [kwargs.get(k, None) for k in self.header]

        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)