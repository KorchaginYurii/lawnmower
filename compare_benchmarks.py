import os
import pandas as pd


LOG_DIR = "logs"

METRICS = [
    "success",
    "collect_rate",
    "energy_per_cabbage",
    "steps",
    "turns",
    "overlap_rate",
    "recharges",
    "sector_switches",
]


def list_csv_files(folder=LOG_DIR):
    files = [
        f for f in os.listdir(folder)
        if f.endswith(".csv") and "benchmark" in f
    ]

    files.sort(reverse=True)
    return files


def choose_file(title, files, folder=LOG_DIR):
    print(f"\n{title}")
    print("-" * 40)

    for i, f in enumerate(files):
        print(f"{i}: {f}")

    while True:
        try:
            idx = int(input("\nChoose file number: "))
            if 0 <= idx < len(files):
                return os.path.join(folder, files[idx])
        except ValueError:
            pass

        print("Invalid choice, try again.")


def summarize(path):
    df = pd.read_csv(path)

    summary = {}

    for m in METRICS:
        if m in df.columns:
            summary[m] = df[m].mean()

    return summary


def main():
    files = list_csv_files()

    if len(files) < 2:
        print("Need at least 2 benchmark CSV files in logs/")
        return

    old_path = choose_file("Choose OLD benchmark", files)
    new_path = choose_file("Choose NEW benchmark", files)

    old = summarize(old_path)
    new = summarize(new_path)

    print("\n========== BENCHMARK COMPARE ==========")
    print(f"OLD: { old_path }")
    print(f"NEW: { new_path }")

    print("\nmetric                 old        new        delta      delta%")
    print("-" * 62)

    for m in METRICS:
        if m not in old or m not in new:
            continue

        o = old[m]
        n = new[m]
        d = n - o

        pct = 0.0 if abs(o) < 1e-9 else d / abs(o) * 100

        print(
            f"{m:<20} "
            f"{o:>8.3f} "
            f"{n:>8.3f} "
            f"{d:>9.3f} "
            f"{pct:>8.2f}%"
        )


if __name__ == "__main__":
    main()