import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

LOG_DIR = "logs"


def choose_file(folder=LOG_DIR):
    files = [
        f for f in os.listdir(folder)
        if f.endswith(".csv") and "lawn_benchmark" in f
    ]

    files.sort(reverse=True)

    if not files:
        print("No lawn benchmark CSV files found")
        return None

    print("\nAvailable lawn benchmark CSV:")
    for i, f in enumerate(files):
        print(f"{i}: {f}")

    idx = int(input("\nChoose file number: "))
    return os.path.join(folder, files[idx])

def list_csv_files(folder=LOG_DIR):
    files = [
        f for f in os.listdir(folder)
        if f.endswith(".csv") and "benchmark" in f
    ]

    files.sort(reverse=True)
    return files

def choose_compare_file(folder=LOG_DIR):
    files = list_csv_files(folder)

    if len(files) < 2:
        return None

    print("\nAvailable compare CSV:")
    for i, f in enumerate(files):
        print(f"{i}: {f}")

    raw = input(
        "\nChoose compare file "
        "(Enter to skip): "
    ).strip()

    if raw == "":
        return None

    idx = int(raw)

    return os.path.join(folder, files[idx])

def safe_mean(df, col, default=0.0):
    if col not in df.columns:
        return default
    return df[col].mean()

def print_top(df, title, sort_col, cols, ascending=False, n=5):
    if sort_col not in df.columns:
        return

    existing_cols = [c for c in cols if c in df.columns]

    print(f"\n========== {title} ==========")
    print(
        df.sort_values(sort_col, ascending=ascending)
        [existing_cols]
        .head(n)
        .to_string(index=False)
    )

def plot_overlap_histogram(df):

    if "overlap_rate" not in df.columns:
        return

    plt.figure(figsize=(8, 5))

    plt.hist(
        df["overlap_rate"],
        bins=20,
    )

    plt.title("Overlap Distribution")

    plt.xlabel("Overlap")
    plt.ylabel("Count")

    plt.grid(True)

def plot_energy_histogram(df):

    if "energy_per_m2" not in df.columns:
        return

    plt.figure(figsize=(8, 5))

    plt.hist(
        df["energy_per_m2"],
        bins=20,
    )

    plt.title("Energy per m2")

    plt.xlabel("Energy per m2")
    plt.ylabel("Count")

    plt.grid(True)

def plot_overlap_vs_energy(df):

    required = [
        "overlap_rate",
        "energy_per_m2",
    ]

    if not all(c in df.columns for c in required):
        return

    plt.figure(figsize=(8, 6))

    plt.scatter(
        df["overlap_rate"],
        df["energy_per_m2"],
    )

    plt.xlabel("Overlap")
    plt.ylabel("Energy per m2")

    plt.title(
        "Overlap vs Energy"
    )

    plt.grid(True)

def compare_dataframes(
    df_a,
    df_b,
    path_a=None,
    path_b=None,
):

    metrics = [
        "success",
        "coverage_rate",
        "overlap_rate",
        "energy_used",
        "energy_per_m2",
        "steps",
        "turns",
        "recharges",
    ]

    rows = []

    for m in metrics:

        if (
            m not in df_a.columns
            or m not in df_b.columns
        ):
            continue

        va = df_a[m].mean()
        vb = df_b[m].mean()

        rows.append({
            "metric": m,
            "A": round(va, 4),
            "B": round(vb, 4),
            "delta": round(vb - va, 4),
        })

    result = pd.DataFrame(rows)

    print("\n========== COMPARISON ==========")

    print(result.to_string(index=False))

    print("================================")

    # ==========================================
    # SAVE CSV
    # ==========================================

    out_dir = Path(LOG_DIR)
    out_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    name_a = Path(path_a).stem
    name_b = Path(path_b).stem

    out_path = out_dir / (
        f"compare_{name_a}_VS_{name_b}_{timestamp}.csv"
    )

    result.to_csv(
        out_path,
        index=False,
    )

    print(f"\n✅ comparison saved: {out_path}")

def main():
    path = choose_file()

    if path is None:
        return

    df = pd.read_csv(path)

    compare_path = choose_compare_file()

    compare_df = None

    if compare_path is not None:
        compare_df = pd.read_csv(compare_path)

    print("\n========== LAWN BENCHMARK SUMMARY ==========")
    print(f"file:               {path}")
    print(f"missions:           {len(df)}")
    print(f"success_rate:       {safe_mean(df, 'success'):.2f}")
    print(f"coverage_rate:      {safe_mean(df, 'coverage_rate'):.3f}")
    print(f"avg_energy_used:    {safe_mean(df, 'energy_used'):.2f}")
    print(f"energy_per_m2:      {safe_mean(df, 'energy_per_m2'):.3f}")
    print(f"avg_steps:          {safe_mean(df, 'steps'):.1f}")
    print(f"avg_turns:          {safe_mean(df, 'turns'):.1f}")
    print(f"avg_overlap:        {safe_mean(df, 'overlap_rate'):.3f}")
    print(f"avg_remaining:      {safe_mean(df, 'remaining_grass'):.1f}")
    print(f"avg_energy_left %:  {safe_mean(df, 'energy_percent_left'):.1f}")

    print_top(
        df,
        "BEST BY COVERAGE",
        "coverage_rate",
        [
            "seed",
            "success",
            "coverage_rate",
            "overlap_rate",
            "energy_used",
            "energy_per_m2",
            "steps",
            "turns",
            "remaining_grass",
            "fail_reason",
        ],
        ascending=False,
    )

    print_top(
        df,
        "WORST BY COVERAGE",
        "coverage_rate",
        [
            "seed",
            "success",
            "coverage_rate",
            "overlap_rate",
            "energy_used",
            "energy_per_m2",
            "steps",
            "turns",
            "remaining_grass",
            "fail_reason",
        ],
        ascending=True,
    )

    print_top(
        df,
        "WORST BY OVERLAP",
        "overlap_rate",
        [
            "seed",
            "success",
            "overlap_rate",
            "coverage_rate",
            "energy_used",
            "energy_per_m2",
            "steps",
            "turns",
            "fail_reason",
        ],
        ascending=False,
    )

    print_top(
        df,
        "WORST BY ENERGY/M2",
        "energy_per_m2",
        [
            "seed",
            "success",
            "energy_per_m2",
            "coverage_rate",
            "overlap_rate",
            "energy_used",
            "steps",
            "turns",
            "fail_reason",
        ],
        ascending=False,
    )

    print_top(
        df,
        "WORST BY STEPS",
        "steps",
        [
            "seed",
            "success",
            "steps",
            "coverage_rate",
            "overlap_rate",
            "energy_used",
            "turns",
            "remaining_grass",
            "fail_reason",
        ],
        ascending=False,
    )

    if "fail_reason" in df.columns:
        print("\n========== FAIL REASONS ==========")
        print(df["fail_reason"].value_counts().to_string())

    if "coverage_rate" in df.columns and "energy_used" in df.columns:
        print("\n========== EFFICIENCY ==========")
        df2 = df.copy()
        df2["coverage_per_energy"] = (
            df2["coverage_rate"] / df2["energy_used"].replace(0, 1e-9)
        )

        print(
            df2.sort_values(
                "coverage_per_energy",
                ascending=False
            )[[
                "seed",
                "coverage_per_energy",
                "coverage_rate",
                "energy_used",
                "overlap_rate",
                "steps",
                "fail_reason",
            ]].head(5)
            .to_string(index=False)
        )
    # ==========================================
    # COMPARE
    # ==========================================

    if compare_df is not None:
        compare_dataframes(
            df,
            compare_df,
            path,
            compare_path,
        )

    # ==========================================
    # PLOTS
    # ==========================================

    plot_overlap_histogram(df)

    plot_energy_histogram(df)

    plot_overlap_vs_energy(df)

    plt.show()

if __name__ == "__main__":
    main()