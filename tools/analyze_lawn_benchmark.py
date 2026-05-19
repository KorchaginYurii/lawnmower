import os
import pandas as pd


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


def main():
    path = choose_file()

    if path is None:
        return

    df = pd.read_csv(path)

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


if __name__ == "__main__":
    main()