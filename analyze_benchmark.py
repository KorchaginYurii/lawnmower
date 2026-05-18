import pandas as pd
import os




def choose_file(folder="logs"):
    files = [
        f for f in os.listdir(folder)
        if f.endswith(".csv") and "benchmark" in f
    ]

    files.sort(reverse=True)

    if not files:
        print("No CSV files found")
        return None

    print("\nAvailable .csv:")
    for i, f in enumerate(files):
        print(f"{i}: {f}")

    idx = int(input("\nChoose file number: "))
    return os.path.join(folder, files[idx])


def main():
    path = choose_file()

    df = pd.read_csv(path)

    print("\n========== SUMMARY ==========")
    print(f"missions:        {len(df)}")
    print(f"success_rate:    {df['success'].mean():.2f}")
    print(f"collect_rate:    {df['collect_rate'].mean():.3f}")
    print(f"energy/cabbage:  {df['energy_per_cabbage'].mean():.3f}")
    print(f"avg_steps:       {df['steps'].mean():.1f}")
    print(f"avg_turns:       {df['turns'].mean():.1f}")
    print(f"avg_overlap:     {df['overlap_rate'].mean():.3f}")
    print(f"avg_recharges:   {df['recharges'].mean():.1f}")

    print("\n========== WORST BY ENERGY/CABBAGE ==========")
    print(
        df.sort_values("energy_per_cabbage", ascending=False)
        [["seed", "success", "energy_per_cabbage", "steps", "turns", "overlap_rate", "recharges"]]
        .head(5)
        .to_string(index=False)
    )

    print("\n========== WORST BY OVERLAP ==========")
    print(
        df.sort_values("overlap_rate", ascending=False)
        [["seed", "success", "overlap_rate", "energy_per_cabbage", "steps", "turns", "recharges"]]
        .head(5)
        .to_string(index=False)
    )

    print("\n========== WORST BY STEPS ==========")
    print(
        df.sort_values("steps", ascending=False)
        [["seed", "success", "steps", "turns", "energy_per_cabbage", "overlap_rate", "recharges"]]
        .head(5)
        .to_string(index=False)
    )

    if "fail_reason" in df.columns:
        print("\n========== FAIL REASONS ==========")
        print(df["fail_reason"].value_counts())


if __name__ == "__main__":
    main()