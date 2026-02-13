"""
Generate w0waCDM control cosmologies for training upweighting.
Samples from test ranges then rescales to: w0 ∈ [-1.0, -0.85], wa ∈ [-0.3, 0.0]
"""
import argparse
import pandas as pd
from pathlib import Path
from DUMP.data.bacco_Pk import sample_cosmologies


def main():
    parser = argparse.ArgumentParser(description="Generate w0waCDM control samples")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of w0waCDM samples")
    parser.add_argument("--seed", type=int, default=999, help="Random seed")
    parser.add_argument("--output", type=str, default="./data/cosmologies_train_control.csv",
                        help="Output file path")

    args = parser.parse_args()

    # Generate w0waCDM cosmologies using test ranges
    cosmologies = sample_cosmologies(
        n_samples=args.n_samples,
        random_seed=args.seed,
        test=True  # Uses w0waCDM ranges
    )

    df = pd.DataFrame(cosmologies)

    # w0: [-1.15, -0.85] → [-1.0, -0.85], wa: [-0.3, 0.3] → [-0.3, 0.0]
    df['w0'] = -1.0 + (df['w0'] + 1.15) / 0.3 * 0.15
    df['wa'] = -0.3 + (df['wa'] + 0.3) / 0.6 * 0.3

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"{len(df)} w0waCDM control cosmologies were generated and saved to {output_path}")
    print("DE params ranges:")
    print(f"w0 ∈ [{df['w0'].min():.3f}, {df['w0'].max():.3f}]")
    print(f"wa ∈ [{df['wa'].min():.3f}, {df['wa'].max():.3f}]")


if __name__ == "__main__":
    main()
