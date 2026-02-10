"""
Generate cosmologies for training and testing using Latin Hypercube Sampling.
"""
import argparse
import pandas as pd
from pathlib import Path
from DUMP.data.bacco_Pk import sample_cosmologies


def main():
    parser = argparse.ArgumentParser(description="Generate cosmology samples")
    parser.add_argument("--n_train", type=int, default=180_000, help="Number of train samples")
    parser.add_argument("--n_val", type=int, default=20_000, help="Number of val samples")
    parser.add_argument("--n_test", type=int, default=100_000, help="Number of test samples")
    parser.add_argument("--seed_train", type=int, default=42, help="Random seed for train set")
    parser.add_argument("--seed_val", type=int, default=13, help="Random seed for val set")
    parser.add_argument("--seed_test", type=int, default=123, help="Random seed for test set")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory")

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate train cosmologies
    train_cosmologies = sample_cosmologies(
        n_samples=args.n_train,
        random_seed=args.seed_train,
        train=True
    )
    train_df = pd.DataFrame(train_cosmologies)
    train_file = output_path / "cosmologies_train.csv"
    train_df.to_csv(train_file, index=False)
    print(f"{len(train_df)} train cosmologies were generated and saved to {train_file}")

    # Generate val cosmologies
    val_cosmologies = sample_cosmologies(
        n_samples=args.n_val,
        random_seed=args.seed_val,
        test=True
    )
    val_df = pd.DataFrame(val_cosmologies)
    val_file = output_path / "cosmologies_val.csv"
    val_df.to_csv(val_file, index=False)
    print(f"{len(val_df)} train cosmologies were generated and saved to {val_file}")

    # Generate test cosmologies
    test_cosmologies = sample_cosmologies(
        n_samples=args.n_test,
        random_seed=args.seed_test,
        test=True
    )
    test_df = pd.DataFrame(test_cosmologies)
    test_file = output_path / "cosmologies_test.csv"
    test_df.to_csv(test_file, index=False)
    print(f"{len(test_df)} train cosmologies were generated and saved to {test_file}")


if __name__ == "__main__":
    main()
