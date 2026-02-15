"""
Generate cosmologies for training and testing using Latin Hypercube Sampling.
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from DUMP.data.bacco_Pk import sample_cosmologies
from DUMP.data.constants import bacco_test_ranges


def main():
    parser = argparse.ArgumentParser(description="Generate cosmology samples")
    parser.add_argument("--n_control", type=int, default=10, help="Number of control samples")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for control set")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory")

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate control cosmologies
    control_cosmologies = sample_cosmologies(
        n_samples=args.n_control,
        random_seed=args.seed,
        test=True
    )

    # Rescale w0 and wa to restricted ranges:
    # w0: from [-1.15, -0.85] to [-1, -0.85] (w0 > -1)
    # wa: from [-0.3, 0.3] to [-0.3, 0] (wa < 0)
    # w0
    w0_orig_low, w0_orig_high = bacco_test_ranges["w0"]
    w0_new_low = -1.0
    w0_normalized = (np.array(control_cosmologies["w0"]) - w0_orig_low) / (w0_orig_high - w0_orig_low)
    control_cosmologies["w0"] = w0_new_low + (w0_orig_high - w0_new_low) * w0_normalized

    # wa
    wa_orig_low, wa_orig_high = bacco_test_ranges["wa"]
    wa_new_high = 0.0
    wa_normalized = (np.array(control_cosmologies["wa"]) - wa_orig_low) / (wa_orig_high - wa_orig_low)
    control_cosmologies["wa"] = wa_orig_low + (wa_new_high - wa_orig_low) * wa_normalized

    control_df = pd.DataFrame(control_cosmologies)
    control_file = output_path / "cosmologies_control.csv"
    control_df.to_csv(control_file, index=False)
    print(f"{len(control_df)} control cosmologies were generated and saved to {control_file}")

if __name__ == "__main__":
    main()
