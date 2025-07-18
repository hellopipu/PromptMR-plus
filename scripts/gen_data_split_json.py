#!/usr/bin/env python3

import os
import json
import random
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Split .mat files into train/val JSON.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the folder containing .mat files (recursively).")
    parser.add_argument("--output_json", type=str, default="split.json",
                        help="Name of the output JSON file.")
    parser.add_argument("--train_ratio", type=float, default=0.835,
                        help="Proportion of files that go into the train set.")
    parser.add_argument("--val_ratio", type=float, default=0.165,
                        help="Proportion of files that go into the val set.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--split_reference", type=str, default="FullSample",
                        help="Substring (directory name) to find in the path, so we can slice after it.")
    args = parser.parse_args()

    random.seed(args.seed)

    # 1. Collect all .mat files recursively
    data_dir = Path(args.data_dir).resolve()
    mat_files = list(data_dir.rglob("*.mat"))
    if not mat_files:
        print("No .mat files found. Exiting.")
        return

    # Shuffle for randomness
    random.shuffle(mat_files)

    # 2. Calculate how many go into train vs val
    total_files = len(mat_files)
    num_train = int(total_files * args.train_ratio)
    # You could do:
    # num_val = total_files - num_train
    # But we'll trust train_ratio + val_ratio ~ 1.0, so val = remainder.

    train_files = mat_files[:num_train]
    val_files = mat_files[num_train:]

    # Helper function: slice path after the split_reference,
    # then join everything with underscores, convert .mat -> .h5
    def make_name_from_path(full_path: Path) -> str:
        # full_path might be:
        #   folder1/TrainingSet/FullSample/Center005/Siemens_30T_Vida/P001/cine_lax.mat
        # We want to find "FullSample" in the path parts and take everything AFTER that

        parts = list(full_path.parts)  # each directory level plus filename
        try:
            idx = parts.index(args.split_reference)
        except ValueError:
            # If your reference isn't found, maybe just skip up to some fixed index
            # or do something else. We'll skip nothing if 'FullSample' not found.
            idx = -1

        # If found, take everything after "FullSample"
        # If not found, the entire path aside from earlier directories
        relevant_parts = parts[idx+1:] if idx != -1 else parts

        # relevant_parts might now be: ["Center005", "Siemens_30T_Vida", "P001", "cine_lax.mat"]
        # Remove .mat extension from the last part
        stem = Path(relevant_parts[-1]).stem  # e.g. "cine_lax"
        relevant_parts[-1] = stem

        # Join with underscores
        # "Center005_Siemens_30T_Vida_P001_cine_lax"
        joined_name = "_".join(relevant_parts)

        # Append .h5
        return joined_name + ".h5"

    # 3. Build the lists for JSON
    train_list = [make_name_from_path(f) for f in train_files]
    val_list   = [make_name_from_path(f) for f in val_files]

    split_dict = {
        "train": train_list,
        "val": val_list
    }

    # 4. Write out to JSON
    with open(args.output_json, "w") as outfile:
        json.dump(split_dict, outfile, indent=4)

    print(f"Split completed. Total: {total_files} files")
    print(f"Train: {len(train_list)}, Val: {len(val_list)}")
    print(f"JSON saved to: {args.output_json}")


if __name__ == "__main__":
    main()
