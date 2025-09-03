#!/usr/bin/env python3

import subprocess
import argparse
import glob
import os
import pdb

def run_ghg_process(fid, output_dir, execute=False):
    """
    Constructs the srun command for ghg_process.py, taking the case prefix and output directory as input.
    Orbit and scene are determined via glob.

    Args:
        fid (str): The fid of the case name (e.g., "emit20250623t173722").
        output_dir (str): The output directory.
        execute (bool): Whether to actually execute the command. Defaults to False.
    """

    case_prefix = fid[4:]
    date_str = case_prefix[:8]
    time_str = case_prefix[9:15]

    # Use glob to find the files and extract orbit and scene
    search_pattern = f"/store/emit/ops/data/acquisitions/{date_str}/emit{date_str}t{time_str}/l1b/emit{date_str}t{time_str}*l1b_rdn*.img"
    files = glob.glob(search_pattern)

    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {search_pattern}")

    # Extract orbit and scene from the filename (assuming consistent naming)
    filename = os.path.basename(files[0])  # Get only the filename, not the full path
    parts = filename.split("_")
    orbit = parts[1]
    scene = parts[2]

    l1b_rdn_img = f"/store/emit/ops/data/acquisitions/{date_str}/emit{date_str}t{time_str}/l1b/emit{date_str}t{time_str}_{orbit}_{scene}_l1b_rdn_b0106_v01.img"
    l1b_obs_img = f"/store/emit/ops/data/acquisitions/{date_str}/emit{date_str}t{time_str}/l1b/emit{date_str}t{time_str}_{orbit}_{scene}_l1b_obs_b0106_v01.img"
    l1b_loc_img = f"/store/emit/ops/data/acquisitions/{date_str}/emit{date_str}t{time_str}/l1b/emit{date_str}t{time_str}_{orbit}_{scene}_l1b_loc_b0106_v01.img"
    l1b_glt_img = f"/store/emit/ops/data/acquisitions/{date_str}/emit{date_str}t{time_str}/l1b/emit{date_str}t{time_str}_{orbit}_{scene}_l1b_glt_b0106_v01.img"
    l1b_bandmask_img = f"/store/emit/ops/data/acquisitions/{date_str}/emit{date_str}t{time_str}/l1b/emit{date_str}t{time_str}_{orbit}_{scene}_l1b_bandmask_b0106_v01.img"
    l2a_mask_img = f"/store/emit/ops/data/acquisitions/{date_str}/emit{date_str}t{time_str}/l2a/emit{date_str}t{time_str}_{orbit}_{scene}_l2a_mask_b0106_v01.img"

    state_subs_path = f"/store/emit/ops/data/acquisitions/{date_str}/emit{date_str}t{time_str}/l2a/emit{date_str}t{time_str}_{orbit}_{scene}_l2a_statesubs_b0106_v01.img"

    command = [
        "srun",
        "-p", "debug",
        "-N", "1",
        "-c", "64",
        "--mem=300G",
        "--pty",
        "python",
        "ghg_process.py",
        l1b_rdn_img,
        l1b_obs_img,
        l1b_loc_img,
        l1b_glt_img,
        l1b_bandmask_img,
        l2a_mask_img,
        output_dir,
        "--state_subs", state_subs_path,
        "--overwrite"
    ]

    print("Command to be executed:")
    print(" ".join(command))  # Print the command

    if execute:
        try:
            subprocess.run(command, check=True)
            print(f"Command executed successfully for case prefix {case_prefix}.")
        except subprocess.CalledProcessError as e:
            print(f"Error executing command for case prefix {case_prefix}: {e}")
    else:
        print("Use --execute to run the command.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct and optionally execute ghg_process.py with case prefix and output directory.")
    parser.add_argument("case_prefix", help="The case prefix (e.g., emit20250623t173722)")
    parser.add_argument("output_dir", help="The output directory")
    parser.add_argument("--execute", action="store_true", help="Execute the command")

    args = parser.parse_args()

    run_ghg_process(args.case_prefix, args.output_dir, args.execute)