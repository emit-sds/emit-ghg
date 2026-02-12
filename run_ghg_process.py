#!/usr/bin/env python3

import subprocess
import argparse
import glob
import os
import pdb
import json

def run_ghg_process_EMIT_cluster(fid, output_dir, execute=False):
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
        os.path.join(output_dir, fid),
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

def run_ghg_process_EMIT_DAAC(fid, input_path, output_dir, execute=False):
    """
    Constructs the srun command for ghg_process.py, taking the case prefix and output directory as input.
    Orbit and scene are determined via glob.

    Args:
        fid (str): The fid of the case name (e.g., "emit20250623t173722").
        output_dir (str): The output directory.
        execute (bool): Whether to actually execute the command. Defaults to False.
    """

    json_file_list_filename = os.path.join(input_path, 'data_files.json')
    if not os.path.exists(json_file_list_filename):
        raise FileNotFoundError(f'Could not find {json_file_list_filename}')
    json_file_list = json.load(open(json_file_list_filename))

    command = [
        "srun",
        "-p", "debug",
        "-N", "1",
        "-c", "1",
        "--mem=30G",
        "--pty",
        "python",
        "ghg_process.py",
        json_file_list['RAD'],
        json_file_list['OBS'],
        json_file_list['RAD'],
        json_file_list['OBS'],
        json_file_list['L2A_MASK'],
        json_file_list['L2A_MASK'],
        output_dir,
        "--state_subs", json_file_list['L2A_MASK'],
        "--overwrite"
    ]

    print("Command to be executed:")
    print(" ".join(command))  # Print the command

    if execute:
        try:
            subprocess.run(command, check=True)
            print(f"Command executed successfully for case prefix {fid}.")
        except subprocess.CalledProcessError as e:
            print(f"Error executing command for case prefix {fid}: {e}")
    else:
        print("Use --execute to run the command.")

def run_ghg_process_AV3_DAAC(fid, input_path, output_dir, execute=False):
    """
    Constructs the srun command for ghg_process.py, taking the case prefix and output directory as input.
    Orbit and scene are determined via glob.

    Args:
        fid (str): The fid of the case name (e.g., "emit20250623t173722").
        output_dir (str): The output directory.
        execute (bool): Whether to actually execute the command. Defaults to False.
    """

    json_file_list_filename = os.path.join(input_path, 'data_files.json')
    if not os.path.exists(json_file_list_filename):
        raise FileNotFoundError(f'Could not find {json_file_list_filename}')
    json_file_list = json.load(open(json_file_list_filename))

    l1b_rdn_nc_search_pattern = f'{input_path}/*_RDN.nc'
    l1b_rdn_files = glob.glob(l1b_rdn_nc_search_pattern)
    if len(l1b_rdn_files) != 1:
        raise ValueError(f'Found {len(l1b_rdn_files)} folders at {l1b_rdn_nc_search_pattern}')
    l1b_rdn_file = l1b_rdn_files[0]

    l1b_mask_filename = os.path.join(output_dir, 'L1B_MASK')

    # Make L1B MASK
    l1b_mask_command = [
        "srun",
        "-p", "debug",
        "-N", "1",
        "-c", "64",
        "--mem=300G",
        "--pty",
        "python",
        'make_emit_masks.py',
        l1b_rdn_file,
        l1b_rdn_file,
        '/store/airborne/software/asds_data/main/l1b_rdn/kurucz_0.1nm.dat',
        l1b_mask_filename
    ]
    print("L1B MASK command to be executed:")
    print(" ".join(l1b_mask_command))  # Print the command
    if execute:
        try:
            subprocess.run(l1b_mask_command, check=True)
            print(f"L1B MASK command executed successfully for case prefix {fid}.")
        except subprocess.CalledProcessError as e:
            print(f"Error executing L1B MASK for case prefix {fid}: {e}")
    else:
        print("Use --execute to run the command.")

    command = [
        "srun",
        "-p", "debug",
        "-N", "1",
        "-c", "1",
        "--mem=30G",
        "--pty",
        "python",
        "ghg_process.py",
        l1b_rdn_file,
        json_file_list['OBS'],
        l1b_rdn_file,
        json_file_list['OBS'],
        json_file_list['BANDMASK'],
        l1b_mask_filename,
        output_dir,
        "--overwrite",
        '--noise_file', '/home/jfahlen/src/emit-ghg/instrument_noise_parameters/AV320250715t180536_001_L1B_RDN_76a54582_NOISE.txt'
    ]

    print("Command to be executed:")
    print(" ".join(command))  # Print the command

    if execute:
        try:
            subprocess.run(command, check=True)
            print(f"Command executed successfully for case prefix {fid}.")
        except subprocess.CalledProcessError as e:
            print(f"Error executing command for case prefix {fid}: {e}")
    else:
        print("Use --execute to run the command.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct and optionally execute ghg_process.py with case prefix and output directory.")
    subparsers = parser.add_subparsers(
        dest="mode",               # stores the chosen sub‑command name
        title="mode",              # optional heading in help output
        description="run mode",    # optional description in help output
        required=True,             # Python ≥3.7: make the sub‑command mandatory
        help="choose which processing mode to run"
    )

    # EMIT_cluster mode parser
    parser_emit = subparsers.add_parser("EMIT_cluster", help="Run EMIT processing")
    parser_emit.add_argument("fid", help="The case prefix (e.g., emit20250623t173722)")
    parser_emit.add_argument("output_dir", help="The output directory")
    parser_emit.add_argument("--execute", action="store_true", help="Execute the command")

    # AV3_DAAC mode parser
    parser_av3 = subparsers.add_parser("AV3_DAAC", help="Run AV3 processing")
    parser_av3.add_argument("fid", help="The case prefix (e.g., AV320240914t203535)")
    parser_av3.add_argument("input_path", help="Path to the input data granule, ex: .../av3_granules/AV320240914t210622_002_granule/")
    parser_av3.add_argument("output_dir", help="The output directory")
    parser_av3.add_argument("--execute", action="store_true", help="Execute the command")

    # EMIT_DAAC mode parser
    parser_av3 = subparsers.add_parser("EMIT_DAAC", help="Run AV3 processing")
    parser_av3.add_argument("fid", help="The case prefix (e.g., emit20250623t173722)")
    parser_av3.add_argument("input_path", help="Path to the input data granule, ex: .../av3_granules/AV320240914t210622_002_granule/")
    parser_av3.add_argument("output_dir", help="The output directory")
    parser_av3.add_argument("--execute", action="store_true", help="Execute the command")

    args = parser.parse_args()

    if args.mode == 'EMIT_cluster':
        run_ghg_process_EMIT_cluster(args.fid, args.output_dir, args.execute)
    elif args.mode == 'EMIT_DAAC':
        run_ghg_process_EMIT_DAAC(args.fid, args.input_path, args.output_dir, args.execute)
    elif args.mode == 'AV3_DAAC':
        run_ghg_process_AV3_DAAC(args.fid, args.input_path, args.output_dir, args.execute)