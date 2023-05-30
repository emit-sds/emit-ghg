

import os
import subprocess
import argparse
import numpy as np
import glob
import time
#from emit_utils.file_checks import envi_header
from spectral.io import envi


def main():

    parser = argparse.ArgumentParser(description="Run visuals workflow")
    parser.add_argument('dates', type=str, nargs='+')
    parser.add_argument('--type', type=str, default='co2', choices=['co2','ch4'])
    args = parser.parse_args()

    path = os.environ['PATH']
    path = path.replace('\Library\\bin;',':')
    os.environ['PATH'] = path

    subprocess.call(f'mkdir temporal_tiled_visuals/{args.type}_mosaic_temporal_refined_dynamic', shell=True)
    subprocess.call(f'mkdir temporal_tiled_visuals/{args.type}_mosaic_temporal_static', shell=True)

    if args.dates[0] == 'all':
        dates = [os.path.basename(x) for x in glob.glob('/beegfs/store/emit/ops/data/acquisitions/202*')]
    else:
        dates = args.dates

    for date in dates:
        
        if args.type == 'ch4':
            dynamic_file_list =  f'temporal_line_lists/{date}_methane_refined_dynamic.txt'
            static_refined_file_list =  f'temporal_line_lists/{date}_methane_refined_static.txt'
            static_file_list =  f'temporal_line_lists/{date}_methane_static.txt'
        elif args.type == 'co2':
            dynamic_file_list =  f'temporal_line_lists/{date}_co2_refined_dynamic.txt'
            static_refined_file_list =  f'temporal_line_lists/{date}_co2_refined_static.txt'
            static_file_list =  f'temporal_line_lists/{date}_co2_static.txt'

        subprocess.call(f'find methane_20221121 -name "emit{date}*_{args.type}_mf_refined_ort" > {dynamic_file_list}',shell=True)
        subprocess.call(f'find methane_20221121 -name "emit{date}*_{args.type}_mf_refined_scaled_color_ort.tif" > {static_refined_file_list}',shell=True)
        subprocess.call(f'find methane_20221121 -name "emit{date}*_{args.type}_mf_scaled_color_ort.tif" > {static_file_list}',shell=True)

        od_date = f'{date[:4]}-{date[4:6]}-{date[6:8]}T00_00_01Z-to-{date[:4]}-{date[4:6]}-{date[6:8]}T23_59_59Z'
        out_fold_refined_dynamic = f'temporal_tiled_visuals/{args.type}_mosaic_temporal_refined_dynamic/{od_date}'
        out_fold_refined_static = f'temporal_tiled_visuals/{args.type}_mosaic_temporal_refined_static/{od_date}'
        out_fold_static = f'temporal_tiled_visuals/{args.type}_mosaic_temporal_static/{od_date}'

        
        #print(f'sbatch -N 1 -c 40 -p debug --mem=180G --wrap="python daily_tiler.py {static_refined_file_list} {out_fold_refined_static}"')
        subprocess.call(f'sbatch -N 1 -c 40 --mem=180G --job-name {args.type}_tile_{date} --wrap="python daily_tiler.py {static_refined_file_list} {out_fold_refined_static}"',shell=True)

        

if __name__ == "__main__":
    main()

