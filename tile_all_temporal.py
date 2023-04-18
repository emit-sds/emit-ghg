

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
    args = parser.parse_args()

    path = os.environ['PATH']
    path = path.replace('\Library\\bin;',':')
    os.environ['PATH'] = path

    subprocess.call('mkdir temporal_tiled_visuals/ch4_mosaic_temporal_refined_dynamic', shell=True)
    subprocess.call('mkdir temporal_tiled_visuals/ch4_mosaic_temporal_static', shell=True)

    if args.dates[0] == 'all':
        dates = [os.path.basename(x) for x in glob.glob('/beegfs/store/emit/ops/data/acquisitions/202*')]
    else:
        dates = args.dates

    for date in dates:
        
        dynamic_file_list =  f'temporal_line_lists/{date}_methane_refined_dynamic.txt'
        static_refined_file_list =  f'temporal_line_lists/{date}_methane_refined_static.txt'
        static_file_list =  f'temporal_line_lists/{date}_methane_static.txt'

        subprocess.call(f'find methane_20221121 -name "emit{date}*_ch4_mf_refined_ort" > {dynamic_file_list}',shell=True)
        subprocess.call(f'find methane_20221121 -name "emit{date}*_ch4_mf_refined_scaled_color_ort.tif" > {static_refined_file_list}',shell=True)
        subprocess.call(f'find methane_20221121 -name "emit{date}*_ch4_mf_scaled_color_ort.tif" > {static_file_list}',shell=True)

        od_date = f'{date[:4]}-{date[4:6]}-{date[6:8]}T00_00_01Z-to-{date[:4]}-{date[4:6]}-{date[6:8]}T23_59_59Z'
        out_fold_refined_dynamic = f'temporal_tiled_visuals/ch4_mosaic_temporal_refined_dynamic/{od_date}'
        out_fold_refined_static = f'temporal_tiled_visuals/ch4_mosaic_temporal_refined_static/{od_date}'
        out_fold_static = f'temporal_tiled_visuals/ch4_mosaic_temporal_static/{od_date}'

        
        #print(f'sbatch -N 1 -c 40 -p debug --mem=180G --wrap="python daily_tiler.py {static_refined_file_list} {out_fold_refined_static}"')
        subprocess.call(f'sbatch -N 1 -c 40 --mem=180G --job-name ch4_tile_{date} --wrap="python daily_tiler.py {static_refined_file_list} {out_fold_refined_static}"',shell=True)
        #subprocess.call(f'sbatch -N 1 -c 40 -p standard --mem=180G --wrap="python daily_tiler.py {static_file_list} {out_fold_static}"',shell=True)

    #parser = argparse.ArgumentParser(description="Run visuals workflow")
    ##parser.add_argument('input_file_list', type=str)
    #parser.add_argument('type',type=str, choices=['sa','sma','rgb','mask','mask-cons','mask-mod'])
    #parser.add_argument('--sa_tile_out',default='temporal_tiled_visuals/sa_mosaic_RGB', type=str)
    #parser.add_argument('--sma_tile_out',default='temporal_tiled_visuals/sma_mosaic_RGB', type=str)
    #parser.add_argument('--rgb_tile_out',default='temporal_tiled_visuals/rgb_mosaic_RGB', type=str)
    #parser.add_argument('--mask_tile_out',default='temporal_tiled_visuals/mask_mosaic_RGB', type=str)
    #parser.add_argument('--mask_tile_out_cons',default='temporal_tiled_visuals/mask_cons_mosaic_RGB', type=str)
    #parser.add_argument('--mask_tile_out_mod',default='temporal_tiled_visuals/mask_mod_mosaic_RGB', type=str)
    #args = parser.parse_args()

    ##loclist = np.genfromtxt(args.input_file_list,dtype=str)
    #loclist = sorted(glob.glob('/beegfs/store/emit/ops/data/acquisitions/*/*/l1b/*_l1b_loc_b0106_v01.img'))

    #run_file = f'already_run_locs_temporal_{args.type}.txt'
    #if os.path.isfile(run_file):
    #    prerun_loc_list = np.genfromtxt(run_file,dtype=str).tolist()
    #else:
    #    prerun_loc_list = []

    #


    #for loc in loclist:
    #  if loc not in prerun_loc_list:
    #
    #    rdn = loc.replace('loc','rdn')
    #    #rdn = loc.replace('l1b','l2a').replace('loc','rfl')
    #    mask = loc.replace('l1b','l2a').replace('loc','mask')
    #    sa = loc.replace('l1b','l2b').replace('loc','abun')
    #    sma = loc.replace('l1b','l3').replace('loc','cover')
    #    
    #    glt = f'glts/{os.path.basename(loc).replace("loc","glt")}'
    #    ortho_sa = f'ortho/sa/{os.path.basename(sa)}'
    #    ortho_sma = f'ortho/sma/{os.path.basename(sma)}'
    #    ortho_rgb = f'ortho/rgb/{os.path.basename(rdn)}'
    #    ortho_mask = f'ortho/rgb/{os.path.basename(mask)}'
    #    
    #    ortho_sa_ql = f'ortho/sa_ql/{os.path.basename(sa)}'
    #    ortho_sma_ql = f'ortho/sma_ql/{os.path.basename(sma)}'
    #    ortho_rgb_ql = f'ortho/rgb_ql/{os.path.basename(rdn)}'
    #    ortho_mask_ql = f'ortho/mask_ql/{os.path.basename(mask)}'
    #    ortho_mask_cons_ql = f'ortho/mask_cons_ql/{os.path.basename(mask)}'
    #    ortho_mask_mod_ql = f'ortho/mask_mod_ql/{os.path.basename(mask)}'

    #    if envi.open(envi_header(rdn)).metadata['emit acquisition daynight'] != 'Day':
    #        continue

    #    cmd_str = ''
    #    if args.type == 'sa':
    #      if os.path.isfile(ortho_sa_ql):
    #        cmd_str= f'gdal2tiles.py {ortho_sa_ql} {args.sa_tile_out} -z 4-11 --srcnodata 0 --processes=40'
    #    elif args.type == 'sma':
    #      if os.path.isfile(ortho_sma_ql):
    #        cmd_str= f'gdal2tiles.py {ortho_sma_ql} {args.sma_tile_out} -z 4-11 --srcnodata 0 --processes=40'
    #    elif args.type == 'rgb':
    #      if os.path.isfile(ortho_rgb_ql):
    #        cmd_str= f'gdal2tiles.py {ortho_rgb_ql} {args.rgb_tile_out} -z 4-11 --srcnodata 0 --processes=40 -r antialias'
    #    elif args.type == 'mask':
    #      if os.path.isfile(ortho_mask_ql):
    #        cmd_str= f'gdal2tiles.py {ortho_mask_ql} {args.mask_tile_out} -z 4-11 --srcnodata 0 --processes=40 -r antialias'
    #    elif args.type == 'mask-cons':
    #      if os.path.isfile(ortho_mask_cons_ql):
    #        cmd_str= f'gdal2tiles.py {ortho_mask_cons_ql} {args.mask_tile_out_cons} -z 4-11 --srcnodata 0 --processes=40 -r antialias'
    #    elif args.type == 'mask-mod':
    #      if os.path.isfile(ortho_mask_mod_ql):
    #        cmd_str= f'gdal2tiles.py {ortho_mask_mod_ql} {args.mask_tile_out_mod} -z 4-11 --srcnodata 0 --processes=40 -r antialias'

    #    if cmd_str != '':

    #        cmd_str += f'&& echo {loc} >> {run_file}'
    #        #cmd_str = f'srun -N 1 -c 40 --mem=180G --pty {cmd_str}'
    #        cmd_str = f'sbatch -N 1 -c 40 --mem=180G --wrap="{cmd_str}"'
    #        print(cmd_str)
    #        subprocess.call(cmd_str,shell=True)

    #        #time.sleep(0.2)

        

if __name__ == "__main__":
    main()

