#! /usr/bin/env python
#
#  Copyright 2023 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors: Philip G. Brodrick, philip.brodrick@jpl.nasa.gov

import argparse
import subprocess

import target_generation
import parallel_mf
import local_surface_control_simple
import scale
import logging
from spectral.io import envi
import numpy as np
import os
from utils import envi_header
from osgeo import gdal



def main(input_args=None):
    parser = argparse.ArgumentParser(description="Robust MF")
    parser.add_argument('radiance_file', type=str,  metavar='INPUT', help='path to input image')   
    parser.add_argument('obs_file', type=str,  help='path to observation image')   
    parser.add_argument('loc_file', type=str,  help='path to location image')   
    parser.add_argument('glt_file', type=str,  help='path to glt image')   
    parser.add_argument('l1b_bandmask_file', type=str,  help='path to l1b bandmask image')   
    parser.add_argument('l2a_mask_file', type=str,  help='path to l2a mask image')   
    parser.add_argument('output_base', type=str,  help='output basepath for output image')    
    parser.add_argument('--state_subs', type=str, default=None,  help='state file from OE retrieval')    
    parser.add_argument('--overwrite', action='store_true',  help='state file from OE retrieval')    
    parser.add_argument('--ace_filter', action='store_true',  help='use an ACE filter during matched filter')    
    parser.add_argument('--loglevel', type=str, default='INFO', help='logging verbosity')    
    parser.add_argument('--logfile', type=str, default=None, help='output file to write log to')    
    parser.add_argument('--mask_flares', type=int, default=1, help='mask flares in output')    
    parser.add_argument('--co2', action='store_true', help='flag to indicate whether to run co2')    
    args = parser.parse_args(input_args)

    irr_file = '/beegfs/scratch/brodrick/src/isofit/data/kurudz_0.1nm.dat'

    radiance_file = args.radiance_file
    radiance_file_hdr = envi_header(radiance_file)
 
    obs_file = args.obs_file
    obs_file_hdr = envi_header(obs_file)

    loc_file = args.loc_file
    loc_file_hdr = envi_header(loc_file)

    logging.basicConfig(format='%(levelname)s:%(asctime)s ||| %(message)s', level=args.loglevel,
                        filename=args.logfile, datefmt='%Y-%m-%d,%H:%M:%S')

    # Target
    co2_target_file = f'{args.output_base}_co2_target'
    ch4_target_file = f'{args.output_base}_ch4_target'

    # MF
    co2_mf_file = f'{args.output_base}_co2_mf'
    ch4_mf_file = f'{args.output_base}_ch4_mf'

    # Uncertainty
    co2_mf_uncert_file = f'{args.output_base}_co2_mf_uncert'
    ch4_mf_uncert_file = f'{args.output_base}_ch4_mf_uncert'

    # Sensitivity
    co2_mf_sens_file = f'{args.output_base}_co2_sens'
    ch4_mf_sens_file = f'{args.output_base}_ch4_sens'

    emit_noise_parameters_file = f'./instrument_noise_parameters/emit_noise.txt'

    # Flares
    flare_file = f'{args.output_base}_flares.json'

    # MF - ORT
    co2_mf_ort_file = f'{args.output_base}_co2_mf_ort'
    ch4_mf_ort_file = f'{args.output_base}_ch4_mf_ort'

    # MF -  ORT - Scaled Color
    ch4_mf_scaled_color_ort_file = f'{args.output_base}_ch4_mf_scaled_color_ort.tif'  
    co2_mf_scaled_color_ort_file = f'{args.output_base}_co2_mf_scaled_color_ort.tif'


    # Sensitivity ort
    co2_sens_ort_file = f'{args.output_base}_co2_sens_ort'
    ch4_sens_ort_file = f'{args.output_base}_ch4_sens_ort'

    # Sensitivity -  ORT - Scaled Color
    co2_sens_scaled_color_ort_file = f'{args.output_base}_co2_sens_scaled_color_ort.tif'  
    ch4_sens_scaled_color_ort_file = f'{args.output_base}_ch4_sens_scaled_color_ort.tif'  


    # Uncertainty ort
    ch4_uncert_ort_file = f'{args.output_base}_ch4_uncert_ort'
    co2_uncert_ort_file = f'{args.output_base}_co2_uncert_ort'

    # Uncertainty -  ORT - Scaled Color
    co2_uncert_scaled_color_ort_file = f'{args.output_base}_co2_uncert_scaled_color_ort.tif'
    ch4_uncert_scaled_color_ort_file = f'{args.output_base}_ch4_uncert_scaled_color_ort.tif'
    
    path = os.environ['PATH']
    path = path.replace('\Library\\bin;',':')
    os.environ['PATH'] = path

    if os.path.isfile(ch4_mf_file):
        dat = gdal.Open(ch4_mf_file).ReadAsArray()
        if np.all(dat == -9999):
            subprocess.call(f'rm {args.output_base}_ch4*',shell=True)

    if (os.path.isfile(co2_target_file) is False and args.co2) or args.overwrite or os.path.isfile(ch4_target_file) is False:
        sza = envi.open(obs_file_hdr).open_memmap(interleave='bip')[...,4]
        mean_sza = np.mean(sza[sza != -9999])

        elevation = envi.open(loc_file_hdr).open_memmap(interleave='bip')[...,2]
        mean_elevation = np.mean(elevation[elevation != -9999]) / 1000.
        mean_elevation = min(max(0, mean_elevation),3)

        if args.state_subs is not None:
            state_ds = envi.open(envi_header(args.state_subs))
            band_names = state_ds.metadata['band names']
            h2o = state_ds.open_memmap(interleave='bip')[...,band_names.index('H2OSTR')]
            mean_h2o = np.mean(h2o[h2o != -9999])
        else:
            # Just guess something...
            exit()
            mean_h2o = 1.3



    if (os.path.isfile(co2_target_file) is False or args.overwrite) and args.co2:
        target_generation.main(['--co2', 
                                '-z', str(mean_sza), 
                                '-s', '100', 
                                '-g', str(mean_elevation), 
                                '-w', str(mean_h2o), 
                                '--output', co2_target_file, 
                                '--hdr', radiance_file_hdr,
                                '--lut_dataset', '/beegfs/scratch/jchapman/CO2CH4TargetGen/dataset_co2_full.hdf5'])
    if os.path.isfile(ch4_target_file) is False or args.overwrite:
        target_generation.main(['--ch4', 
                                '-z', str(mean_sza), 
                                '-s', '100', 
                                '-g', str(mean_elevation), 
                                '-w', str(mean_h2o), 
                                '--output', ch4_target_file, 
                                '--hdr', radiance_file_hdr,
                                '--lut_dataset', '/beegfs/scratch/jchapman/CO2CH4TargetGen/dataset_ch4_full.hdf5'])


    if (os.path.isfile(co2_mf_file) is False or args.overwrite) and args.co2:
        subargs = [args.radiance_file, 
                   co2_target_file, 
                   co2_mf_file, 
                   '--n_mc', '1', 
                   '--l1b_bandmask_file', args.l1b_bandmask_file, 
                   '--l2a_mask_file', args.l2a_mask_file, 
                   '--wavelength_range', '500', '1340', '1500', '1790', '1950', '2450', 
                   '--fixed_alpha', '0.0000000001', 
                   '--mask_clouds_water', 
                   '--flare_outfile', flare_file, 
                   '--noise_parameters_file', emit_noise_parameters_file, 
                   '--sens_output_file', co2_mf_sens_file, 
                   '--uncert_output_file', co2_mf_uncert_file]
        if args.ace_filter:
            subargs.append('--use_ace_filter')
        parallel_mf.main(subargs)
    
    if os.path.isfile(ch4_mf_file) is False or args.overwrite:
        logging.info('starting parallel mf')
        
        subargs = [args.radiance_file, 
                   ch4_target_file, 
                   ch4_mf_file, 
                   '--n_mc', '1', 
                   '--l1b_bandmask_file', args.l1b_bandmask_file, 
                   '--l2a_mask_file', args.l2a_mask_file, 
                   '--wavelength_range', '500', '1340', '1500', '1790', '1950', '2450', 
                   '--fixed_alpha', '0.0000000001', 
                   '--mask_clouds_water', 
                   '--flare_outfile', flare_file, 
                   '--noise_parameters_file', emit_noise_parameters_file, 
                   '--sens_output_file', ch4_mf_sens_file, 
                   '--uncert_output_file', ch4_mf_uncert_file]

        if args.mask_flares == 1:
            subargs.append('--mask_flares')
        if args.ace_filter:
            subargs.append('--use_ace_filter')
        parallel_mf.main(subargs)

    # ORT MF
    if (os.path.isfile(co2_mf_ort_file) is False or args.overwrite) and args.co2:
        subprocess.call(f'python apply_glt.py {args.glt_file} {co2_mf_file} {co2_mf_ort_file}',shell=True)
    if os.path.isfile(ch4_mf_ort_file) is False or args.overwrite:
        subprocess.call(f'python apply_glt.py {args.glt_file} {ch4_mf_file} {ch4_mf_ort_file}',shell=True)

    # ORT Sensitivity
    if os.path.isfile(co2_sens_ort_file) is False or args.overwrite and args.co2:
        subprocess.call(f'python apply_glt.py {args.glt_file} {co2_mf_sens_file} {co2_sens_ort_file}',shell=True)
    if os.path.isfile(co2_uncert_ort_file) is False or args.overwrite:
        subprocess.call(f'python apply_glt.py {args.glt_file} {co2_mf_uncert_file} {co2_uncert_ort_file}',shell=True)
    
    # ORT Uncertainty
    if os.path.isfile(ch4_sens_ort_file) is False or args.overwrite and args.co2:
        subprocess.call(f'python apply_glt.py {args.glt_file} {ch4_mf_sens_file} {ch4_sens_ort_file}',shell=True)
    if os.path.isfile(ch4_uncert_ort_file) is False or args.overwrite:
        subprocess.call(f'python apply_glt.py {args.glt_file} {ch4_mf_uncert_file} {ch4_uncert_ort_file}',shell=True)
    

    # Color MF
    if (os.path.isfile(co2_mf_scaled_color_ort_file) is False or args.overwrite) and args.co2:
        scale.main([co2_mf_ort_file, co2_mf_scaled_color_ort_file, '1', '100000', '--cmap', 'viridis'])
    if os.path.isfile(ch4_mf_scaled_color_ort_file) is False or args.overwrite:
        scale.main([ch4_mf_ort_file, ch4_mf_scaled_color_ort_file, '1', '1000', '--cmap', 'plasma'])

    # Color Sensitivity
    if (os.path.isfile(co2_sens_scaled_color_ort_file) is False or args.overwrite) and args.co2:
        scale.main([co2_sens_ort_file, co2_sens_scaled_color_ort_file, '0', '2', '--cmap', 'RdBu_r'])
    if os.path.isfile(ch4_sens_scaled_color_ort_file) is False or args.overwrite:
        scale.main([ch4_sens_ort_file, ch4_sens_scaled_color_ort_file, '0', '2', '--cmap', 'RdBu_r'])

    # Color Uncertainty
    if (os.path.isfile(co2_uncert_scaled_color_ort_file) is False or args.overwrite) and args.co2:
        scale.main([co2_uncert_ort_file, co2_uncert_scaled_color_ort_file, '1', '100000', '--cmap', 'viridis'])
    if os.path.isfile(ch4_uncert_scaled_color_ort_file) is False or args.overwrite:
        scale.main([ch4_uncert_ort_file, ch4_uncert_scaled_color_ort_file, '1', '1000', '--cmap', 'plasma'])


    rdn_kmz = args.radiance_file.replace('.img','.kmz')
    dst_rdn_kmz = f'{args.output_base}_rdn_rgb.kmz'
    if os.path.isfile(rdn_kmz) and os.path.isfile(dst_rdn_kmz) is False:
        subprocess.call(f'cp {rdn_kmz} {dst_rdn_kmz}',shell=True)







if __name__ == '__main__':
    main()
