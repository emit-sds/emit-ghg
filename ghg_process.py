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
import logging
import os

import target_generation
import parallel_mf
import scale
import apply_glt
from spectral.io import envi
import numpy as np
from utils import envi_header, convert_to_cog
from files import Filenames

metadata = {
    'ch4': {
        'mf': {
            'name': 'EMIT_L2B_CH4ENH',
            'description': 'Methane Enhancement Values',
            'units': 'ppm m'
        },
        'sens': {
            'name': 'EMIT_L2B_CH4SENS',
            'description': 'Methane Enhancement Sensitivity Values',
            'units': 'unitless'
        },
        'unc': {
            'name': 'EMIT_L2B_CH4UNCERT',
            'description': 'Methane Enhancement Uncertainty Values',
            'units':  'ppm m'
        }
    },
    'co2': {
        'mf': {
            'name': 'EMIT_L2B_CO2ENH',
            'description': 'Carbon Dioxide Enhancement Values',
            'units': 'ppm m'
        },
        'sens': {
            'name': 'EMIT_L2B_CO2SENS',
            'description': 'Carbon Dioxide Enhancement Sensitivity Values',
            'units': 'unitless'
        },
        'unc': {
            'name': 'EMIT_L2B_CO2UNCERT',
            'description': 'Carbon Dioxide Enhancement Uncertainty Values',
            'units': 'ppm m'
        }
    }
}

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
    parser.add_argument('--lut_file', type=str, default='/store/shared/ghg/dataset_ch4_full.hdf5')
    parser.add_argument('--noise_file', type=str, default='instrument_noise_parameters/emit_noise.txt')
    parser.add_argument('--wavelength_range', nargs='+', type=float, default=[500, 1340, 1500, 1790, 1950, 2450],
                        help='wavelengths to use: None = default for gas, 2x values = min/max pairs of regions')
    parser.add_argument('--co2', action='store_true', help='flag to indicate whether to run co2')
    parser.add_argument('--software_version', type=str, default=None)
    parser.add_argument('--product_version', type=str, default=None)
    args = parser.parse_args(input_args)

    if args.wavelength_range is not None and len(args.wavelength_range) % 2 != 0:
        raise ValueError('wavelength_range must have an even number of elements')

    radiance_file = args.radiance_file
    radiance_file_hdr = envi_header(radiance_file)

    obs_file = args.obs_file
    obs_file_hdr = envi_header(obs_file)

    loc_file = args.loc_file
    loc_file_hdr = envi_header(loc_file)

    logging.basicConfig(format='%(levelname)s:%(asctime)s ||| %(message)s', level=args.loglevel,
                        filename=args.logfile, datefmt='%Y-%m-%d,%H:%M:%S')

    files = Filenames(args.output_base)

    # if os.path.isfile(files.mf_file):
    #     dat = gdal.Open(files.mf_file).ReadAsArray()
        #if np.all(dat == -9999):
        #    subprocess.call(f'rm {args.output_base}_ch4*',shell=True)

    print(files.target_file)

    if os.path.isfile(files.target_file) is False or args.overwrite:
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

    gas = 'ch4'
    if args.co2:
        gas = 'co2'

    # Run target generation
    if (os.path.isfile(files.target_file) is False or args.overwrite):
        target_params = ['-z', str(mean_sza),
                         '-s', '100',
                         '-g', str(mean_elevation),
                         '-w', str(mean_h2o),
                         '--output', files.target_file,
                         '--hdr', radiance_file_hdr,
                         '--lut_dataset', args.lut_file]
        if args.co2:
            target_params.append('--co2')

        print(target_params)

        target_generation.main(target_params)

    # Run MF
    if (os.path.isfile(files.mf_file) is False or args.overwrite):
        subargs = [args.radiance_file,
                   files.target_file,
                   files.mf_file,
                   '--n_mc', '1',
                   '--l1b_bandmask_file', args.l1b_bandmask_file,
                   '--l2a_mask_file', args.l2a_mask_file,
                   '--fixed_alpha', '0.0000000001',
                   '--mask_clouds_water',
                   '--flare_outfile', files.flare_file,
                   '--noise_parameters_file', args.noise_file,
                   '--sens_output_file', files.mf_sens_file,
                   '--uncert_output_file', files.mf_uncert_file]

        if args.wavelength_range is not None:
            subargs.extend(['--wavelength_range'] + [str(val) for val in args.wavelength_range])

        if args.mask_flares == 1:
            subargs.append('--mask_flares')

        if args.ace_filter:
            subargs.append('--use_ace_filter')
        parallel_mf.main(subargs)

    # ORT MF
    if (os.path.isfile(files.mf_ort_file) is False or args.overwrite):
        apply_glt.main([args.glt_file, files.mf_file, files.mf_ort_file])
        convert_to_cog(files.mf_ort_file,
                       files.mf_ort_cog,
                       metadata[gas]['mf'],
                       args.software_version,
                       args.product_version)
    # ORT Sensitivity
    if os.path.isfile(files.sens_ort_file) is False or args.overwrite:
        apply_glt.main([args.glt_file, files.mf_sens_file, files.sens_ort_file])
    if os.path.isfile(files.sens_ort_cog) is False or args.overwrite:
        convert_to_cog(files.sens_ort_file,
                       files.sens_ort_cog,
                       metadata[gas]['sens'],
                       args.software_version,
                       args.product_version)
    # ORT Uncertainty
    if os.path.isfile(files.uncert_ort_file) is False or args.overwrite:
        apply_glt.main([args.glt_file, files.mf_uncert_file, files.uncert_ort_file])
    if os.path.isfile(files.uncert_ort_cog) is False or args.overwrite:
        convert_to_cog(files.uncert_ort_file,
                       files.uncert_ort_cog,
                       metadata[gas]['unc'],
                       args.software_version,
                       args.product_version)
    # Quicklook MF
    if (os.path.isfile(files.mf_ort_ql) is False or args.overwrite) and args.co2:
        scale.main([files.mf_ort_file, files.mf_ort_ql, '1', '100000', '--cmap', 'viridis'])
    if os.path.isfile(files.mf_ort_ql) is False or args.overwrite:
        scale.main([files.mf_ort_file, files.mf_ort_ql, '1', '1000', '--cmap', 'plasma'])

    # Color MF
    # if (os.path.isfile(files.mf_scaled_color_ort_file) is False or args.overwrite) and args.co2:
    #     scale.main([files.mf_ort_file, files.mf_scaled_color_ort_file, '1', '100000', '--cmap', 'viridis'])
    # if os.path.isfile(files.mf_scaled_color_ort_file) is False or args.overwrite:
    #     scale.main([files.mf_ort_file, files.mf_scaled_color_ort_file, '1', '1000', '--cmap', 'plasma'])

    # # Color Sensitivity (same for co2 and ch4)
    # if os.path.isfile(files.sens_scaled_color_ort_file) is False or args.overwrite:
    #     scale.main([files.sens_ort_file, files.sens_scaled_color_ort_file, '0', '2', '--cmap', 'RdBu_r'])

    # # Color Uncertainty
    # if (os.path.isfile(files.uncert_scaled_color_ort_file) is False or args.overwrite) and args.co2:
    #     scale.main([files.uncert_ort_file, files.uncert_scaled_color_ort_file, '1', '100000', '--cmap', 'viridis'])
    # if os.path.isfile(files.uncert_scaled_color_ort_file) is False or args.overwrite:
    #     scale.main([files.uncert_ort_file, files.uncert_scaled_color_ort_file, '1', '1000', '--cmap', 'plasma'])


if __name__ == '__main__':
    main()
