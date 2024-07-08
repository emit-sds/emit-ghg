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
# Authors: Brian D. Bue
# Authors: David R. Thompson
# Authors: Jay Fahlen
# Authors: Red Willow Coleman

import argparse
from spectral.io import envi

import sys
import scipy
import scipy.ndimage
import numpy as np
from utils import envi_header, write_bil_chunk, ReadAbstractDataSet, WriteAbstractDataSet
import json
from utils import SerialEncoder
import pdb

import logging
import os

# Borrowed from isofit/isofit/wrappers/ray.py
if os.environ.get("GHG_DEBUG"):
    logging.info("Using internal ray")
    import rray as ray
else:
    import ray

def main(input_args=None):
    parser = argparse.ArgumentParser(description="Robust MF")
    parser.add_argument('radiance_file', type=str,  metavar='INPUT', help='path to input image')   
    parser.add_argument('library', type=str,  metavar='LIBRARY', help='path to target library file')
    parser.add_argument('output_file', type=str,  metavar='OUTPUT', help='path for output image (mf ch4 ppm)')    

    parser.add_argument('--covariance_style', type=str, default='looshrinkage', choices=['empirical', 'looshrinkage'], help='style of covariance estimation') 
    parser.add_argument('--fixed_alpha', type=float, default=None, help='fixed value for shrinkage (with looshrinkage covariance style only)')    
    parser.add_argument('--num_cores', type=int, default=-1, help='number of cores (-1 (default))')
    parser.add_argument('--n_mc', type=int, default=10, help='number of monte carlo runs')
    parser.add_argument('--mc_bag_fraction',type=float, default=0.7, help='fraction of data to use in each MC instance')
    parser.add_argument('--ray_temp_dir', type=str, default=None, help='ray temp directory (None (default))')    
    parser.add_argument('--wavelength_range', nargs='+', type=float, default=None, help='wavelengths to use: None = default for gas, 2x values = min/max pairs of regions')         
    parser.add_argument('--remove_dominant_pcs',action='store_true', help='remove dominant PCs from covariance calculation')         
    parser.add_argument('--subsample_strategy',type=str,choices=['random','spatial_blocks'], help='sampling strategy for mc runs')         
    parser.add_argument('--l1b_bandmask_file',type=str,default=None, help='path to the l1b bandmask file for saturation')         
    parser.add_argument('--l2a_mask_file', type=str,  help='path to l2a mask image for clouds and water')   
    parser.add_argument('--mask_clouds_water',action='store_true', help='mask clouds and water from output matched filter')         
    parser.add_argument('--mask_saturation',action='store_true', help='mask saturated pixels from output matched filter')         
    parser.add_argument('--mask_flares',action='store_true', help='mask flared pixels from output matched filter')         
    parser.add_argument('--ppm_scaling', type=float, default=100000.0, help='scaling factor to unit convert outputs - based on target')         
    parser.add_argument('--ace_filter', action='store_true', help='Use the Adaptive Cosine Estimator (ACE) Filter')    
    parser.add_argument('--target_scaling', type=str,choices=['mean','pixel'],default='mean', help='value to scale absorption coefficients by')    
    parser.add_argument('--nodata_value', type=float, default=-9999, help='output nodata value')         
    parser.add_argument('--flare_outfile', type=str, default=None, help='output geojson to write flare location centers')         
    parser.add_argument('--chunksize', type=int, default=None, help='chunk radiance (for memory issues with large scenes)')         
    parser.add_argument('--loglevel', type=str, default='DEBUG', help='logging verbosity')    
    parser.add_argument('--logfile', type=str, default=None, help='output file to write log to')         
    args = parser.parse_args(input_args)

    #Set up logging
    logging.basicConfig(format='%(levelname)s:%(asctime)s ||| %(message)s', level=args.loglevel,
                        filename=args.logfile, datefmt='%Y-%m-%d,%H:%M:%S')
   
    
    logging.info('Started processing input file: "%s"'%str(args.radiance_file))
    ds = ReadAbstractDataSet(args.radiance_file, netcdf_key = 'radiance', envi_interleave = 'bil')
    wavelengths = np.array([float(x) for x in ds.metadata['wavelength']])

    if args.wavelength_range is None:
        if 'ch4' in args.library:
            args.wavelength_range = [2137, 2493]
        elif 'co2' in args.library:
            args.wavelength_range = [1922, 2337]
        else:
            logging.error('could not set a default active range - neither co2 nor ch4 found in library name')
            sys.exit(0)
    else:
        if args.wavelength_range[0] > args.wavelength_range[1]:
            logging.error('wavelength range must be in increasing order')
            sys.exit(0)
        if len(args.wavelength_range) % 2 != 0:
            logging.error('wavelength range must be in pairs')
            sys.exit(0)

    active_wl_idx = []
    for n in range(len(args.wavelength_range)//2):
        la = np.where(np.logical_and(wavelengths > args.wavelength_range[2*n], wavelengths <= args.wavelength_range[2*n+1]))[0]
        active_wl_idx.extend(la.tolist())
    always_exclude_idx = []
    if 'emit' in args.radiance_file.lower():
        always_exclude_idx = np.where(np.logical_and(wavelengths < 1321, wavelengths > 1275))[0].tolist()
    active_wl_idx = np.array([x for x in active_wl_idx if x not in always_exclude_idx])

    logging.info(f'Active wavelength range: {args.wavelength_range}: {len(active_wl_idx)} channels')

    logging.info("load target library")
    library_reference = np.float64(np.loadtxt(args.library))
    absorption_coefficients = library_reference[active_wl_idx,2]

    logging.info('Create output file, initialized with nodata')
    outmeta = ds.metadata
    outmeta['data type'] = np2envitype(np.float32)
    outmeta['bands'] = 1
    outmeta['description'] = 'Matched Filter Results'
    outmeta['band names'] = 'Matched Filter'
    outmeta['interleave'] = 'bil'    
    outmeta['z plot range'] = '{0, 1500}' #adapt to include co2
    outmeta['data ignore value'] = args.nodata_value
    for kwarg in ['smoothing factors','wavelength','wavelength units','fwhm']:
        outmeta.pop(kwarg,None)

    output_ds = WriteAbstractDataSet(args.output_file, outmeta = outmeta)
    output_shape = (int(outmeta['lines']),int(outmeta['bands']),int(outmeta['samples']))
    output_ds.write(np.ones(output_shape)*args.nodata_value, 0, output_shape)

 
    if args.chunksize is None:
        chunk_edges = [0, output_shape[0]]
    else:
        chunk_edges = np.arange(0, output_shape[0], args.chunksize).tolist()
        chunk_edges.append(output_shape[0])

    rayargs = {'_temp_dir': args.ray_temp_dir, 'ignore_reinit_error': True, 'include_dashboard': False}
    rayargs['num_cpus'] = args.num_cores
    if args.num_cores == -1:
        import multiprocessing
        rayargs['num_cpus'] = multiprocessing.cpu_count() - 1
    ray.init(**rayargs)
    rdn_id = None
    absorption_coefficients_id = ray.put(absorption_coefficients)
    del absorption_coefficients
    for _ce, ce in enumerate(chunk_edges[:-1]):
        
        if rdn_id is not None:
            del rdn_id; rdn_id = None
        logging.info(f"load radiance for chunk {_ce +1} / {len(chunk_edges) - 1}")
        radiance = ds[ce:chunk_edges[_ce+1],...].copy()
        chunk_shape = (chunk_edges[_ce+1] - ce, output_shape[1], output_shape[2])

        logging.info("load masks")
        good_pixel_mask = np.ones((radiance.shape[0],radiance.shape[2]),dtype=bool)
        saturation = None
        if args.l1b_bandmask_file is not None:
            logging.debug("loading pixel mask")
            dilated_saturation, saturation = calculate_saturation_mask(args.l1b_bandmask_file, radiance, chunk_edges=[ce,chunk_edges[_ce+1]])
            good_pixel_mask[dilated_saturation] = False

        logging.debug("adding flare mask")
        dilated_flare_mask, flare_mask = calculate_flare_mask(radiance, good_pixel_mask, wavelengths)
        good_pixel_mask[dilated_flare_mask] = False

        if args.flare_outfile is not None:
            logging.info(f'writing flare locations to {args.flare_outfile}')
            write_hotspot_vector(args.flare_outfile, flare_mask, saturation)

        logging.debug("adding cloud / water mask")
        clouds_and_surface_water_mask = None
        if args.l2a_mask_file is not None:
            masks = ReadAbstractDataSet(args.l2a_mask_file, netcdf_key = 'mask', envi_interleave = 'bip')[...]
            clouds_and_surface_water_mask = np.sum(masks[ce:chunk_edges[_ce+1],:,:3],axis=-1) > 0
            good_pixel_mask = np.where(clouds_and_surface_water_mask, False, good_pixel_mask)

        logging.info('initializing ray, adding data to shared memory')

        rdn_id = ray.put(radiance)
        del radiance 


        logging.info('Run jobs')
        jobs = [mf_one_column.remote(col, rdn_id, absorption_coefficients_id, active_wl_idx, good_pixel_mask, args) for col in range(output_shape[2])]
        rreturn = [ray.get(jid) for jid in jobs]

        logging.info('Collecting and writing output')
        output_dat = np.zeros(chunk_shape,dtype=np.float32)
        for ret in rreturn:
            if ret[0] is not None:
                output_dat[:, 0, ret[1]] = ret[0][:,0]
    
        if args.mask_clouds_water and clouds_and_surface_water_mask is not None:
            logging.info('Masking clouds and water')
            output_dat = output_dat.transpose((0,2,1))
            output_dat[clouds_and_surface_water_mask,:] = 0 # could be nodata, but setting to 0 keeps maps continuous
            output_dat = output_dat.transpose((0,2,1))

        if args.mask_saturation and saturation is not None:
            logging.info('Masking saturation')
            output_dat = output_dat.transpose((0,2,1))
            output_dat[saturation,:] = 0 # could be nodata, but setting to 0 keeps maps continuous
            output_dat = output_dat.transpose((0,2,1))

        if args.mask_flares and saturation is not None:
            logging.info('Masking saturation')
            output_dat = output_dat.transpose((0,2,1))
            output_dat[dilated_saturation,:] = -1 # could be nodata, but setting to 0 keeps maps continuous
            output_dat[dilated_flare_mask,:] = -1 # could be nodata, but setting to 0 keeps maps continuous
            output_dat = output_dat.transpose((0,2,1))

        output_ds.write(output_dat, ce, chunk_shape)
        logging.info('Complete')


def np2envitype(np_dtype):
    _dtype = np.dtype(np_dtype).char
    return envi.dtype_to_envi[_dtype]


def cov(A,**kwargs):
    kwargs.setdefault('ddof',1)
    return np.cov(A.T,**kwargs)

def write_hotspot_vector(output_file, flares, saturation):
    # find center of hotspots from data
    labels_f = scipy.ndimage.label(flares)[0]
    un_labels_f = np.unique(labels_f)
    outdict = {"crs": {"properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}, "type": "name"},
               "features":[],
               "name":"radiance_hotspots",
               "type":"FeatureCollection"}
    for lab in un_labels_f[1:]:
        locs = np.where(labels_f == lab)
        outdict['features'].append({"geometry":{"coordinates":[np.mean(locs[1]),np.mean(locs[0]),0.0],"type":"Point"},
                                    "properties":{"hotspot_type":"flare"},
                                    "type":"Feature"})

    if saturation is not None:
        labels_s = scipy.ndimage.label(saturation)[0]
        un_labels_s = np.unique(labels_s)
        for lab in un_labels_s[1:]:
            locs = np.where(labels_s == lab)
            outdict['features'].append({"geometry":{"coordinates":[np.mean(locs[1]),np.mean(locs[0]),0.0],"type":"Point"},
                                        "properties":{"hotspot_type":"saturation"},
                                        "type":"Feature"})

    with open(output_file, 'w') as fout:
        fout.write(json.dumps(outdict, cls=SerialEncoder)) 


def fit_looshrinkage_alpha(data, alphas, I_reg=[]):
    # loocv shrinkage estimation via Theiler et al.
    stability_scaling=100.0 
    nchan = data.shape[1]

    nll = np.zeros(len(alphas))
    n = data.shape[0]
    
    X = data*stability_scaling
    S = cov(X)
    T = np.diag(np.diag(S)) if len(I_reg)==0 else cov(I_reg*stability_scaling)
        
    nchanlog2pi = nchan*np.log(2.0*np.pi)
    nll[:] = np.inf

    # Closed form for leave one out cross validation error
    for i,alpha in enumerate(alphas):
        try:
            # See Theiler, "The Incredible Shrinking Covariance Estimator",
            # Proc. SPIE, 2012. eqn. 29
            beta = (1.0-alpha) / (n-1.0)
            G_alpha = n * (beta*S) + (alpha*T)
            G_det = scipy.linalg.det(G_alpha, check_finite=False)
            if G_det==0:
                continue
            r_k  = (X.dot(scipy.linalg.inv(G_alpha, check_finite=False)) * X).sum(axis=1)
            q = 1.0 - beta * r_k
            nll[i] = 0.5*(nchanlog2pi+np.log(G_det))+1.0/(2.0*n) * \
                     (np.log(q)+(r_k/q)).sum()
        except np.linalg.LinAlgError:
            logging.warning('looshrinkage encountered a LinAlgError')

    mindex = np.argmin(nll)
    if nll[mindex]!=np.inf:
        alpha = alphas[mindex]
    else:
        mindex = -1
        alpha = 0.0
    
    return alpha


def apply_looshrinkage_alpha(data:np.array, alpha: float, I_reg=[]):
    """Calculate the covariance matrix using the shrinkage estimation via Theiler et al.

    Args:
        data (np.array): data to estimate covariance matrix from
        alpha (float): shrinkage parameter
        I_reg (list, optional):  Defaults to [].

    Returns:
        (np.array): covariance matrix
    """

    # Final nonregularized covariance and shrinkage target
    S = cov(data)
    T = None
    if len(I_reg)==0:
        T = np.diag(np.diag(S))
    else:
        T = cov(I_reg)
        
    # Final covariance 
    C = (1.0 - alpha) * S + alpha * T

    return C


def calculate_mf_covariance(radiance: np.array, model: str, fixed_alpha: float = None):
    """ Calculate covariance and mean of radiance data

    Args:
        radiance (np.array): radiance data

    Returns:
        tuple: (covariance, mean)
    """
    if model == 'looshrinkage':
        if fixed_alpha is None:
            alpha = fit_looshrinkage_alpha(radiance, (10.0 ** np.arange(-10,0+0.05,0.05)), I_reg=[])
        else:
            alpha = fixed_alpha
        C = apply_looshrinkage_alpha(radiance, alpha)
    elif model == 'empirical':
        C  = cov(radiance)
    else:
        logging.error('covariance model not recognized')
        sys.exit(0)

    return C


def get_mc_subset(mc_iteration: int, args, good_pixel_idx: np.array):
    if args.n_mc <= 1:
        cov_subset = good_pixel_idx
    else:
        if args.subsample_strategy == 'random':
            perm = np.random.permutation(len(good_pixel_idx))
            subset_size = int(args.mc_bag_fraction*len(good_pixel_idx))
            cov_subset = good_pixel_idx[perm[:subset_size]]
        elif args.subsample_strategy == 'spatial_blocks':
            splits = np.linspace(0,float(args.n_mc)-0.001,len(good_pixel_idx)).astype(int)
            subset_size = np.sum(splits != mc_iteration)
            cov_subset = good_pixel_idx[splits != mc_iteration]
    return cov_subset


def calculate_saturation_mask(bandmask_file: str, radiance: np.array, dilation_iterations=10, chunk_edges=None):
    '''l1b_bandmask marks static bad pixels and saturated pixels. The minimum subtraction below
    removes the contributions from static bad pixels, except in instances when the radiance
    has been otherwise flagged with bad values (-9999). The bad9999 mask identifies these and
    excludes them.'''

    if chunk_edges is None:
        l1b_bandmask_loaded = ReadAbstractDataSet(bandmask_file, netcdf_key = 'band_mask', envi_interleave = 'bip')[:,:,:].astype(np.uint8)
    else:
        l1b_bandmask_loaded = ReadAbstractDataSet(bandmask_file, netcdf_key = 'band_mask', envi_interleave = 'bip')[chunk_edges[0]:chunk_edges[1],:,:].astype(np.uint8)

    bad9999 = np.any(radiance < -1, axis = 1)
    l1b_bandmask_unpacked = np.unpackbits(l1b_bandmask_loaded, axis= -1)
    l1b_bandmask_summed = np.sum(l1b_bandmask_unpacked, axis = -1)
    max_vals = np.max(l1b_bandmask_summed, axis = 0)
    min_vals = np.min( np.where(bad9999, max_vals, l1b_bandmask_summed), axis = 0)
    saturation_mask = l1b_bandmask_summed - min_vals
    saturation_mask[bad9999] = 0
    dilated_saturation_mask = scipy.ndimage.binary_dilation(saturation_mask != 0, iterations = dilation_iterations) < 1
    return np.logical_not(dilated_saturation_mask), saturation_mask != 0


def calculate_flare_mask(radiance: np.array, preflagged_pixels: np.array, wavelengths: np.array):
    b270_idx = np.argmin(np.abs(wavelengths - 2389.486)) 
    hot_mask = np.where(np.logical_and(radiance[:,b270_idx,:] > 1.5, preflagged_pixels == True), 1., 0.)
    hot_mask_dilated = scipy.ndimage.uniform_filter(hot_mask, [5,5]) > 0.01
    return hot_mask_dilated, hot_mask


@ray.remote
def mf_one_column(col: int, rdn_full: np.array, absorption_coefficients: np.array, active_wl_idx: np.array, good_pixel_mask: np.array, args):
    """ Run the matched filter on a single column of the input image

    Args:
        col (int): column to run on
        rdn_full (np.array): full radiance dataset, bil interleave
        outimg_mm_shape (tuple): output image shape
        absorption_coefficients (np.array): absorption coefficients for target
        active_wl_idx (np.array): active wavelength indices
        good_pixel_mask (np.array): mask of valid pixels to use for covariance / mean estimates
        args (_type_): arguments from input

    Returns:
        (np.array): matched filter results from the column
    """

    logging.basicConfig(format='%(levelname)s:%(asctime)s ||| %(message)s', level=args.loglevel,
                        filename=args.logfile, datefmt='%Y-%m-%d,%H:%M:%S')
    logging.debug(f'Col: {col}')

    rdn = np.float64(rdn_full[:, active_wl_idx, col].copy())
    no_radiance_mask = np.all(np.logical_and(np.isfinite(rdn), rdn > -0.05), axis=1)
    good_pixel_idx = np.where(np.logical_and(good_pixel_mask[:,col], no_radiance_mask))[0]
    if len(good_pixel_idx) < 10:
        logging.debug('Too few good pixels found in col {col}: skipping')
        return None, None
    
    # array to hold results in
    mf_mc = np.ones((rdn.shape[0],args.n_mc)) * args.nodata_value

    np.random.seed(13)
    for _mc in range(args.n_mc):
        
        # get subset of pixels to use for covariance / mean estimates
        cov_subset = get_mc_subset(_mc, args, good_pixel_idx)

        # optional radiance adjustment for max radiance pcs...experimental
        if args.remove_dominant_pcs:
            pca_mean = rdn[cov_subset,:].mean(axis=0) 
            pcavals, pcavec = scipy.linalg.eigh(cov(rdn - pca_mean))
            loc_rdn = (rdn - pca_mean ) @ pcavec[:,:-5]
            target = (absorption_coefficients.copy() * pca_mean) @ pcavec[:,:-5]
        else:
            loc_rdn = rdn
            target = absorption_coefficients.copy() * np.mean(loc_rdn[cov_subset,:],axis=0)

        # calculate covariance and mean
        try:
            C = calculate_mf_covariance(loc_rdn[cov_subset,:], args.covariance_style, args.fixed_alpha)
            Cinv = scipy.linalg.inv(C, check_finite=False)
        except np.linalg.LinAlgError:
            logging.warn('singular matrix. skipping this column')
            return None, None
        mu = np.mean(loc_rdn[cov_subset,:], axis=0)

        # Matched filter time
        normalizer = target.dot(Cinv).dot(target.T)
        if args.ace_filter:
            rx = np.sum((loc_rdn[no_radiance_mask,:] - mu) @ Cinv * (loc_rdn[no_radiance_mask,:] - mu), axis = 1)
            normalizer = normalizer * rx

        mf = ((loc_rdn[no_radiance_mask,:] - mu).dot(Cinv).dot(target.T)) / normalizer
        
        # scale outputs
        mf_mc[no_radiance_mask,_mc] = mf * args.ppm_scaling
    
    output = np.vstack([np.mean(mf_mc,axis=-1), np.std(mf_mc,axis=-1)]).T
    output[np.logical_not(no_radiance_mask),:] = args.nodata_value

    logging.debug(f'Column {col} mean: {np.mean(output[good_pixel_idx,0])}')
    return output.astype(np.float32), col


if __name__ == '__main__':
    main()
    ray.shutdown()




