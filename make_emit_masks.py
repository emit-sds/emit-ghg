"""
Mask generation for imaging spectroscopy, oriented towards EMIT.

Authors: David R. Thompson, david.r.thompson@jpl.nasa.gov,
         Philip G. Brodrick, philip.brodrick@jpl.nasa.gov
"""

import os
import argparse
from osgeo import gdal
import numpy as np
from spectral.io import envi
from isofit.core.sunposition import sunpos
from isofit.core.common import resample_spectrum
from datetime import datetime
from scipy.ndimage.morphology import distance_transform_edt
from emit_utils.file_checks import envi_header
import ray
import multiprocessing


def haversine_distance(lon1, lat1, lon2, lat2, radius=6335439):
    """ Approximate the great circle distance using Haversine formula

    :param lon1: point one longitude
    :param lat1: point one latitude
    :param lon2: point two longitude
    :param lat2: point two latitude
    :param radius: radius to use (default is approximate radius at equator)

    :return: great circle distance in radius units
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1

    d = 2 * radius * np.arcsin(np.sqrt(np.sin(delta_lat/2)**2 + np.cos(lat1)
                               * np.cos(lat2) * np.sin(delta_lon/2)**2))

    return d


@ray.remote
def build_line_masks(start_line: int, stop_line: int, rdnfile: str, locfile: str, atmfile: str, dt: datetime, h2o_band: np.array, aod_bands: np.array, pixel_size: float, outfile: str, wl: np.array, irr: np.array):
    # determine glint bands having negligible water reflectance
    BLUE = np.logical_and(wl > 440, wl < 460)
    NIR = np.logical_and(wl > 950, wl < 1000)
    SWIRA = np.logical_and(wl > 1250, wl < 1270)
    SWIRB = np.logical_and(wl > 1640, wl < 1660)
    SWIRC = np.logical_and(wl > 2200, wl < 2500)
    b450 = np.argmin(abs(wl-450))
    b762 = np.argmin(abs(wl-762))
    b780 = np.argmin(abs(wl-780))
    b1000 = np.argmin(abs(wl-1000))
    b1250 = np.argmin(abs(wl-1250))
    b1380 = np.argmin(abs(wl-1380))
    b1650 = np.argmin(abs(wl-1650))

    rdn_ds = envi.open(envi_header(rdnfile)).open_memmap(interleave='bil')
    loc_ds = envi.open(envi_header(locfile)).open_memmap(interleave='bil')

    if atmfile is not None:
        atm_ds = envi.open(envi_header(atmfile)).open_memmap(interleave='bil')

    return_mask = np.zeros((stop_line - start_line, 8, rdn_ds.shape[2]))
    for line in range(start_line, stop_line):
        print(f'{line} / {stop_line - start_line}')
        loc = loc_ds[line,...].copy().astype(np.float32).T
        rdn = rdn_ds[line,...].copy().astype(np.float32).T

        if atmfile is not None:
            atm = atm_ds[line,...].copy().astype(np.float32).T

        elevation_m = loc[:, 2]
        latitude = loc[:, 1]
        longitudeE = loc[:, 0]
        az, zen, ra, dec, h = sunpos(dt, latitude, longitudeE,
                                     elevation_m, radians=True).T

        rho = (((rdn * np.pi) / (irr.T)).T / np.cos(zen)).T

        rho[rho[:, 0] <= -9990, :] = -9999.0
        bad = (latitude <= -9990).T

        # Cloud threshold from Sandford et al.
        total = np.array(rho[:, b450] > 0.28, dtype=int) + \
            np.array(rho[:, b1250] > 0.46, dtype=int) + \
            np.array(rho[:, b1650] > 0.22, dtype=int)

        maskbands = 8
        mask = np.zeros((maskbands, rdn.shape[0]))
        mask[0, :] = total > 2

        # Cirrus Threshold from Gao and Goetz, GRL 20:4, 1993
        mask[1, :] = np.array(rho[:, b1380] > 0.1, dtype=int)

        # Water threshold as in CORAL
        mask[2, :] = np.array(rho[:, b1000] < 0.05, dtype=int)

        # Threshold spacecraft parts using their lack of an O2 A Band
        mask[3, :] = np.array(rho[:, b762]/rho[:, b780] > 0.8, dtype=int)

        max_cloud_height = 3000.0
        mask[4, :] = np.tan(zen) * max_cloud_height / pixel_size

        # AOD 550
        if atmfile is not None:
            mask[5, :] = atm[:, aod_bands].sum(axis=1)
            mask[6, :] = atm[:, h2o_band].T
        else:
            mask[5, :] = -1
            mask[6, :] = -1

        # Remove water and spacecraft flags if cloud flag is on (mostly cosmetic)
        mask[2:4, np.logical_or(mask[0,:] == 1, mask[1,:] ==1)] = 0

        mask[:, bad] = -9999.0
        return_mask[line - start_line,...] = mask.copy()

    return return_mask, start_line, stop_line


# parser = argparse.ArgumentParser()
# args = parser.parse_args([])
# args.rdnfile = f'/Users/achlus/data1/av3/2023/20230711/AV320230711t225833/L1B_RDN/AV320230711t225833_000_L1B_RDN_01f22eae_RDN'
# args.locfile = f'/Users/achlus/data1/av3/2023/20230711/AV320230711t225833/L1B_ORT/AV320230711t225833_L1B_ORT_d69040b4_LOC'
# args.outfile = "/Users/achlus/data1/temp/AV320230711t225833_L2A_MSK_01f22eae_MASK"
# args.irrfile = "/Users/achlus/data1/temp/kurucz_0.1nm.dat"
# args.atmfile = None
# args.wavelengths = None
# args.n_cores =  -1
# args.aerosol_threshold = 0.5

def main():

    parser = argparse.ArgumentParser(description="Remove glint")
    parser.add_argument('rdnfile', type=str, metavar='RADIANCE')
    parser.add_argument('locfile', type=str, metavar='LOCATIONS')
    parser.add_argument('irrfile', type=str, metavar='SOLAR_IRRADIANCE')
    parser.add_argument('outfile', type=str, metavar='OUTPUT_MASKS')
    parser.add_argument('--atmfile', type=str, metavar='SUBSET_LABELS', default=None)
    parser.add_argument('--wavelengths', type=str, default=None)
    parser.add_argument('--n_cores', type=int, default=-1)
    parser.add_argument('--aerosol_threshold', type=float, default=0.5)
    args = parser.parse_args()

    rdn_hdr = envi.read_envi_header(envi_header(args.rdnfile))
    rdn_shp = envi.open(envi_header(args.rdnfile)).open_memmap(interleave='bil').shape

    aod_bands, h2o_band = [], []

    if args.atmfile is not None:

        atm_hdr = envi.read_envi_header(envi_header(args.atmfile))
        atm_shp = envi.open(envi_header(args.atmfile)).open_memmap(interleave='bil').shape

        if atm_shp[0] != rdn_shp[0] or atm_shp[2] != rdn_shp[2]:
            raise ValueError('Label and input file dimensions do not match.')

        # Find H2O and AOD elements in state vector
        for i, name in enumerate(atm_hdr['band names']):
            if 'H2O' in name:
                h2o_band.append(i)
            elif 'AER' in name or 'AOT' in name or 'AOD' in name:
                aod_bands.append(i)

    loc_shp = envi.open(envi_header(args.locfile)).open_memmap(interleave='bil').shape

    # Check file size consistency
    if loc_shp[0] != rdn_shp[0] or loc_shp[2] != rdn_shp[2]:
        raise ValueError('LOC and input file dimensions do not match.')
    if loc_shp[1] != 3:
        raise ValueError('LOC file should have three bands.')

    # Get wavelengths and bands
    if args.wavelengths is not None:
        c, wl, fwhm = np.loadtxt(args.wavelengths).T
    else:
        if not 'wavelength' in rdn_hdr:
            raise IndexError('Could not find wavelength data anywhere')
        else:
            wl = np.array([float(f) for f in rdn_hdr['wavelength']])
        if not 'fwhm' in rdn_hdr:
            raise IndexError('Could not find fwhm data anywhere')
        else:
            fwhm = np.array([float(f) for f in rdn_hdr['fwhm']])

    # find pixel size
    if 'map info' in rdn_hdr.keys():
        pixel_size = float(rdn_hdr['map info'][5].strip())
    else:
        loc_memmap = envi.open(envi_header(args.locfile)).open_memmap(interleave='bip')
        center_y = int(loc_shp[0]/2)
        center_x = int(loc_shp[2]/2)
        center_pixels = loc_memmap[center_y-1:center_y+1, center_x, :2]
        pixel_size = haversine_distance(
            center_pixels[0, 1], center_pixels[0, 0], center_pixels[1, 1], center_pixels[1, 0])
        del loc_memmap, center_pixels

    # find solar zenith
    fid = os.path.split(args.rdnfile)[1].split('_')[0]
    for prefix in ['prm', 'ang', 'emit','AV3']:
        fid = fid.replace(prefix, '')
    dt = datetime.strptime(fid, '%Y%m%dt%H%M%S')

    day_of_year = dt.timetuple().tm_yday
    print(day_of_year, dt)

    # convert from microns to nm
    if not any(wl > 100):
        wl = wl*1000.0

    # irradiance
    irr_wl, irr = np.loadtxt(args.irrfile, comments='#').T
    irr = irr / 10  # convert to uW cm-2 sr-1 nm-1
    irr_resamp = resample_spectrum(irr, irr_wl, wl, fwhm)
    irr_resamp = np.array(irr_resamp, dtype=np.float32)

    rdn_dataset = gdal.Open(args.rdnfile, gdal.GA_ReadOnly)
    maskbands = 8

    # Build output dataset
    driver = gdal.GetDriverByName('ENVI')
    driver.Register()

    outDataset = driver.Create(args.outfile, rdn_shp[2], rdn_shp[0], maskbands, gdal.GDT_Float32, options=['INTERLEAVE=BIL'])
    outDataset.SetProjection(rdn_dataset.GetProjection())
    outDataset.SetGeoTransform(rdn_dataset.GetGeoTransform())
    del outDataset

    rayargs = {'local_mode': args.n_cores == 1}
    if args.n_cores <= 0:
        args.n_cores = multiprocessing.cpu_count()
    rayargs['num_cpus'] = args.n_cores
    ray.init(**rayargs)

    linebreaks = np.linspace(0, rdn_shp[0], num=args.n_cores*3).astype(int)

    irrid = ray.put(irr_resamp)
    jobs = [build_line_masks.remote(linebreaks[_l], linebreaks[_l+1], args.rdnfile, args.locfile, args.atmfile, dt, h2o_band, aod_bands, pixel_size, args.outfile, wl, irrid) for _l in range(len(linebreaks)-1)]
    rreturn = [ray.get(jid) for jid in jobs]
    ray.shutdown()

    mask = np.zeros((rdn_shp[0], maskbands, rdn_shp[2]))
    for lm, start_line, stop_line in rreturn:
        mask[start_line:stop_line,...] = lm

    bad = np.squeeze(mask[:, 0, :]) <= -9990
    good = np.squeeze(mask[:, 0, :]) > -9990

    # Create buffer around clouds (main and cirrus)
    cloudinv = np.logical_not(np.squeeze(np.logical_or(mask[:, 0, :], mask[:,1,:])))
    cloudinv[bad] = 1
    cloud_distance = distance_transform_edt(cloudinv)
    invalid = (np.squeeze(mask[:, 4, :]) >= cloud_distance)
    mask[:, 4, :] = invalid.copy()

    # Combine Cloud, Cirrus, Water, Spacecraft, and Buffer masks
    mask[:, 7, :] = np.logical_or(np.sum(mask[:,0:5,:], axis=1) > 0, mask[:,5,:] > args.aerosol_threshold)

    hdr = rdn_hdr.copy()
    hdr['bands'] = str(maskbands)
    hdr['band names'] = ['Cloud flag', 'Cirrus flag', 'Water flag',
                         'Spacecraft Flag', 'Dilated Cloud Flag',
                         'AOD550', 'H2O (g cm-2)', 'Aggregate Flag']

    hdr['interleave'] = 'bil'
    del hdr['wavelength']
    del hdr['fwhm']
    envi.write_envi_header(envi_header(args.outfile), hdr)
    mask.astype(dtype=np.float32).tofile(args.outfile)


if __name__ == "__main__":
    main()
