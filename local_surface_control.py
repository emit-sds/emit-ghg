# David R Thompson
# Summer 2017
# Jet Propulsion Laboratory
from spectral.io import envi 
import scipy as s
from sklearn.cross_decomposition import PLSRegression
from sklearn import linear_model
from sklearn import tree
import argparse
from utils import envi_header
import ray
from datetime import datetime
import numpy as np
import os
from isofit.core.common import resample_spectrum
from isofit.core.sunposition import sunpos
from scipy.ndimage.morphology import distance_transform_edt


@ray.remote
def build_line_masks(start_line: int, stop_line: int, rdnfile: str, locfile: str, dt: datetime, pixel_size: float, wl: np.array, irr: np.array, ret_rho=False):
    # determine glint bands having negligible water reflectance
    b450 = np.argmin(abs(wl-450))
    b762 = np.argmin(abs(wl-762))
    b780 = np.argmin(abs(wl-780))
    b1000 = np.argmin(abs(wl-1000))
    b1250 = np.argmin(abs(wl-1250))
    b1380 = np.argmin(abs(wl-1380))
    b1650 = np.argmin(abs(wl-1650))

    rdn_ds = envi.open(envi_header(rdnfile)).open_memmap(interleave='bil')
    loc_ds = envi.open(envi_header(locfile)).open_memmap(interleave='bil')

    return_mask = np.zeros((stop_line - start_line, 8, rdn_ds.shape[2]))
    return_rho = None
    if ret_rho:
        return_rho = np.zeros((stop_line - start_line, rdn_ds.shape[1], rdn_ds.shape[2]))
    for line in range(start_line, stop_line):
        #print(f'{line} / {stop_line - start_line}')
        loc = loc_ds[line,...].copy().astype(np.float32).T
        rdn = rdn_ds[line,...].copy().astype(np.float32).T

        elevation_m = loc[:, 2]
        latitude = loc[:, 1]
        longitudeE = loc[:, 0]
        az, zen, ra, dec, h = sunpos(dt, latitude, longitudeE,
                                     elevation_m, radians=True).T

        rho = (((rdn * np.pi) / (irr.T)).T / np.cos(zen)).T

        rho[rho[:, 0] < -9990, :] = -9999.0

        if ret_rho:
            return_rho[line - start_line,...] = rho.copy().T
        bad = (latitude < -9990).T

        # aggressive
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
        #mask[5, :] = x[:, aod_bands].sum(axis=1)
        aerosol_threshold = 0.4

        #mask[6, :] = x[:, h2o_band].T
        mask[6,:] = rdn[:,0] < -10

        mask[7, :] = np.array((mask[0, :] + mask[2, :] +
                               (mask[3, :] > aerosol_threshold)) > 0, dtype=int)
        mask[:, bad] = -9999.0
        return_mask[line - start_line,...] = mask.copy()

    return return_mask, start_line, stop_line, return_rho


def get_mask(rdnfile, locfile, irrfile):

    rdn_hdr = envi.read_envi_header(envi_header(rdnfile))
    rdn_shp = envi.open(envi_header(rdnfile)).open_memmap(interleave='bil').shape
    loc_shp = envi.open(envi_header(locfile)).open_memmap(interleave='bil').shape

    # find solar zenith
    fid = os.path.split(rdnfile)[1].split('_')[0]
    for prefix in ['prm', 'ang', 'emit']:
        fid = fid.replace(prefix, '')
    dt = datetime.strptime(fid, '%Y%m%dt%H%M%S')

    day_of_year = dt.timetuple().tm_yday
    print(day_of_year, dt)

    wl = np.array([float(x) for x in rdn_hdr['wavelength']])
    fwhm = np.array([float(x) for x in rdn_hdr['fwhm']])
    # convert from microns to nm
    if not any(wl > 100):
        wl = wl*1000.0
        fwhm = fwhm*1000.0

    # irradiance
    irr_wl, irr = np.loadtxt(irrfile, comments='#').T
    irr = irr / 10  # convert to uW cm-2 sr-1 nm-1
    irr_resamp = resample_spectrum(irr, irr_wl, wl, fwhm)
    irr_resamp = np.array(irr_resamp, dtype=np.float32)

    # find pixel size
    if 'map info' in rdn_hdr.keys():
        pixel_size = float(rdn_hdr['map info'][5].strip())
    else:
        loc_memmap = envi.open(envi_header(locfile)).open_memmap(interleave='bip')
        center_y = int(loc_shp[0]/2)
        center_x = int(loc_shp[2]/2)
        center_pixels = loc_memmap[center_y-1:center_y+1, center_x, :2]
        pixel_size = haversine_distance(
            center_pixels[0, 1], center_pixels[0, 0], center_pixels[1, 1], center_pixels[1, 0])
        del loc_memmap, center_pixels



    linebreaks = np.linspace(0, rdn_shp[0], num=40*3).astype(int)

    irrid = ray.put(irr_resamp)
    jobs = [build_line_masks.remote(linebreaks[_l], linebreaks[_l+1], rdnfile, locfile, dt, pixel_size, wl, irrid, True) for _l in range(len(linebreaks)-1)]
    rreturn = [ray.get(jid) for jid in jobs]
    ray.shutdown()


    mask = np.zeros((rdn_shp[0], 8, rdn_shp[2]))
    rho = np.zeros(rdn_shp)
    for lm, start_line, stop_line, lr in rreturn:
        mask[start_line:stop_line,...] = lm 
        rho[start_line:stop_line,...] = lr

    outmask = np.sum(mask[:, 0:3,:],axis=1) > 0
    #outmask[mask[:,6] == 1] = True
    #cloud_distance = distance_transform_edt(outmask)
    #invalid = (np.squeeze(mask[:, 4, :]) >= cloud_distance)
    #outmask[invalid] = True

    return outmask, rho

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
def local_model(x_full, y_full, good_full, start_l, end_l, start_c, end_c):

    x = x_full[start_l:end_l,start_c:end_c,:]
    y = y_full[start_l:end_l,start_c:end_c]
    ret_y = y.copy()
    good = good_full[start_l:end_l,start_c:end_c]

    if np.sum(good) > 10:
        x = x.reshape((x.shape[0]*x.shape[1],x.shape[2]))
        y = y.reshape((y.shape[0]*y.shape[1],1))
        good = good.reshape((good.shape[0]*good.shape[1]))
        good[np.any(np.isfinite(x) == False,axis=1)] = False
        good[np.isfinite(y.flatten()) == False] = False
        
        reg = linear_model.LinearRegression()
        reg.fit(x[good,:],y[good,:])

        pred = reg.predict(x)

        pred = pred.reshape((end_l-start_l,end_c-start_c))
        good = good.reshape((end_l-start_l,end_c-start_c))
        print(np.mean(pred))

        ret_y[good] -= pred[good]
    else:
        print(f'no good found: {np.sum(good)}')

    return ret_y, start_l, end_l, start_c, end_c


def subtract_local_model(ray_x, ray_y, ray_good, shape, l_chunk=160, c_chunk=160):


    jobs = []
    for line_start in range(0,shape[0],l_chunk):
        for col_start in range(0,shape[1],c_chunk):
            jobs.append(local_model.remote(ray_x, ray_y, ray_good, line_start, min(line_start + l_chunk,shape[0]), col_start, min(col_start + c_chunk,shape[1])))
            
    rreturn = [ray.get(jid) for jid in jobs]


    output = np.zeros(shape)
    for ret, start_line, stop_line, start_col, stop_col in rreturn:
        output[start_line:stop_line, start_col:stop_col] = ret

    return output



def main(input_args=None):
  parser = argparse.ArgumentParser(description="Control for surface")
  parser.add_argument('cmf', type=str,  metavar='CMF',
                      help='path to input image')   
  parser.add_argument('rdnfile', type=str,  
                      help='path to radiance file')
  parser.add_argument('locfile', type=str,  
                      help='path to location file')
  parser.add_argument('irrfile', type=str,  
                      help='path to irradiance file')
  parser.add_argument('output', type=str,  metavar='OUTPUT',
                      help='path for revised output image (mf ch4 ppm)')    
  parser.add_argument('--n_cores', type=int,  default=-1, metavar='num_cores',
                      help='number of cores to use')    
  parser.add_argument('--type', type=str,  default='ch4', choices=['ch4','co2'])
  args = parser.parse_args(input_args)


  if args.n_cores == -1:
    import multiprocessing
    args.n_cores = multiprocessing.cpu_count() - 1 
    
  rayargs = {'ignore_reinit_error': True, 'num_cpus': args.n_cores, 'include_dashboard': False}
  ray.init(**rayargs)

  mask, rfl = get_mask(args.rdnfile, args.locfile, args.irrfile)
  rfl = rfl.transpose((0,2,1))

  cmf_ds = envi.open(envi_header(args.cmf))
  cmf = np.squeeze(cmf_ds.open_memmap(interleave='bip').copy())

  print(mask.shape, cmf.shape, rfl.shape)

  wl = s.array([float(f) for f in envi.open(envi_header(args.rdnfile)).metadata['wavelength']])

  if args.type == 'ch4':
    active = s.where(s.logical_or(s.logical_and(wl>380,wl<1250), 
                        s.logical_or(s.logical_and(wl>1500,wl<1610),
                          s.logical_and(wl>2030,wl<2140))))[0]
  elif args.type == 'co2':

    active = s.where(s.logical_or(s.logical_and(wl>380,wl<=1190), 
                        s.logical_or(s.logical_and(wl>=1630,wl<=1700),
                          s.logical_and(wl>2130,wl<2500))))[0]
  else:
    raise AttributeError('Invalid type')
    

  rfl = rfl[...,active]

  good = np.logical_and.reduce((cmf != -9999, np.logical_not(mask), rfl[...,-1] > 0.02))
  #good = np.logical_and.reduce((cmf != -9999, np.logical_not(mask)))

  rfl_id = ray.put(rfl)
  cmf_id = ray.put(cmf)
  good_id = ray.put(good)

  subtracted_cmf = subtract_local_model(rfl_id, cmf_id, good_id, cmf.shape)

  subtracted_cmf[mask == 1] = 0
  subtracted_cmf[np.logical_and(subtracted_cmf != -9999, subtracted_cmf < 0)] = 0

  outmeta = cmf_ds.metadata
  outmeta['description'] = 'masked  / loc filtered matched filter results'
  outimg = envi.create_image(envi_header(args.output),outmeta,force=True,ext='')
  out_mm = outimg.open_memmap(interleave='bip', writable=True)
  out_mm[...,0] = subtracted_cmf
  del out_mm

if __name__ == "__main__":
  main()
