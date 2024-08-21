#! /usr/bin/env python
#
#  Copyright 2022 California Institute of Technology
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
# ISOFIT: Imaging Spectrometer Optimal FITting
# Authors: Markus Foote

# version working-6 with full modtran runs and warnings and optimizations
from os.path import exists
import numpy as np
import scipy.ndimage
import argparse
import spectral
import h5py


def check_param(value, min, max, name):
    if value < min or value > max:
        raise ValueError('The value for {name} exceeds the sampled parameter space.'
                         'The limits are[{min}, {max}], requested {value}.')


@np.vectorize
# [0.,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]
def get_5deg_zenith_angle_index(zenith_value):
    check_param(zenith_value, 0, 80, 'Zenith Angle')
    return zenith_value / 5


@np.vectorize
def get_5deg_sensor_height_index(sensor_value):  # [1, 2, 4, 10, 20, 120]
    # Only check lower bound here, atmosphere ends at 120 km so clamping there is okay.
    check_param(sensor_value, 1, np.inf, 'Sensor Height')
    # There's not really a pattern here, so just linearly interpolate between values -- piecewise linear
    if sensor_value < 1.0:
        return np.float64(0.0)
    elif sensor_value < 2.0:
        idx = sensor_value - 1.0
        return idx
    elif sensor_value < 4:
        return sensor_value / 2
    elif sensor_value < 10:
        return (sensor_value / 6) + (4.0 / 3.0)
    elif sensor_value < 20:
        return (sensor_value / 10) + 2
    elif sensor_value < 120:
        return (sensor_value / 100) + 3.8
    else:
        return 5


@np.vectorize
def get_5deg_ground_altitude_index(ground_value):  # [0, 0.5, 1.0, 2.0, 3.0]
    check_param(ground_value, 0, 3, 'Ground Altitude')
    if ground_value < 1:
        return 2 * ground_value
    else:
        return 1 + ground_value


@np.vectorize
def get_5deg_water_vapor_index(water_value):  # [0,1,2,3,4,5,6]
    check_param(water_value, 0, 6, 'Water Vapor')
    return water_value


@np.vectorize
# [0.0,1000,2000,4000,8000,16000,32000,64000]
def get_5deg_methane_index(methane_value):
    # the parameter clamps should rarely be calle because there are default concentrations, but the --concentraitons parameter exposes these
    check_param(methane_value, 0, 64000, 'Methane Concentration')
    if methane_value <= 0:
        return 0
    elif methane_value < 1000:
        return methane_value / 1000
    return np.log2(methane_value / 500)


@np.vectorize
def get_carbon_dioxide_index(coo_value):
    check_param(coo_value, 0, 1280000, 'Carbon Dioxode Concentration')
    if coo_value <= 0:
        return 0
    elif coo_value < 20000:
        return coo_value / 20000
    return np.log2(coo_value / 10000)


def get_5deg_lookup_index(zenith=0, sensor=120, ground=0, water=0, conc=0, gas='ch4'):
    if 'ch4' in gas:
        idx = np.asarray([[get_5deg_zenith_angle_index(zenith)],
                          [get_5deg_sensor_height_index(sensor)],
                          [get_5deg_ground_altitude_index(ground)],
                          [get_5deg_water_vapor_index(water)],
                          [get_5deg_methane_index(conc)]])
    elif 'co2' in gas:
        idx = np.asarray([[get_5deg_zenith_angle_index(zenith)],
                          [get_5deg_sensor_height_index(sensor)],
                          [get_5deg_ground_altitude_index(ground)],
                          [get_5deg_water_vapor_index(water)],
                          [get_carbon_dioxide_index(conc)]])
    else:
        raise ValueError('Unknown gas provided.')
    return idx


def spline_5deg_lookup(grid_data, zenith=0, sensor=120, ground=0, water=0, conc=0, gas='ch4', order=1):
    coords = get_5deg_lookup_index(
        zenith=zenith, sensor=sensor, ground=ground, water=water, conc=conc, gas=gas)
    # correct_lookup = np.asarray([scipy.ndimage.map_coordinates(
    #     im, coordinates=coords, order=order, mode='nearest') for im in np.moveaxis(grid_data, 5, 0)])
    if order == 1:
        coords_fractional_part, coords_whole_part = np.modf(coords)
        coords_near_slice = tuple((slice(int(c), int(c+2)) for c in coords_whole_part))
        near_grid_data = grid_data[coords_near_slice]
        new_coord = np.concatenate((coords_fractional_part * np.ones((1, near_grid_data.shape[-1])),
                                    np.arange(near_grid_data.shape[-1])[None, :]), axis=0)
        lookup = scipy.ndimage.map_coordinates(near_grid_data, coordinates=new_coord, order=1, mode='nearest')
    elif order == 3:
        lookup = np.asarray([scipy.ndimage.map_coordinates(
            im, coordinates=coords_fractional_part, order=order, mode='nearest') for im in np.moveaxis(near_grid_data, 5, 0)])
    return lookup.squeeze()

def load_ghg_dataset(ghg_hdf):
    datafile = h5py.File(ghg_hdf, 'r', rdcc_nbytes=4194304)
    return datafile['modtran_data'], datafile['modtran_param'], datafile['wave']

def load_pca_dataset():
    filename = 'modtran_ch4_full/dataset_ch4_pca.npz'
    correcthash = 'd5e9849157a00c220c26a8785789137d078a00ac749cc2b59c98bc7ece932815'
    import hashlib
    with open(filename, 'rb') as f:
        filehash = hashlib.sha256(f.read()).hexdigest()
    if correcthash != filehash:
        raise RuntimeError('Dataset file is invalid.')
    datafile = np.load(filename)
    reconstruct = datafile['scores'].dot(datafile['components'])
    parameters = datafile['parameters']
    wavelengths = datafile['wavelengths']
    simulation_spectra = reconstruct.reshape(
        parameters.shape[:-1] + wavelengths.shape)
    return simulation_spectra, parameters, wavelengths, 'ch4'

def generate_library(gas_concentration_vals, dataset, gas, zenith=0, sensor=120, ground=0, water=0, order=1):
    grid, params, wave = load_ghg_dataset(dataset)
    rads = np.empty((len(gas_concentration_vals), grid.shape[-1]))
    for i, ppmm in enumerate(gas_concentration_vals):
        rads[i, :] = spline_5deg_lookup(
            grid, zenith=zenith, sensor=sensor, ground=ground, water=water, conc=ppmm, gas=gas, order=order)
    return rads, wave

def generate_template_from_bands(centers, fwhm, params, dataset, gas, **kwargs):
    """Calculate a unit absorption spectrum for methane by convolving with given band information.

    :param centers: wavelength values for the band centers, provided in nanometers.
    :param fwhm: full width half maximum for the gaussian kernel of each band.
    :return template: the unit absorption spectum
    """
    # import scipy.stats
    SCALING = 1e5
    centers = np.asarray(centers)
    fwhm = np.asarray(fwhm)
    if np.any(~np.isfinite(centers)) or np.any(~np.isfinite(fwhm)):
        raise RuntimeError(
            'Band Wavelengths Centers/FWHM data contains non-finite data (NaN or Inf).')
    if centers.shape[0] != fwhm.shape[0]:
        raise RuntimeError(
            'Length of band center wavelengths and band fwhm arrays must be equal.')
#     lib = spectral.io.envi.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ch4.hdr'),
#                                 os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ch4.lut'))
#     rads = np.asarray(lib.asarray()).squeeze()
#     wave = np.asarray(lib.bands.centers)

    # Ignore None, better if it had just not been passed
    if 'concentrations' in kwargs and kwargs['concentrations'] is None:
        kwargs.pop('concentrations')
    concentrations = np.asarray(kwargs.get(
        'concentrations', [0.0, 1000, 2000, 4000, 8000, 16000, 32000, 64000]))
    rads, wave = generate_library(concentrations,
                                  dataset,
                                  gas,
                                  **params)
    # sigma = fwhm / ( 2 * sqrt( 2 * ln(2) ) )  ~=  fwhm / 2.355
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    # response = scipy.stats.norm.pdf(wave[:, None], loc=centers[None, :], scale=sigma[None, :])
    # Evaluate normal distribution explicitly
    var = sigma ** 2
    denom = (2 * np.pi * var) ** 0.5
    numer = np.exp(-(np.asarray(wave)[:, None] - centers[None, :])**2 / (2*var))
    response = numer / denom
    # Normalize each gaussian response to sum to 1.
    response = np.divide(response, response.sum(
        axis=0), where=response.sum(axis=0) > 0, out=response)
    # implement resampling as matrix multiply
    resampled = rads.dot(response)
    lograd = np.log(resampled, out=np.zeros_like(
        resampled), where=resampled > 0)
    slope, _, _, _ = np.linalg.lstsq(
        np.stack((np.ones_like(concentrations), concentrations)).T, lograd, rcond=None)
    spectrum = slope[1, :] * SCALING
    target = np.stack((np.arange(1, spectrum.shape[0]+1), centers, spectrum)).T
    return target


def main(input_args=None):
    parser = argparse.ArgumentParser(
        description='Create a unit absorption spectrum for specified parameters.')
    parser.add_argument('-z', '--zenith_angle', type=float, required=True,
                        help='Zenith Angle (in degrees) for generated spectrum.')
    parser.add_argument('-s', '--sensor_altitude', type=float,
                        required=True, help='Absolute Sensor Altitude (in km) above sea level.')
    parser.add_argument('-g', '--ground_elevation', type=float,
                        required=True, help='Ground Elevation (in km).')
    parser.add_argument('-w', '--water_vapor', type=float,
                        required=True, help='Column water vapor (in cm).')
    parser.add_argument('-l','--lut_dataset', type=str, required=True, help='GHG LUT path.')
    parser.add_argument('--order', choices=(1, 3), default=1,
                        type=int, required=False, help='Spline interpolation degree.')
    gas = parser.add_mutually_exclusive_group(required=False)
    gas.add_argument('--co2', action='store_const', dest='gas', const='co2')
    gas.add_argument('--ch4', action='store_const', dest='gas', const='ch4')
    wave = parser.add_mutually_exclusive_group(required=True)
    wave.add_argument(
        '--hdr', type=str, help='ENVI Header file for the flightline to match band centers/fwhm.')
    wave.add_argument('--txt', type=str,
                      help='Text-based table for band centers/fwhm.')
    parser.add_argument('--source', type=str,
                        choices=['full', 'pca'], default='full')
    parser.add_argument('-o', '--output', type=str,
                        default='generated_uas.txt', help='Output file to save spectrum.')
    parser.add_argument('--concentrations', type=float, default=None,
                        required=False, nargs='+', help='override the ppmm lookup values')
    parser.set_defaults(gas='ch4')
    args = parser.parse_args(input_args)
    param = {'zenith': args.zenith_angle,
             # Model uses sensor height above ground
             'sensor': args.sensor_altitude - args.ground_elevation,
             'ground': args.ground_elevation,
             'water': args.water_vapor,
             'order': args.order}
    if args.hdr and exists(args.hdr):
        image = spectral.io.envi.open(args.hdr)
        centers = np.array([float(x) for x in image.metadata['wavelength']])
        fwhm = np.array([float(x) for x in image.metadata['fwhm']])
    elif args.txt and exists(args.txt):
        data = np.loadtxt(args.txt, usecols=(0, 1),delimiter=',')
        centers = data[:, 0]
        fwhm = data[:, 1]
    else:
        raise RuntimeError(
            'Failed to load band centers and fwhm from file. Check that the specified file exists.')
    concentrations = args.concentrations

    uas = generate_template_from_bands(centers,
                                       fwhm,
                                       param,
                                       concentrations=concentrations,
                                       dataset=args.lut_dataset,
                                       gas = args.gas)


    np.savetxt(args.output, uas, delimiter=' ',
               fmt=('%03d', '% 10.3f', '%.18f'))


if __name__ == '__main__':
    main()
