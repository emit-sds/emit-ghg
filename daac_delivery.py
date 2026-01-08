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
# Author: Winston Olson-Duvall, winston.olson-duvall@jpl.nasa.gov

import argparse
import datetime
import hashlib
import json
import os
import subprocess

import numpy as np
import spectral.io.envi as envi

from osgeo import gdal

from emit_main.workflow.workflow_manager import WorkflowManager


def envi_header(inputpath):
    """
    Convert a envi binary/header path to a header, handling extensions
    Args:
        inputpath: path to envi binary file
    Returns:
        str: the header file associated with the input reference.

    """
    if os.path.splitext(inputpath)[-1] == '.img' or os.path.splitext(inputpath)[-1] == '.dat' or os.path.splitext(inputpath)[-1] == '.raw':
        # headers could be at either filename.img.hdr or filename.hdr.  Check both, return the one that exists if it
        # does, if not return the latter (new file creation presumed).
        hdrfile = os.path.splitext(inputpath)[0] + '.hdr'
        if os.path.isfile(hdrfile):
            return hdrfile
        elif os.path.isfile(inputpath + '.hdr'):
            return inputpath + '.hdr'
        return hdrfile
    elif os.path.splitext(inputpath)[-1] == '.hdr':
        return inputpath
    else:
        return inputpath + '.hdr'


def get_band_mean(input_file: str, band) -> float:
    """
    Determines the mean of a band
    Args:
        input_file (str): obs file (EMIT style)
        band (int, optional): Band number retrieve average from.
    Returns:
        float: mean value of given band
    """
    ds = envi.open(envi_header(input_file))
    target = ds.open_memmap(interleave='bip')[..., band]

    good = target > -9990

    return np.mean(target[good])


def initialize_ummg(granule_name: str, creation_time: datetime, collection_name: str, collection_version: str,
                    start_time: datetime, stop_time: datetime, pge_name: str, pge_version: str,
                    sds_software_build_version: str = None, ghg_software_build_version: str = None,
                    ghg_software_delivery_version: str = None, doi: str = None,
                    orbit: int = None, orbit_segment: int = None, scene: int = None, solar_zenith: float = None,
                    solar_azimuth: float = None, water_vapor: float = None, aod: float = None,
                    mean_fractional_cover: float = None, mean_spectral_abundance: float = None,
                    cloud_fraction: str = None, source_scenes: list = None, plume_id: int = None):
    """ Initialize a UMMG metadata output file
    Args:
        granule_name: granule UR tag
        creation_time: creation timestamp
        collection_name: short name of collection reference
        collection_version: collection version
        sds_software_build_version: version of software build
        pge_name: PGE name  from build configuration
        pge_version: PGE version from build configuration
        cloud_fraction: rounded fraction of cloudcover if applicable

    Returns:
        dictionary representation of ummg
    """

    ummg = {"ProviderDates": []}
    ummg['MetadataSpecification'] = {'URL': 'https://cdn.earthdata.nasa.gov/umm/granule/v1.6.5', 'Name': 'UMM-G',
                                     'Version': '1.6.5'}

    ummg['Platforms'] = [{'ShortName': 'ISS', 'Instruments': [{'ShortName': 'EMIT Imaging Spectrometer'}]}]
    ummg['GranuleUR'] = granule_name

    if "PLM" in collection_name:
        ummg['TemporalExtent'] = {
            'SingleDateTime': start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
    else:
        ummg['TemporalExtent'] = {
            'RangeDateTime': {
                'BeginningDateTime': start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                'EndingDateTime': stop_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            }
        }

    # Use ProviderDate type "Update" per DAAC. Use this for data granule ProductionDateTime field too.
    ummg['ProviderDates'].append({'Date': creation_time.strftime("%Y-%m-%dT%H:%M:%SZ"), 'Type': "Update"})
    ummg['CollectionReference'] = {
        "ShortName": collection_name,
        "Version": str(collection_version)
    }

    # First attribute is required and others may be optional
    ummg['AdditionalAttributes'] = []
    if sds_software_build_version is not None:
        ummg['AdditionalAttributes'].append(
            {'Name': 'EMIT_SDS_SOFTWARE_BUILD_VERSION', 'Values': [str(sds_software_build_version)]})
    if ghg_software_build_version is not None:
        ummg['AdditionalAttributes'].append(
            {'Name': 'EMIT_GHG_SOFTWARE_BUILD_VERSION', 'Values': [str(ghg_software_build_version)]})
    if ghg_software_delivery_version is not None:
        ummg['AdditionalAttributes'].append(
            {'Name': 'EMIT_GHG_SOFTWARE_DELIVERY_VERSION', 'Values': [str(ghg_software_delivery_version)]})
    if doi is not None:
        ummg['AdditionalAttributes'].append({'Name': 'Identifier_product_doi_authority', 'Values': ["https://doi.org"]})
        ummg['AdditionalAttributes'].append({'Name': 'Identifier_product_doi', 'Values': [str(doi)]})
    if source_scenes is not None:
        ummg['AdditionalAttributes'].append({'Name': 'SOURCE_SCENES', 'Values': source_scenes})
    if plume_id is not None:
        ummg['AdditionalAttributes'].append({'Name': 'PLUME_ID', 'Values': [str(plume_id)]})
    if orbit is not None:
        ummg['AdditionalAttributes'].append({'Name': 'ORBIT', 'Values': [str(orbit)]})
    if orbit_segment is not None:
        ummg['AdditionalAttributes'].append({'Name': 'ORBIT_SEGMENT', 'Values': [str(orbit_segment)]})
    if scene is not None:
        ummg['AdditionalAttributes'].append({'Name': 'SCENE', 'Values': [str(scene)]})
    if solar_zenith is not None:
        ummg['AdditionalAttributes'].append({'Name': 'SOLAR_ZENITH', 'Values': [f"{solar_zenith:.2f}"]})
    if solar_azimuth is not None:
        ummg['AdditionalAttributes'].append({'Name': 'SOLAR_AZIMUTH', 'Values': [f"{solar_azimuth:.2f}"]})
    if water_vapor is not None:
        ummg['AdditionalAttributes'].append({'Name': 'WATER_VAPOR', 'Values': [f"{water_vapor:.2f}"]})
    if aod is not None:
        ummg['AdditionalAttributes'].append({'Name': 'AEROSOL_OPTICAL_DEPTH', 'Values': [f"{aod:.2f}"]})
    if mean_fractional_cover is not None:
        ummg['AdditionalAttributes'].append(
            {'Name': 'MEAN_FRACTIONAL_COVER', 'Values': [f"{mean_fractional_cover:.2f}"]})
    if mean_spectral_abundance is not None:
        ummg['AdditionalAttributes'].append(
            {'Name': 'MEAN_SPECTRAL_ABUNDANCE', 'Values': [f"{mean_spectral_abundance:.2f}"]})

    ummg['PGEVersionClass'] = {'PGEName': pge_name, 'PGEVersion': pge_version}

    if cloud_fraction is not None:
        ummg['CloudCover'] = int(cloud_fraction)

    return ummg


def calc_checksum(path, hash_alg="sha512"):
    checksum = {}
    if hash_alg.lower() == "sha512":
        h = hashlib.sha512()
    with open(path, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            h.update(byte_block)
    return h.hexdigest()


def add_data_files_ummg(ummg: dict, data_file_names: list, daynight: str, file_formats: list =['NETCDF-4']):
    """
    Add boundary points list to UMMG in correct format
    Args:
        ummg: existing UMMG to augment
        data_file_names: list of paths to existing data files to add
        file_formats: description of file types

    Returns:
        dictionary representation of ummg with new data granule
    """

    if len(data_file_names) != len(file_formats):
        err = f'Length of data_file_names must match length of file_formats.  Currentely lengths are: {len(data_file_names)} and {len(file_formats)}'
        raise AttributeError(err)

    prod_datetime_str = None
    for subdict in ummg['ProviderDates']:
        if subdict['Type'] == 'Update':
            prod_datetime_str = subdict['Date']
            break

    archive_info = []
    for filename, fileformat in zip(data_file_names, file_formats):
        archive_info.append({
                             "Name": os.path.basename(filename[1]),
                             "SizeInBytes": os.path.getsize(filename[0]),
                             "Format": fileformat,
                             "Checksum": {
                                 'Value': calc_checksum(filename[0]),
                                 'Algorithm': 'SHA-512'
                                 }
                            })

    ummg['DataGranule'] = {
        'DayNightFlag': daynight,
        'ArchiveAndDistributionInformation': archive_info
    }

    if prod_datetime_str is not None:
        ummg['DataGranule']['ProductionDateTime'] = prod_datetime_str

    return ummg


def add_boundary_ummg(ummg: dict, boundary_points: list):
    """
    Add boundary points list to UMMG in correct format
    Args:
        ummg: existing UMMG to augment
        boundary_points: list of lists, each major list entry is a pair of (lon, lat) coordinates

    Returns:
        dictionary representation of ummg
    """


    formatted_points_list = []
    for point in boundary_points:
        formatted_points_list.append({'Longitude': point[0], 'Latitude': point[1]})

    # For GPolygon, if the first and last points are not equal, add the first point again to close out
    if boundary_points[0] != boundary_points[-1]:
        formatted_points_list.append({'Longitude': boundary_points[0][0], 'Latitude': boundary_points[0][1]})

    hsd = {'HorizontalSpatialDomain':
              {"Geometry":
                  {'GPolygons': [
                      {'Boundary':
                           {'Points': formatted_points_list}}
                  ]}
              }
          }


    ummg['SpatialExtent'] = hsd
    return ummg


class SerialEncoder(json.JSONEncoder):
    """Encoder for json to help ensure json objects can be passed to the workflow manager.
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return super(SerialEncoder, self).default(obj)


def stage_files(wm, acq, files):
    # Copy to S3 and rename
    for f in files:
        cmd_aws_s3 = [wm.config["aws_cli_exe"], "s3", "cp", f[0], acq.aws_s3_uri_base + f[1], "--profile", wm.config["aws_profile"]]
        output = subprocess.run(" ".join(cmd_aws_s3), shell=True, capture_output=True)
        if output.returncode != 0:
            print(f"cmd_aws_s3: {' '.join(cmd_aws_s3)}")
            print("stage_files run command failed: %s" % (output.args))
            raise RuntimeError(output.stderr.decode("utf-8"))


def submit_cnm_notification(wm, acq, base_dir, granule_ur, files, formats, collection, collection_version):
    # Build notification dictionary
    utc_now = datetime.datetime.now(tz=datetime.timezone.utc)
    cnm_submission_id = f"{granule_ur}_{utc_now.strftime('%Y%m%dt%H%M%S')}"
    cnm_submission_path = os.path.join(base_dir, cnm_submission_id + "_cnm.json")

    provider = wm.config["daac_provider_forward"]
    queue_url = wm.config["daac_submission_url_forward"]

    notification = {
        "collection": collection,
        "provider": provider,
        "identifier": cnm_submission_id,
        "version": wm.config["cnm_version"],
        "product": {
            "name": granule_ur,
            "dataVersion": collection_version,
            "files": []
        }
    }

    for i, f in enumerate(files):
        notification["product"]["files"].append(
            {
                "name": f[1],
                "uri": acq.aws_s3_uri_base + f[1],
                "type": formats[i],
                "size": os.path.getsize(f[0]),
                "checksumType": "sha512",
                "checksum": calc_checksum(f[0], "sha512")
            }
        )

    # Write notification submission to file
    print(f"Writing CNM notification file to {cnm_submission_path}")
    with open(cnm_submission_path, "w") as f:
        f.write(json.dumps(notification, indent=4))
    wm.change_group_ownership(cnm_submission_path)

    # Submit notification via AWS SQS
    print(f"Submitting CNM notification via AWS SQS")
    cnm_submission_output = cnm_submission_path.replace(".json", ".out")
    cmd_aws = [wm.config["aws_cli_exe"], "sqs", "send-message", "--queue-url", queue_url, "--message-body",
               f"file://{cnm_submission_path}", "--profile", wm.config["aws_profile"], ">", cnm_submission_output]
    # print(f"cmd_aws: {' '.join(cmd_aws)}")
    output = subprocess.run(" ".join(cmd_aws), shell=True, capture_output=True)
    if output.returncode != 0:
        raise RuntimeError(output.stderr.decode("utf-8"))


def deliver_plm(base_dir, fname, wm, ghg_config):
    if "CH4" in fname:
        GHG = "CH4"
    elif "CO2" in fname:
        GHG = "CO2"
    else:
        print("Unable to match a GHG for plm delivery.  Returning...")
        return
    print(f"Doing {GHG} plm delivery with fname {fname}")

    # Get scene details and create Granule UR
    # Example granule_ur: EMIT_L2B_{CH4,CO2}PLM_001_20230805T195459_000109
    acq = wm.acquisition
    plume_id = str(int(fname.split(".")[0].split("-")[1])).zfill(6)
    collection = f"EMITL2B{GHG}PLM"
    prod_type = f"{GHG}PLM"
    granule_ur = f"EMIT_L2B_{prod_type}_{ghg_config['collection_version']}_{acq.start_time.strftime('%Y%m%dT%H%M%S')}_{plume_id}"

    # Create local/tmp daac names and paths
    local_plm_path = os.path.join(base_dir, fname)
    local_geojson_path = local_plm_path.replace(".tif", ".json")
    local_browse_path = local_plm_path.replace(".tif", ".png")
    local_ummg_path = local_plm_path.replace(".tif", ".cmr.json")
    daac_plm_name = f"{granule_ur}.tif"
    daac_geojson_name = f"{granule_ur.replace(f'{GHG}PLM', f'{GHG}PLMMETA')}.json"
    daac_browse_name = f"{granule_ur}.png"
    daac_ummg_name = f"{granule_ur}.cmr.json"
    # daac_ummg_path = os.path.join(base_dir, daac_ummg_name)
    files = [(local_plm_path, daac_plm_name),
             (local_geojson_path, daac_geojson_name),
             (local_browse_path, daac_browse_name),
             (local_ummg_path, daac_ummg_name)]
    # print(f"files: {files}")

    # Get the software_build_version (extended build num when GHG enhancement data product was created)
    hdr = envi.read_envi_header(acq.ch4_hdr_path) if GHG == "CH4" else envi.read_envi_header(acq.co2_hdr_path)
    sds_software_build_version = hdr["emit software build version"]

    # Get ghg_software_build_version
    ds = gdal.Open(os.path.join(base_dir, fname))
    meta = ds.GetMetadata()
    ghg_software_build_version = meta["software_build_version"]

    # Get source scenes
    source_scenes = meta["Source_Scenes"].split(",")
    # Calculate mean values for solar zenith, solar azimuth, and cloud fraction
    total_solar_zenith = 0
    total_solar_azimuth = 0
    total_cloud_fraction = 0
    for scene in source_scenes:
        if scene.startswith("emit"):
            acq_id = scene[:19]
        else:
            acq_id = f"emit{scene.split('_')[4].replace('T', 't')}"
        tmp_acq = WorkflowManager(wm.config_path, acquisition_id=acq_id).acquisition
        total_solar_azimuth += get_band_mean(tmp_acq.obs_img_path, 3)
        total_solar_zenith += get_band_mean(tmp_acq.obs_img_path, 4)
        total_cloud_fraction += tmp_acq.cloud_fraction
    mean_solar_zenith = total_solar_zenith / len(source_scenes)
    mean_solar_azimuth = total_solar_azimuth / len(source_scenes)
    mean_cloud_fraction = total_cloud_fraction / len(source_scenes)

    # Create the UMM-G file
    print(f"Creating ummg file at {local_ummg_path}")
    tif_creation_time = datetime.datetime.fromtimestamp(os.path.getmtime(local_plm_path), tz=datetime.timezone.utc)
    ummg = initialize_ummg(granule_ur, tif_creation_time, collection, ghg_config["collection_version"],
                           acq.start_time, acq.stop_time, ghg_config["repo_name"], ghg_config["repo_version"],
                           sds_software_build_version=sds_software_build_version,
                           ghg_software_build_version=ghg_software_build_version,
                           ghg_software_delivery_version=ghg_config["repo_version"],
                           doi=ghg_config["dois"][collection], orbit=int(acq.orbit), orbit_segment=int(acq.scene),
                           solar_zenith=mean_solar_zenith, solar_azimuth=mean_solar_azimuth,
                           cloud_fraction=mean_cloud_fraction, source_scenes=source_scenes, plume_id=int(plume_id))
    ummg = add_data_files_ummg(ummg, files[:3], acq.daynight, ["TIFF", "JSON", "PNG"])

    with open(local_geojson_path) as f:
        geojson =json.load(f)
        features = geojson["features"]
        found_coords = False
        for f in features:
            if "geometry" in f:
                ummg = add_boundary_ummg(ummg, f["geometry"]["coordinates"][0])
                found_coords = True
        if not found_coords:
            raise RuntimeError(f"Couldn't find coordinates for {fname} in {local_geojson_path}")
    with open(local_ummg_path, 'w', errors='ignore') as fout:
        fout.write(json.dumps(ummg, indent=2, sort_keys=False, cls=SerialEncoder))

    # Copy files to staging server
    print(f"Staging files to AWS S3 server")
    stage_files(wm, acq, files)

    # Build and submit CNM notification
    submit_cnm_notification(wm, acq, base_dir, granule_ur, files, ["data", "data", "browse", "metadata"],
                            collection, ghg_config["collection_version"])


def main():
    # Set up args
    parser = argparse.ArgumentParser(description="Deliver GHG products to LP DAAC")
    parser.add_argument("path", help="The path to the product to be delivered.")
    parser.add_argument("--env", default="ops", help="The operating environment - dev, test, ops")
    parser.add_argument("--collection_version", default="001", help="The DAAC collection version")
    args = parser.parse_args()

    # Get workflow manager and ghg config options
    sds_config_path = f"/store/emit/{args.env}/repos/emit-main/emit_main/config/{args.env}_sds_config.json"

    # Get the current emit-ghg version
    cmd = ["git", "symbolic-ref", "-q", "--short", "HEAD", "||", "git", "describe", "--tags", "--exact-match"]
    output = subprocess.run(" ".join(cmd), shell=True, capture_output=True)
    if output.returncode != 0:
        raise RuntimeError(output.stderr.decode("utf-8"))
    repo_version = output.stdout.decode("utf-8").replace("\n", "")

    # TODO: If collection version and DOI don't go hand in hand then adjust dois below
    ghg_config = {
        "collection_version": args.collection_version,
        "repo_name": "emit-ghg",
        "repo_version": repo_version,
        "dois": {
            "EMITL2BCH4ENH": f"10.5067/EMIT/EMITL2BCH4ENH.{args.collection_version}",
            "EMITL2BCH4PLM": f"10.5067/EMIT/EMITL2BCH4PLM.{args.collection_version}",
            "EMITL2BCO2ENH": f"10.5067/EMIT/EMITL2BCO2ENH.{args.collection_version}",
            "EMITL2BCO2PLM": f"10.5067/EMIT/EMITL2BCO2PLM.{args.collection_version}"
        }
    }

    print(f"Using sds_config_path: {sds_config_path}")
    print(f"Using ghg_config: {ghg_config}")

    # Determine which type of product we have
    base_dir = os.path.dirname(args.path)
    fname = os.path.basename(args.path)
    acq_id = fname[:19]
    print(f"Getting workflow manager with acq_id {acq_id}")
    wm = WorkflowManager(config_path=sds_config_path, acquisition_id=acq_id)

    if "Plume" in fname:
        deliver_plm(base_dir, fname, wm, ghg_config)


if __name__ == '__main__':
    main()
