"""
Delivers L2B CH4 Enhancement Data and Plumes to the DAAC

Author: Winston Olson-Duvall, winston.olson-duvall@jpl.nasa.gov
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess

import spectral.io.envi as envi

from emit_main.workflow.workflow_manager import WorkflowManager


def initialize_ummg(granule_name: str, creation_time: datetime, collection_name: str, collection_version: str,
                    start_time: datetime, stop_time: datetime, pge_name: str, pge_version: str,
                    sds_software_build_version: str = None, ghg_software_build_version: str = None, doi: str = None,
                    orbit: int = None, orbit_segment: int = None, scene: int = None, solar_zenith: float = None,
                    solar_azimuth: float = None, water_vapor: float = None, aod: float = None,
                    mean_fractional_cover: float = None, mean_spectral_abundance: float = None,
                    cloud_fraction: str = None):
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

    ummg = get_required_ummg()
    ummg['MetadataSpecification'] = {'URL': 'https://cdn.earthdata.nasa.gov/umm/granule/v1.6.3', 'Name': 'UMM-G',
                                     'Version': '1.6.3'}

    ummg['Platforms'] = [{'ShortName': 'ISS', 'Instruments': [{'ShortName': 'EMIT Imaging Spectrometer'}]}]
    ummg['GranuleUR'] = granule_name

    ummg['TemporalExtent'] = {
        'RangeDateTime': {
            'BeginningDateTime': start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            'EndingDateTime': stop_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
    }

    # As of 8/16/22 we are expecting related urls to either be in the collection metadata or inserted by the DAAC
    # ummg['RelatedUrls'] = [{'URL': 'https://earth.jpl.nasa.gov/emit/', 'Type': 'PROJECT HOME PAGE', 'Description': 'Link to the EMIT Project Website.'}]
    # ummg = add_related_url(ummg, 'https://github.com/emit-sds/emit-documentation', 'VIEW RELATED INFORMATION',
    #                        description='Link to Algorithm Theoretical Basis Documents', url_subtype='ALGORITHM DOCUMENTATION')
    # ummg = add_related_url(ummg, 'https://github.com/emit-sds/emit-documentation', 'VIEW RELATED INFORMATION',
    #                        description='Link to Data User\'s Guide', url_subtype='USER\'S GUIDE')

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

    # For GPolygon, add the first point again to close out
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
    # Copy files to staging server
    partial_dir_arg = f"--partial-dir={acq.daac_partial_dir}"
    target = f"{wm.config['daac_server_internal']}:{acq.daac_staging_dir}/"
    group = f"emit-{wm.config['environment']}" if wm.config["environment"] in ("test", "ops") else "emit-dev"
    # This command only makes the directory and changes ownership if the directory doesn't exist
    cmd_make_target = ["ssh", wm.config["daac_server_internal"], "\"if", "[", "!", "-d",
                       f"'{acq.daac_staging_dir}'", "];", "then", "mkdir", f"{acq.daac_staging_dir};", "chgrp",
                       group, f"{acq.daac_staging_dir};", "fi\""]
    output = subprocess.run(" ".join(cmd_make_target), shell=True, capture_output=True)
    if output.returncode != 0:
        raise RuntimeError(output.stderr.decode("utf-8"))

    # files is a list of (source, target)
    for f in files:
        cmd_rsync = ["rsync", "-av", partial_dir_arg, f[0], target + f[1]]
        output = subprocess.run(" ".join(cmd_rsync), shell=True, capture_output=True)
        if output.returncode != 0:
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
        notification["files"].append(
            {
                "name": f[1],
                "uri": acq.daac_uri_base + f[1],
                "type": formats[i],
                "size": os.path.getsize(f[0]),
                "checksumType": "sha512",
                "checksum": calc_checksum(f[0], "sha512")
            }
        )

    # Write notification submission to file
    with open(cnm_submission_path, "w") as f:
        f.write(json.dumps(notification, indent=4))
    wm.change_group_ownership(cnm_submission_path)

    # Submit notification via AWS SQS
    cnm_submission_output = cnm_submission_path.replace(".json", ".out")
    cmd_aws = [wm.config["aws_cli_exe"], "sqs", "send-message", "--queue-url", queue_url, "--message-body",
               f"file://{cnm_submission_path}", "--profile", wm.config["aws_profile"], ">", cnm_submission_output]
    output = subprocess.run(" ".join(cmd_aws), shell=True, capture_output=True)
    if output.returncode != 0:
        raise RuntimeError(output.stderr.decode("utf-8"))


def deliver_ch4enh(base_dir, fname, wm, ghg_config):
    # Get scene details and create Granule UR
    # Example granule_ur: EMIT_L2B_CH4ENH_001_20230805T195459_2321713_001
    acq = wm.acquistion
    collection = "EMITL2BCH4ENH"
    prod_type = "CH4ENH"
    granule_ur = f"EMIT_L2B_{prod_type}_{ghg_config['delivery_version']}_{acq.start_time.strftime('%Y%m%dT%H%M%S')}_{acq.orbit}_{acq.daac_scene}"

    # Create local/tmp daac names and paths
    local_enh_path = os.path.join(base_dir, fname)
    local_browse_path = local_enh_path.replace(".tif", ".png")
    local_ummg_path = local_enh_path.replace(".tif", ".cmr.json")
    daac_enh_name = f"{granule_ur}.tif"
    daac_browse_name = f"{granule_ur}.png"
    daac_ummg_name = f"{granule_ur}.cmr.json"
    daac_ummg_path = os.path.join(base_dir, daac_ummg_name)
    files = [(local_enh_path, daac_enh_name),
             (local_browse_path, daac_browse_name),
             local_ummg_path, daac_ummg_name]

    # Get the software_build_version (extended build num when product was created)
    hdr = envi.read_envi_header(acq.rdn_hdr_path)
    sds_software_build_version = hdr["emit software build version"]

    # TODO: Get ghg_software_build_version
    ghg_software_build_version = ""

    # Create the UMM-G file
    tif_creation_time = datetime.datetime.fromtimestamp(os.path.getmtime(local_enh_path), tz=datetime.timezone.utc)
    daynight = "Day" if acq.submode == "science" else "Night"
    ummg = initialize_ummg(acq.abun_granule_ur, tif_creation_time, collection, ghg_config["collection_version"],
                           acq.start_time, acq.stop_time, ghg_config["repo_name"], ghg_config["repo_version"],
                           sds_software_build_version=sds_software_build_version,
                           ghg_software_build_version=ghg_software_build_version,
                           doi=wm.config["dois"]["EMITL2BCH4ENH"], orbit=int(acq.orbit), orbit_segment=int(acq.scene),
                           scene=int(acq.daac_scene), solar_zenith=acq.mean_solar_zenith,
                           solar_azimuth=acq.mean_solar_azimuth, cloud_fraction=acq.cloud_fraction)
    ummg = add_data_files_ummg(ummg, files[:2], daynight, ["COG", "PNG"])

    ummg = add_boundary_ummg(ummg, acq.gring)
    with open(daac_ummg_path, 'w', errors='ignore') as fout:
        fout.write(json.dumps(ummg, indent=2, sort_keys=False, cls=SerialEncoder))

    # Copy files to staging server
    stage_files(wm, acq, files)

    # Build and submit CNM notification
    submit_cnm_notification(wm, acq, base_dir, granule_ur, files, ["data", "browse", "metadata"], collection,
                            ghg_config["collection_version"])





def main():
    # Set up args
    parser = argparse.ArgumentParser(description="Deliver GHG products to LP DAAC")
    parser.add_argument("path", help="The path to the product to be delivered.")
    parser.add_argument("--env", default="ops", help="The operating environment - dev, test, ops")
    args = parser.parse_args()

    # Get workflow manager and ghg config options
    sds_config_path = f"/store/emit/{args.env}/repos/emit-main/emit_main/config/{args.env}_sds_config.json"
    ghg_config = {
        "collection_version": "001",
        "repo_name": "emit-ghg",
        "repo_version": "v0.0.0",
        "dois": {
            "EMITL2BCH4ENH": "",
            "EMITL2BCH4PLM": ""
        }
    }

    # Determine which type of product we have
    base_dir = os.path.dirname(args.path)
    fname = os.path.basename(args.path)
    acq_id = fname.split("_")[0]
    wm = WorkflowManager(config_path=sds_config_path, acquisition_id = acq_id)

    # TODO: Get filenames
    if fname.endswith(""):
        deliver_ch4enh(base_dir, fname, wm, ghg_config)
    elif fname.endswith(""):
        pass
        # deliver_ch4plm(base_dir, fname, wm, ghg_config)


if __name__ == '__main__':
    main()
