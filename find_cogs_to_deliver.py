import argparse
import glob
import json
import os
import requests


def main():
    # Set up args
    parser = argparse.ArgumentParser(description="Deliver GHG products to LP DAAC")
    # parser.add_argument("ghg", help="Must specify either 'ch4' or 'co2'")
    parser.add_argument("base_dir", help="Base directory - use this to determine which GHG")
    parser.add_argument("--cmr", action="store_true", help="Use CMR query to identify already published granules")
    parser.add_argument("--token_file", help="Path to text file containing CMR token")
    args = parser.parse_args()

    # Default base_dirs in the past have been
    # - /scratch/brodrick/methane/visions_delivery_20240409
    # - /scratch/brodrick/methane/visions_delivery_co2

    base_dir = args.base_dir
    ghg = "ch4"
    if "co2" in base_dir:
        ghg = "co2"

    if not args.cmr:
        # If not querying CMR, then check the filesystem for cmr.json files
        cogs = glob.glob(os.path.join(base_dir, "20*/*/*tif"))
        cmr_files = glob.glob(os.path.join(base_dir, "20*/*/*cmr.json"))
        basenames = [os.path.basename(f).replace(".cmr.json", "") for f in cmr_files]
        for cog in cogs:
            if os.path.basename(cog).replace(".tif", "") not in basenames:
                print(cog)
    else:
        # If querying CMR, then need compare COGs against CMR records
        CMR_OPS = 'https://cmr.earthdata.nasa.gov/search'  # CMR API Endpoint
        url = f'{CMR_OPS}/{"granules"}'

        # C2748097305-LPCLOUD CH4ENH emit20220810t064957ch4_enh.tif EMIT_L2B_CH4ENH_001_20220810T064957_2222205_033
        # C2748088093-LPCLOUD CH4PLM emit20220820t101039_CH4_PlumeComplex-2715.tif EMIT_L2B_CH4PLM_001_20220820T101039_002715
        # C2872578364-LPCLOUD CO2ENH emit20231225t061316co2_enh.tif EMIT_L2B_CO2ENH_001_20231225T061316_2335904_007
        # C2867824144-LPCLOUD CO2PLM emit20231204t082834_CO2_PlumeComplex-277.tif EMIT_L2B_CO2PLM_001_20231204T082834_000277

        enh_coll = "C2748097305-LPCLOUD" if ghg == "ch4" else "C2872578364-LPCLOUD"
        plm_coll = "C2748088093-LPCLOUD" if ghg == "ch4" else "C2867824144-LPCLOUD"
        datetime_range = '2022-08-01T00:00:00Z,2024-12-31T08:00:00Z'  # Overall date range of granules to be searched
        page_size = 2000

        # Get token from token file
        if args.token_file is not None:
            token_file = args.token_file
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            token_file = os.path.join(script_dir, "cmr_token.txt")
        with open(token_file, "r") as f:
            token = f.read().replace("\n", "")

        # First check the enhancement collection
        # print(f"Checking {ghg} enhancement collection {enh_coll}")
        response = requests.get(url,
                                params={'concept_id': enh_coll,
                                        'temporal': datetime_range,
                                        'page_size': page_size
                                        },
                                headers={
                                    'Accept': 'application/json',
                                    'Authorization': f'Bearer {token}'
                                }
                                )
        # print(response.status_code)
        # print(f"Number of granules found: {response.headers['CMR-Hits']}")  # Resulting quantity of granules/items.
        granules = response.json()['feed']['entry']
        results = [g["title"].split("_")[4] for g in granules]
        enh_cogs = glob.glob(os.path.join(base_dir, "20*/*enh/*tif"))
        # print(f"Number of enh COGs on filesystem: {len(enh_cogs)}")
        for cog in enh_cogs:
            # If cog not in cmr results, then print
            if os.path.basename(cog)[4:19].upper() not in results:
                print(cog)

        # Next check the plume collection
        # print(f"Checking {ghg} plume collection {plm_coll}")
        response = requests.get(url,
                                params={'concept_id': plm_coll,
                                        'temporal': datetime_range,
                                        'page_size': page_size
                                        },
                                headers={
                                    'Accept': 'application/json',
                                    'Authorization': f'Bearer {token}'
                                }
                                )
        # print(response.status_code)
        # print(f"Number of granules found: {response.headers['CMR-Hits']}")  # Resulting quantity of granules/items.
        granules = response.json()['feed']['entry']
        results = [g["title"].split("_")[4] + "_" + g["title"].split("_")[5] for g in granules]
        plm_cogs = glob.glob(os.path.join(base_dir, "20*/*plm/*tif"))
        # print(f"Number of plm COGs on filesystem: {len(plm_cogs)}")
        for cog in plm_cogs:
            # If cog not in cmr results, then print (use timestamp_plumeid to check unique)
            timestamp = os.path.basename(cog)[4:19].upper()
            plume_id = os.path.basename(cog).split("-")[-1].replace(".tif", "").zfill(6)
            unique_plm = f"{timestamp}_{plume_id}"
            if unique_plm not in results:
                print(cog)


if __name__ == '__main__':
    main()

