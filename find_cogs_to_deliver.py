import glob
import os
import sys

if len(sys.argv) == 1:
    print("Must specify either 'ch4' or 'co2' as an argument")
    sys.exit(1)
ghg =  sys.argv[1]

if ghg == "ch4":
    cogs = glob.glob(f"/scratch/brodrick/methane/visions_delivery_20240409/20*/*/*tif")
    cmr_files = glob.glob(f"/scratch/brodrick/methane/visions_delivery_20240409/20*/*/*cmr.json")
if ghg == "co2":
    cogs = glob.glob(f"/scratch/brodrick/methane/visions_delivery_co2/20*/*/*tif")
    cmr_files = glob.glob(f"/scratch/brodrick/methane/visions_delivery_co2/20*/*/*cmr.json")

basenames = [os.path.basename(f).replace(".cmr.json", "") for f in cmr_files]

new_cogs = []
for cog in cogs:
    if os.path.basename(cog).replace(".tif", "") not in basenames:
        print(cog)
