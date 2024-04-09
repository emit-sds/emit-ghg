
import requests
import math
import time
import argparse
import numpy as np

CMR_OPS = 'https://cmr.earthdata.nasa.gov/search' # CMR API Endpoint
url = f'{CMR_OPS}/{"granules"}'
collections = 'C2748088093-LPCLOUD' # L2BCH4PLM
datetime_range = '2022-08-01T00:00:00Z,2024-12-31T08:00:00Z' # Overall date range of granules to be searched
page_size=2000

# open arguments
parser = argparse.ArgumentParser(description='Get a list of granules from a CMR collection.')
parser.add_argument('token', type=str, help='CMR token')
parser.add_argument('outfile', type=str, help='output file')
args = parser.parse_args()

print(args.token)

timelist = [] 
for n in range(1,3):
    print(url)
    response = requests.get(url, 
                            params={'concept_id': collections, 
                                    'temporal': datetime_range, 
                                    'page_size': page_size, 
                                    'page_num': n,
                                    },
                            headers={
                                'Accept': 'application/json',
                                'Authorization':f'Bearer {args.token}'
                                }
                           )
    print(response.status_code)
    granules = response.json()['feed']['entry']
    
    for g in granules:
        timelist.extend([(g['title'],g['updated'])])
# Show the count of how many granules meet the above parameters
print(f"Number of granules found: {response.headers['CMR-Hits']}") # Resulting quantity of granules/items.




updated = [x[1] for x in timelist]
order = np.argsort(updated)

timelist = [timelist[i] for i in order[::-1]]

out_fidlist = [x[0] for x in timelist]

with open(args.outfile,'w') as f:
    for item in out_fidlist:
        f.write("%s\n" % item)

