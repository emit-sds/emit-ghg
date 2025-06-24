import argparse
import uuid
from osgeo import gdal

def create_mosaic(input_files, output_file):
    vrt_options = gdal.BuildVRTOptions(resolution='average')
    vrt_ds = gdal.BuildVRT(f'/vsimem/{uuid.uuid4().hex}.vrt', input_files, options=vrt_options)

    translate_options = gdal.TranslateOptions(format="COG",
                                              creationOptions=["TILING_SCHEME=GoogleMapsCompatible"])
    gdal.Translate(output_file, vrt_ds, options=translate_options)

def main():
    parser = argparse.ArgumentParser(description="Export Cloud Optimized GeoTIFF mosaic")
    parser.add_argument('input', nargs='+', help='Input files')
    parser.add_argument('output', help='Output file')
    args = parser.parse_args()

    create_mosaic(args.input, args.output)

if __name__ == "__main__":
    main()