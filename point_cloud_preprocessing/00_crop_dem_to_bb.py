#!/usr/bin/env python3
"""
Crop DEM to match the shared bounding box of the LAS point clouds and convert to ellipsoidal heights.

This script:
1. Crops a DEM raster to the same spatial extent as the LAS point clouds
2. Converts orthometric heights (NAVD88) to ellipsoidal heights (NAD83 2011)
   using the GEOID12B model, ensuring vertical datum compatibility with LAS data

Outputs:
- qspatial_2019_orthometric.tif: Cropped DEM with original orthometric heights
- qspatial_2019_ellipsoidal.tif: Final DEM with ellipsoidal heights matching LAS data
"""

import json
import subprocess
from pathlib import Path
from osgeo import gdal, osr
import numpy as np
import tempfile

def get_las_bounds(las_file):
    """Get bounding box of LAS file using PDAL info."""
    try:
        result = subprocess.run(
            ['pdal', 'info', '--metadata', str(las_file)],
            capture_output=True,
            text=True,
            check=True
        )
        metadata = json.loads(result.stdout)

        bounds = {
            'minx': metadata['metadata']['minx'],
            'maxx': metadata['metadata']['maxx'],
            'miny': metadata['metadata']['miny'],
            'maxy': metadata['metadata']['maxy'],
            'minz': metadata['metadata']['minz'],
            'maxz': metadata['metadata']['maxz']
        }

        # Also get the SRS info
        srs_wkt = metadata['metadata']['spatialreference']

        return bounds, srs_wkt
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to get bounds for {las_file}: {e}")

def calculate_shared_bounds(las_files):
    """Calculate the smallest shared bounding box from multiple LAS files."""
    bounds_list = []
    srs_wkt = None

    for las_file in las_files:
        bounds, wkt = get_las_bounds(las_file)
        bounds_list.append(bounds)
        if srs_wkt is None:
            srs_wkt = wkt

    shared_bounds = {
        'minx': max(b['minx'] for b in bounds_list),
        'maxx': min(b['maxx'] for b in bounds_list),
        'miny': max(b['miny'] for b in bounds_list),
        'maxy': min(b['maxy'] for b in bounds_list),
    }

    # Validate that we have a valid bounding box
    if (shared_bounds['minx'] >= shared_bounds['maxx'] or
        shared_bounds['miny'] >= shared_bounds['maxy']):
        raise ValueError("No spatial overlap between input files")

    return shared_bounds, srs_wkt

def crop_dem_to_bounds(input_dem, output_dem, bounds, target_srs_wkt=None):
    """
    Crop DEM to specified bounds using GDAL.

    Args:
        input_dem: Path to input DEM file
        output_dem: Path to output cropped DEM
        bounds: Dictionary with minx, maxx, miny, maxy
        target_srs_wkt: Target SRS in WKT format (optional, for reprojection)
    """
    # Open the input DEM
    src_ds = gdal.Open(str(input_dem))
    if src_ds is None:
        raise RuntimeError(f"Failed to open {input_dem}")

    # Get source SRS
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(src_ds.GetProjection())

    # Set up target SRS if provided
    if target_srs_wkt:
        target_srs = osr.SpatialReference()
        target_srs.ImportFromWkt(target_srs_wkt)

        # Check if reprojection is needed
        if not src_srs.IsSame(target_srs):
            print("  Note: DEM and LAS use different coordinate systems")
            print(f"    DEM SRS: {src_srs.GetAttrValue('PROJCS') or src_srs.GetAttrValue('GEOGCS')}")
            print(f"    LAS SRS: {target_srs.GetAttrValue('PROJCS') or target_srs.GetAttrValue('GEOGCS')}")

    # Use GDAL Warp to crop (and optionally reproject)
    warp_options = gdal.WarpOptions(
        outputBounds=[bounds['minx'], bounds['miny'], bounds['maxx'], bounds['maxy']],
        outputBoundsSRS=target_srs_wkt if target_srs_wkt else None,
        dstSRS=target_srs_wkt if target_srs_wkt else None,
        resampleAlg='bilinear',
        format='GTiff',
        creationOptions=['COMPRESS=LZW', 'TILED=YES']
    )

    # Perform the warp operation
    gdal.Warp(str(output_dem), src_ds, options=warp_options)

    # Close dataset
    src_ds = None

    # Verify output
    result_ds = gdal.Open(str(output_dem))
    if result_ds is None:
        raise RuntimeError(f"Failed to create {output_dem}")

    # Get output info
    geotransform = result_ds.GetGeoTransform()
    width = result_ds.RasterXSize
    height = result_ds.RasterYSize

    # Calculate actual bounds of output
    actual_minx = geotransform[0]
    actual_maxy = geotransform[3]
    actual_maxx = actual_minx + width * geotransform[1]
    actual_miny = actual_maxy + height * geotransform[5]

    # Get elevation statistics
    band = result_ds.GetRasterBand(1)
    stats = band.GetStatistics(True, True)
    min_elev, max_elev, mean_elev, std_elev = stats

    result_ds = None

    return {
        'width': width,
        'height': height,
        'pixel_size': abs(geotransform[1]),
        'bounds': {
            'minx': actual_minx,
            'maxx': actual_maxx,
            'miny': actual_miny,
            'maxy': actual_maxy
        },
        'elevation': {
            'min': min_elev,
            'max': max_elev,
            'mean': mean_elev,
            'std': std_elev
        }
    }

def convert_to_ellipsoidal_height(input_dem, output_dem, geoid_file):
    """
    Convert orthometric heights to ellipsoidal heights using GDAL warp + raster math.

    This approach:
    1. Warps the geoid to match the DEM's projection and resolution
    2. Adds the geoid separation to the orthometric heights: ellipsoidal = orthometric + geoid

    Args:
        input_dem: Path to input DEM with orthometric heights
        output_dem: Path to output DEM with ellipsoidal heights
        geoid_file: Path to geoid model file (GEOID12B binary format)
    """
    print(f"    Warping geoid to match DEM projection and resolution...")

    # Open the DEM to get its spatial reference and geotransform
    dem_ds = gdal.Open(str(input_dem))
    if dem_ds is None:
        raise RuntimeError(f"Failed to open {input_dem}")

    dem_proj = dem_ds.GetProjection()
    dem_geotransform = dem_ds.GetGeoTransform()
    dem_width = dem_ds.RasterXSize
    dem_height = dem_ds.RasterYSize

    # Calculate DEM bounds
    minx = dem_geotransform[0]
    maxy = dem_geotransform[3]
    maxx = minx + dem_width * dem_geotransform[1]
    miny = maxy + dem_height * dem_geotransform[5]

    # Create temporary warped geoid file
    temp_geoid = tempfile.NamedTemporaryFile(suffix='_warped_geoid.tif', delete=False)
    temp_geoid_path = temp_geoid.name
    temp_geoid.close()

    try:
        # Warp geoid to match DEM
        warp_options = gdal.WarpOptions(
            format='GTiff',
            outputBounds=[minx, miny, maxx, maxy],
            width=dem_width,
            height=dem_height,
            dstSRS=dem_proj,
            resampleAlg='bilinear',
            creationOptions=['COMPRESS=LZW', 'TILED=YES']
        )

        print(f"    Warping {geoid_file} to match DEM...")
        gdal.Warp(temp_geoid_path, str(geoid_file), options=warp_options)

        # Verify warped geoid was created
        geoid_ds = gdal.Open(temp_geoid_path)
        if geoid_ds is None:
            raise RuntimeError(f"Failed to warp geoid file")

        print(f"    Adding geoid separation to orthometric heights...")

        # Read the DEM data
        dem_band = dem_ds.GetRasterBand(1)
        dem_data = dem_band.ReadAsArray()
        dem_nodata = dem_band.GetNoDataValue()

        # Read the warped geoid data
        geoid_band = geoid_ds.GetRasterBand(1)
        geoid_data = geoid_band.ReadAsArray()

        # Ensure arrays have the same shape
        if dem_data.shape != geoid_data.shape:
            raise RuntimeError(f"DEM and geoid array shapes don't match: {dem_data.shape} vs {geoid_data.shape}")

        # Apply conversion: ellipsoidal = orthometric + geoid
        ellipsoidal_data = dem_data.copy()

        if dem_nodata is not None:
            # Only apply to valid pixels
            valid_mask = dem_data != dem_nodata
            ellipsoidal_data[valid_mask] = dem_data[valid_mask] + geoid_data[valid_mask]
        else:
            # Apply to all pixels
            ellipsoidal_data = dem_data + geoid_data

        # Create output DEM
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(
            str(output_dem),
            dem_width,
            dem_height,
            1,
            gdal.GDT_Float32,
            ['COMPRESS=LZW', 'TILED=YES']
        )

        # Set spatial reference and geotransform
        out_ds.SetProjection(dem_proj)
        out_ds.SetGeoTransform(dem_geotransform)

        # Write the ellipsoidal data
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(ellipsoidal_data)
        if dem_nodata is not None:
            out_band.SetNoDataValue(dem_nodata)

        # Get statistics
        stats = out_band.GetStatistics(True, True)
        min_elev, max_elev, mean_elev, std_elev = stats

        # Calculate average geoid separation (for reporting)
        if dem_nodata is not None:
            valid_geoid = geoid_data[valid_mask]
            avg_geoid_sep = np.mean(valid_geoid) if len(valid_geoid) > 0 else 0
        else:
            avg_geoid_sep = np.mean(geoid_data)

        # Close datasets
        dem_ds = None
        geoid_ds = None
        out_ds = None

        return {
            'elevation': {
                'min': min_elev,
                'max': max_elev,
                'mean': mean_elev,
                'std': std_elev
            },
            'avg_geoid_separation': avg_geoid_sep,
            'method': 'gdal_warp_plus_raster_math'
        }

    finally:
        # Clean up temporary file
        try:
            Path(temp_geoid_path).unlink()
        except:
            pass  # Ignore cleanup errors

def main():
    # Input paths
    input_dem = Path('/Users/kdoherty/bitterroot_canopy/data/raster/lidar_raw/dems/2019/46114f1_2019_qspatial_dem.tif')
    geoid_file = Path('data/raster/dems/g2012bu1.bin')
    las_dir = Path('data/point_cloud')

    # LAS files to get bounds from (three timepoints)
    las_files = [
        las_dir / '1000.las',
        las_dir / '1200.las',
        las_dir / '1500.las'
    ]

    # Output paths
    output_dir = Path('data/raster/dems')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Intermediate file (orthometric heights, cropped)
    orthometric_dem = output_dir / 'qspatial_2019_orthometric.tif'
    # Final file (ellipsoidal heights)
    ellipsoidal_dem = output_dir / 'qspatial_2019_ellipsoidal.tif'

    # Validate input files exist
    if not input_dem.exists():
        raise FileNotFoundError(f"Input DEM not found: {input_dem}")

    if not geoid_file.exists():
        raise FileNotFoundError(f"Geoid file not found: {geoid_file}")

    for las_file in las_files:
        if not las_file.exists():
            raise FileNotFoundError(f"LAS file not found: {las_file}")

    print("Analyzing three timepoint LAS files for shared bounding box...")

    # Get shared bounds from LAS files
    shared_bounds, las_srs_wkt = calculate_shared_bounds(las_files)

    print(f"\nShared bounding box across timepoints:")
    print(f"  X: {shared_bounds['minx']:.3f} → {shared_bounds['maxx']:.3f}")
    print(f"  Y: {shared_bounds['miny']:.3f} → {shared_bounds['maxy']:.3f}")
    print(f"  Width: {shared_bounds['maxx'] - shared_bounds['minx']:.3f} m")
    print(f"  Height: {shared_bounds['maxy'] - shared_bounds['miny']:.3f} m")

    print(f"\nStep 1: Cropping DEM to LAS bounds...")
    print(f"  Input: {input_dem}")
    print(f"  Output: {orthometric_dem}")

    # Crop the DEM (orthometric heights)
    crop_result = crop_dem_to_bounds(input_dem, orthometric_dem, shared_bounds, las_srs_wkt)

    print(f"\nCrop complete!")
    print(f"  Output dimensions: {crop_result['width']} x {crop_result['height']} pixels")
    print(f"  Pixel size: {crop_result['pixel_size']:.3f} m")
    print(f"  Final bounds:")
    print(f"    X: {crop_result['bounds']['minx']:.3f} → {crop_result['bounds']['maxx']:.3f}")
    print(f"    Y: {crop_result['bounds']['miny']:.3f} → {crop_result['bounds']['maxy']:.3f}")
    print(f"  Orthometric elevation range:")
    print(f"    Min: {crop_result['elevation']['min']:.2f} m")
    print(f"    Max: {crop_result['elevation']['max']:.2f} m")
    print(f"    Mean: {crop_result['elevation']['mean']:.2f} m (±{crop_result['elevation']['std']:.2f})")

    print(f"\nStep 2: Converting to ellipsoidal heights...")
    print(f"  Input: {orthometric_dem}")
    print(f"  Geoid: {geoid_file}")
    print(f"  Output: {ellipsoidal_dem}")

    # Convert orthometric heights to ellipsoidal heights
    conversion_result = convert_to_ellipsoidal_height(orthometric_dem, ellipsoidal_dem, geoid_file)

    print(f"\nConversion complete!")
    print(f"  Ellipsoidal elevation range:")
    print(f"    Min: {conversion_result['elevation']['min']:.2f} m")
    print(f"    Max: {conversion_result['elevation']['max']:.2f} m")
    print(f"    Mean: {conversion_result['elevation']['mean']:.2f} m (±{conversion_result['elevation']['std']:.2f})")

    # Show conversion method used
    if 'method' in conversion_result:
        print(f"  Conversion method: {conversion_result['method']}")

    # Show geoid separation statistics
    if 'avg_geoid_separation' in conversion_result:
        print(f"  Average geoid separation: {conversion_result['avg_geoid_separation']:.3f} m")
    else:
        # Fallback calculation
        ortho_mean = crop_result['elevation']['mean']
        ellip_mean = conversion_result['elevation']['mean']
        geoid_sep = ellip_mean - ortho_mean
        print(f"  Average geoid separation: {geoid_sep:.3f} m")

if __name__ == "__main__":
    main()