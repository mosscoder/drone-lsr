#!/usr/bin/env python3
import os
import tempfile
import logging
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import rasterio
from rasterio import windows
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject, calculate_default_transform
from tqdm import tqdm
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TIMES = [1000, 1200, 1500]
ORTHO_URL = 'https://storage.googleapis.com/mpg-aerial-survey/surveys/241107_lowerpartridge_{TIME}/processing/dronedeploy/241107_lowerpartridge_{TIME}-visible.tif'

RGB_OUTPUT_DIR = Path('data/raster/tiles/rgb')
CANOPY_RASTER = Path('data/raster/dems/canopy_height.tif')
CANOPY_OUTPUT_DIR = Path('data/raster/tiles/chm')
TILE_SIZE = 1024
N_JOBS = max(1, os.cpu_count() - 1)


def get_orthomosaic_info(time):
    ortho_url = ORTHO_URL.format(TIME=time)
    with rasterio.open(ortho_url) as src:
        return {
            'time': time,
            'url': ortho_url,
            'bounds': src.bounds,
            'crs': src.crs,
            'resolution': (src.res[0], src.res[1]),
            'width': src.width,
            'height': src.height,
        }


def find_reference_orthomosaic():
    logging.info('Loading orthomosaic information...')
    infos = []
    for time in TIMES:
        info = get_orthomosaic_info(time)
        infos.append(info)
        logging.info('Time %s: Resolution = %s', time, info['resolution'])

    reference = min(infos, key=lambda x: x['resolution'][0])
    logging.info('Selected reference orthomosaic: Time %s with resolution %s', reference['time'], reference['resolution'])
    return reference, infos


def create_tile_grid(reference):
    bounds = reference['bounds']
    res_x, res_y = reference['resolution']

    tile_width = TILE_SIZE * res_x
    tile_height = TILE_SIZE * res_y

    width = bounds.right - bounds.left
    height = bounds.top - bounds.bottom

    n_cols = int(np.ceil(width / tile_width))
    n_rows = int(np.ceil(height / tile_height))

    logging.info('Creating tile grid: %d rows x %d columns', n_rows, n_cols)

    tiles = []
    for row in range(n_rows):
        for col in range(n_cols):
            left = bounds.left + col * tile_width
            right = min(left + tile_width, bounds.right)
            top = bounds.top - row * tile_height
            bottom = max(top - tile_height, bounds.bottom)

            tiles.append({
                'row': row,
                'col': col,
                'bounds': (left, bottom, right, top),
                'crs': reference['crs'],
            })
    return tiles


def check_tile_has_data(src, window):
    alpha = src.read(4, window=window)
    return np.all(alpha > 0)


def process_rgb_tile(args):
    tile, ortho_infos = args
    row, col = tile['row'], tile['col']
    tile_bounds = tile['bounds']
    location_id = f"{row}_{col}"

    has_data = {}
    for ortho in ortho_infos:
        with rasterio.open(ortho['url']) as src:
            window = windows.from_bounds(*tile_bounds, transform=src.transform)
            has_data[ortho['time']] = check_tile_has_data(src, window)

    if not all(has_data.values()):
        missing = [t for t, ok in has_data.items() if not ok]
        logging.debug('Skipping tile %s_%s - missing data at %s', row, col, missing)
        return False, location_id

    for ortho in ortho_infos:
        output_path = RGB_OUTPUT_DIR / f'{row}_{col}_{ortho["time"]}.png'
        if output_path.exists():
            continue
        with rasterio.open(ortho['url']) as src:
            window = windows.from_bounds(*tile_bounds, transform=src.transform)
            data = src.read(window=window)
            transform = from_bounds(*tile_bounds, TILE_SIZE, TILE_SIZE)

            if data.shape[1] != TILE_SIZE or data.shape[2] != TILE_SIZE:
                resampled = np.zeros((4, TILE_SIZE, TILE_SIZE), dtype=data.dtype)
                for band_idx in range(4):
                    reproject(
                        source=data[band_idx],
                        destination=resampled[band_idx],
                        src_transform=src.window_transform(window),
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=src.crs,
                        resampling=Resampling.bilinear,
                    )
                data = resampled

            data_pil = np.transpose(data, (1, 2, 0))
            if data_pil.dtype != np.uint8:
                data_pil = data_pil.astype(np.uint8)
            Image.fromarray(data_pil, mode='RGBA').save(output_path, 'PNG')

    return True, location_id


def tile_rgb(reference, ortho_infos, tiles):
    RGB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info('Tiling RGB orthomosaics with %d workers', N_JOBS)
    successes = set()
    skipped = set()
    with Pool(N_JOBS) as pool:
        with tqdm(total=len(tiles), desc='Processing RGB tiles') as pbar:
            for success, location_id in pool.imap(process_rgb_tile, [(tile, ortho_infos) for tile in tiles]):
                if success:
                    successes.add(location_id)
                else:
                    skipped.add(location_id)
                pbar.update(1)

    logging.info('RGB tiling complete: %d success, %d skipped', len(successes), len(skipped))
    return successes


def reproject_canopy_to_tmp(reference, tmp_path: Path):
    """Reproject the canopy raster to the orthomosaic CRS into a temporary GeoTIFF."""
    target_crs = reference['crs']
    with rasterio.open(CANOPY_RASTER) as src:
        if src.crs == target_crs:
            logging.info('Canopy raster already in target CRS %s', target_crs)
            nodata = src.nodata if src.nodata is not None else np.nan
            meta = src.meta.copy()
            meta.update({'driver': 'GTiff', 'dtype': 'float32', 'nodata': nodata})
            data = src.read(1).astype(np.float32)
            with rasterio.open(tmp_path, 'w', **meta) as dst:
                dst.write(data, 1)
            return

        transform, width, height = calculate_default_transform(
            src.crs,
            target_crs,
            src.width,
            src.height,
            *src.bounds,
        )

        nodata = src.nodata if src.nodata is not None else np.nan
        kwargs = src.meta.copy()
        kwargs.update({
            'driver': 'GTiff',
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'dtype': 'float32',
            'count': 1,
            'nodata': nodata,
        })

        destination = np.empty((height, width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
            src_nodata=src.nodata,
            dst_nodata=nodata,
        )

        with rasterio.open(tmp_path, 'w', **kwargs) as dst:
            dst.write(destination, 1)


def tile_canopy(reference, tiles, valid_locations):
    if not CANOPY_RASTER.exists():
        raise FileNotFoundError(f'Canopy raster missing at {CANOPY_RASTER}')

    CANOPY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info('Tiling canopy height model to %s', CANOPY_OUTPUT_DIR)

    fd, tmp_name = tempfile.mkstemp(suffix='.tif')
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        reproject_canopy_to_tmp(reference, tmp_path)

        # Remove stale canopy tiles that no longer have corresponding RGB tiles
        existing_tiles = list(CANOPY_OUTPUT_DIR.glob('*.tif'))
        for existing in existing_tiles:
            if existing.stem not in valid_locations:
                logging.debug('Removing stale canopy tile %s', existing.name)
                existing.unlink()

        with rasterio.open(tmp_path) as src:
            if reference['crs'] and src.crs and reference['crs'] != src.crs:
                logging.warning('CRS mismatch between orthomosaic (%s) and canopy (%s)', reference['crs'], src.crs)

            for tile in tqdm(tiles, desc='Processing canopy tiles'):
                row = tile['row']
                col = tile['col']
                location_id = f"{row}_{col}"
                if location_id not in valid_locations:
                    continue
                output_path = CANOPY_OUTPUT_DIR / f'{row}_{col}.tif'
                if output_path.exists():
                    continue

                bounds = tile['bounds']
                transform = from_bounds(*bounds, TILE_SIZE, TILE_SIZE)
                window = windows.from_bounds(*bounds, transform=src.transform)
                data = src.read(
                    1,
                    window=window,
                    out_shape=(TILE_SIZE, TILE_SIZE),
                    resampling=Resampling.bilinear,
                    boundless=True,
                    fill_value=np.nan,
                )

                meta = {
                    'driver': 'GTiff',
                    'height': TILE_SIZE,
                    'width': TILE_SIZE,
                    'count': 1,
                    'dtype': 'float32',
                    'crs': reference['crs'],
                    'transform': transform,
                    'nodata': np.nan,
                }

                with rasterio.open(output_path, 'w', **meta) as dst:
                    dst.write(data.astype(np.float32), 1)

    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass

    logging.info('Canopy tiling complete')


def main():
    reference, ortho_infos = find_reference_orthomosaic()
    tiles = create_tile_grid(reference)
    logging.info('Generated %d tiles', len(tiles))

    valid_locations = tile_rgb(reference, ortho_infos, tiles)
    tile_canopy(reference, tiles, valid_locations)


if __name__ == '__main__':
    main()
