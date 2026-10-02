#!/usr/bin/env python3
"""Dependency smoke test; no model weights or datasets required.

Run inside the container: python test_container_dependencies.py
Use --skip-gpu for a CPU-only check. The sbatch wrapper requires a GPU by default.
"""

import argparse
import importlib
import importlib.metadata as metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import traceback


def check_requirements():
    from packaging.requirements import Requirement

    path = Path('/opt/requirements_container.txt')
    if not path.exists():
        raise FileNotFoundError(f'Container build manifest missing: {path}')
    problems = []
    count = 0
    for line in path.read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        req = Requirement(line)
        if req.marker and not req.marker.evaluate():
            continue
        count += 1
        try:
            installed = metadata.version(req.name)
        except metadata.PackageNotFoundError:
            problems.append(f'{req.name}: missing')
            continue
        if req.specifier and not req.specifier.contains(installed, prereleases=True):
            problems.append(f'{req.name}: installed {installed}, expected {req.specifier}')
    if problems:
        raise RuntimeError('\n'.join(problems))
    print(f'  {count} installed distributions match {path}')


def check_geospatial():
    import numpy as np
    from osgeo import gdal
    import pyproj
    import rasterio

    gdal.UseExceptions()
    pixels = np.arange(16, dtype=np.float32).reshape(4, 4)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / 'roundtrip.tif')
        ds = gdal.GetDriverByName('GTiff').Create(path, 4, 4, 1, gdal.GDT_Float32)
        ds.SetGeoTransform((0, 1, 0, 4, 0, -1))
        ds.SetProjection(pyproj.CRS.from_epsg(4326).to_wkt())
        ds.GetRasterBand(1).WriteArray(pixels)
        ds = None
        with rasterio.open(path) as src:
            np.testing.assert_array_equal(src.read(1), pixels)
            assert src.crs.to_epsg() == 4326
    transformer = pyproj.Transformer.from_crs(4326, 3857, always_xy=True)
    np.testing.assert_allclose(transformer.transform(0, 0), (0, 0), atol=1e-6)
    print(f'  GDAL {gdal.VersionInfo()}, rasterio GDAL {rasterio.__gdal_version__}')


def check_data_formats():
    import hdf5plugin  # Registers compression filters used by Graha.
    import numpy as np
    import pandas as pd
    import xarray as xr
    from pycocotools import mask

    with tempfile.TemporaryDirectory() as tmp:
        parquet = Path(tmp) / 'sample.parquet'
        frame = pd.DataFrame({'value': [1, 2, 3]})
        frame.to_parquet(parquet, engine='pyarrow')
        pd.testing.assert_frame_equal(frame, pd.read_parquet(parquet))
        netcdf = Path(tmp) / 'sample.nc'
        data = xr.Dataset({'value': ('x', np.arange(3, dtype=np.float32))})
        data.to_netcdf(netcdf, engine='h5netcdf')
        with xr.open_dataset(netcdf, engine='h5netcdf') as loaded:
            xr.testing.assert_equal(data, loaded)
    binary = np.asfortranarray([[0, 1], [1, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(mask.decode(mask.encode(binary)), binary)


def check_torch(device):
    import torch
    from torchvision.ops import nms

    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable: check GPU allocation and apptainer --nv')
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  PyTorch {torch.__version__}, CUDA build {torch.version.cuda}, device {device}')
    model = torch.nn.Conv2d(3, 4, 3).to(device)
    output = model(torch.randn(2, 3, 16, 16, device=device))
    output.square().mean().backward()
    assert model.weight.grad is not None and torch.isfinite(model.weight.grad).all().item()
    boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.]], device=device)
    scores = torch.tensor([0.9, 0.8], device=device)
    assert nms(boxes, scores, 0.5).tolist() == [0]
    if device == 'cuda':
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-gpu', action='store_true', help='Run only CPU checks')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent
    sys.path[:0] = [str(repo), str(repo / 'graha-lunar-fm')]
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    print(f'Python: {sys.version}\nExecutable: {sys.executable}\nArchitecture: {platform.machine()}')
    failures = []

    def run(name, function):
        print(f'\nCHECK {name}', flush=True)
        try:
            function()
        except Exception:
            failures.append(name)
            traceback.print_exc(file=sys.stdout)
            print(f'FAIL {name}', flush=True)
        else:
            print(f'PASS {name}', flush=True)

    run('container requirements', check_requirements)
    run('pip dependency consistency', lambda: subprocess.run(
        [sys.executable, '-m', 'pip', 'check'], check=True))
    # Third-party import roots from the notebook audit (including lazy imports).
    modules = '''IPython PIL affine einops fiona h5py hdf5plugin huggingface_hub
        ipyleaflet ipywidgets lightning matplotlib numpy omegaconf osgeo.gdal
        pandas pyproj rasterio rioxarray scipy shapely skimage terratorch tifffile
        tiler timm tokenizers torch torchgeo torchmetrics torchvision tqdm xarray
        yaml pyarrow h5netcdf pycocotools.mask'''.split()
    modules += [
        'model', 'model.chip_notebook_utils', 'lfm.labeling.craters',
        'lfm.all_models.all_tasks.graha_inference',
        'lfm.all_models.inst_seg.data_cube_inference',
        'lfm.all_models.sem_seg.data_cube_inference',
        'lfm.full_model.inst_seg.instance_graha_components',
        'lfm.full_model.sem_seg.semantic_graha_components',
        'terratorch_integration',
    ]
    for module in modules:
        run(f'import {module}', lambda module=module: importlib.import_module(module))
    run('GDAL/rasterio/PROJ roundtrip', check_geospatial)
    run('Parquet/NetCDF/COCO roundtrips', check_data_formats)
    run('PyTorch CPU backward + torchvision NMS', lambda: check_torch('cpu'))
    if args.skip_gpu:
        print('\nSKIP GPU checks (--skip-gpu)')
    else:
        run('PyTorch CUDA backward + torchvision NMS', lambda: check_torch('cuda'))
    print(f'\nRESULT: {len(failures)} failed checks')
    for name in failures:
        print(f'  FAIL {name}')
    print('Smoke tests only; full notebook workflows and trained models are not exercised.')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
