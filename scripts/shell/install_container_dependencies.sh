#!/usr/bin/env bash
# Shared by the Docker and Apptainer container builds.
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive
export PYTHONNOUSERSITE=1
export GDAL_CONFIG=/usr/bin/gdal-config
export PROJ_DATA=/usr/share/proj
export PROJ_LIB=/usr/share/proj
export GDAL_DATA=/usr/share/gdal

echo "=== Installing system GDAL / PROJ ==="

apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    pkg-config \
    gdal-bin \
    libgdal-dev \
    proj-bin \
    proj-data \
    libproj-dev \
    libgeos-dev \
    ca-certificates

gdalinfo --version
gdal-config --version

echo "=== Installing build-time Python dependencies ==="
python -m pip install \
    -c /etc/pip/constraint.txt \
    numpy==2.2.6 \
    'Cython>=3.0' \
    packaging \
    pytest \
    setuptools \
    versioneer \
    wheel

echo "=== Building GDAL Python bindings against active NumPy ==="

python -m pip install \
    --no-build-isolation \
    --no-binary=GDAL \
    --ignore-installed \
    --no-deps \
    "GDAL==$(gdal-config --version)"

echo "=== Building Python geospatial wrappers against system GDAL / PROJ ==="

python -m pip install \
    -c /etc/pip/constraint.txt \
    --no-build-isolation \
    --no-binary=rasterio,pyproj,pyogrio \
    rasterio==1.5.0 \
    pyproj==3.7.2 \
    pyogrio==0.13.0

echo "=== Protecting container-provided builds ==="

# Local wheel/source paths in old freezes do not exist in a new build.
# Require those distributions from the base image and protect their versions,
# together with the native geospatial stack built above, during resolution.
python - <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

requirements = Path('/opt/requirements_container.txt').read_text()
managed = requirements.split('# BEGIN CONTAINER-MANAGED PACKAGES\n', 1)[1]
managed = managed.split('# END CONTAINER-MANAGED PACKAGES', 1)[0]
names = {line.strip() for line in managed.splitlines()
         if line.strip() and not line.lstrip().startswith('#')}
names.update({'numpy', 'rasterio', 'pyproj', 'pyogrio'})
constraints = []
for name in sorted(names):
    try:
        installed = version(name)
    except PackageNotFoundError:
        raise SystemExit(f'Required container-provided package is missing: {name}')
    constraints.append(f'{name}=={installed}')
Path('/opt/lfm-native-constraints.txt').write_text('\n'.join(constraints) + '\n')
PY

echo "=== Installing combined project dependencies ==="

python -m pip install \
    -c /etc/pip/constraint.txt \
    -c /opt/lfm-native-constraints.txt \
    --no-binary=rasterio,pyproj,pyogrio \
    -r /opt/requirements_container.txt

python -m pip check

echo "=== Checking native Python bindings without a GPU ==="
python - <<'PY'
import numpy as np
import pyogrio
import pyproj
import rasterio
import torch
from osgeo import gdal
from torchvision.ops import nms

gdal.UseExceptions()
pixels = np.arange(4, dtype=np.uint8).reshape(2, 2)
dataset = gdal.GetDriverByName('MEM').Create('', 2, 2, 1, gdal.GDT_Byte)
dataset.GetRasterBand(1).WriteArray(pixels)
np.testing.assert_array_equal(dataset.ReadAsArray(), pixels)
assert pyproj.CRS.from_epsg(4326).to_epsg() == 4326
boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.]])
assert nms(boxes, torch.tensor([0.9, 0.8]), 0.5).tolist() == [0]
print('GDAL:', gdal.VersionInfo(), 'rasterio GDAL:', rasterio.__gdal_version__)
print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda)
PY

python -m pip freeze > /opt/lfm-installed-requirements.txt

rm -rf /var/lib/apt/lists/*
rm -rf /var/cache/apt/archives/*
# Retain requirements and native constraints in /opt for build provenance.
