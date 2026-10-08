# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/pytorch:26.06-py3

ENV PYTHONNOUSERSITE=1 \
    GDAL_CONFIG=/usr/bin/gdal-config \
    PROJ_DATA=/usr/share/proj \
    PROJ_LIB=/usr/share/proj \
    GDAL_DATA=/usr/share/gdal

COPY requirements_container.txt /opt/requirements_container.txt
COPY scripts/shell/install_container_dependencies.sh /opt/install_container_dependencies.sh
RUN bash /opt/install_container_dependencies.sh

# Repository code and datasets are mounted at runtime, as with the .def image.
