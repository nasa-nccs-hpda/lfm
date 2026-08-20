
from pathlib import Path

from joblib import cpu_count
import numpy
from sklearn import cluster

from osgeo import gdal


# ----------------------------------------------------------------------------
# Class Clusterer
# ----------------------------------------------------------------------------
class Clusterer(object):

    # ------------------------------------------------------------------------
    # getClusters
    # ------------------------------------------------------------------------
    @staticmethod
    def getClusters(bands: list, numClusters: int=5) -> numpy.ndarray:

        img = numpy.moveaxis(bands, 0, -1)
        img1d = img.reshape(-1, img.shape[-1])

        cl = cluster.MiniBatchKMeans(n_clusters=numClusters,
                                     random_state=0,
                                     batch_size=256*cpu_count()
                                    )

        model = cl.fit(img1d)
        imgCl = model.labels_
        imgCl = imgCl.reshape(img[:,:,0].shape)

        return imgCl

    # ------------------------------------------------------------------------
    # labelsToGeotiff
    # These renderers seem to write tiles or pyramids to disk.
    # To make rendering code easier, write the labels as a geotiff; then the
    # renderer will not need to do it.
    # ------------------------------------------------------------------------
    @staticmethod
    def labelsToGeotiff(referenceDs: gdal.Dataset,
                        labelsFile: Path,
                        labels: numpy.ndarray) -> gdal.Dataset:

        if labelsFile.exists():
            labelsFile.unlink()

        labelsDs = gdal.GetDriverByName('GTiff').Create( \
                        str(labelsFile),
                        xsize=referenceDs.RasterXSize,
                        ysize=referenceDs.RasterYSize,
                        eType=gdal.GDT_Float32,
                        options=['COMPRESS=LZW']
                   )

        labelsDs.SetSpatialRef(referenceDs.GetSpatialRef())
        labelsDs.SetGeoTransform(referenceDs.GetGeoTransform())
        outBand = labelsDs.GetRasterBand(1)
        outBand.WriteArray(labels)
        outBand.SetDescription('Cluster labels')
        outBand.ComputeStatistics(0)  # For min, max used in color table
        labelsDs.FlushCache()
        labelsDs.BuildOverviews()

        return labelsDs
