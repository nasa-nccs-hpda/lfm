from pathlib import Path
import sys

import numpy

from osgeo import gdal


# ----------------------------------------------------------------------------
# Class ImageHelper
#
# TODO: add accessors using @property.
# ----------------------------------------------------------------------------
class ImageHelper(object):
    
    # ------------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------------
    def __init__(self):
        
        self._dataset: gdal.Dataset = None
        self._inputFile: Path = None
        self._bandId: int = 1
        self._band: numpy.ndarray = None
        self._noDataValue: float = -9999.0
        self._minValue: float = sys.float_info.max
        self._maxValue: float = sys.float_info.min
  
    # ------------------------------------------------------------------------
    # initFromDataset
    # ------------------------------------------------------------------------
    def initFromDataset(self, 
                        dataset: gdal.Dataset,
                        noDataValue: float,
                        bandId: int = 1,
                       ) -> None:
        
        self._dataset = dataset
        
        self._completeInitialization(noDataValue, bandId)
        
    # ------------------------------------------------------------------------
    # initFromFile
    # ------------------------------------------------------------------------
    def initFromFile(self, 
                     inputFile: Path,
                     noDataValue: float,
                     bandId: int = 1,
                    ) -> None:
        
        # TODO: check validity and comment.
        self._inputFile = inputFile
        self._dataset: gdal.Dataset = gdal.Open(str(self._inputFile))
        
        self._completeInitialization(noDataValue, bandId)
        
    # ------------------------------------------------------------------------
    # completeInitialization
    # ------------------------------------------------------------------------
    def _completeInitialization(self,
                                noDataValue: float,
                                bandId: int = 1,
                               ) -> None:
        
        self._bandId = bandId
        
        # Generate overviews, if they do not exist.
        if self._dataset.GetRasterBand(1).GetOverviewCount() == 0:
            dummy = self._dataset.BuildOverviews()

        # ---
        # Read the band.
        # ---
        self._band: numpy.ndarray = \
            self._dataset.GetRasterBand(self._bandId).ReadAsArray()
        
        # ---
        # Initialize the no-data value.
        # ---
        self._noDataValue = self._dataset.GetRasterBand(self._bandId). \
            GetNoDataValue() or noDataValue
       
        # ---
        # Compute the minimum and maximum pixels values to help the renderer.
        # ---
        forExtremes = self._band[self._band != self._noDataValue]
        self._minValue: float = forExtremes.min()
        self._maxValue: float = forExtremes.max()

    # ------------------------------------------------------------------------
    # getCorners
    #
    # Why is this not built into gdal?
    # ------------------------------------------------------------------------
    def getCorners(self):
        
        minx, xres, xskew, maxy, yskew, yres = \
            self._dataset.GetGeoTransform()
        
        maxx = minx + (self._dataset.RasterXSize * xres)
        miny = maxy + (self._dataset.RasterYSize * yres)

        return (minx, miny, maxx, maxy)
    
    # ------------------------------------------------------------------------
    # getBandIndex
    # ------------------------------------------------------------------------
    def getBandIndex(self) -> int:
        
        return self._bandId
    
    # ------------------------------------------------------------------------
    # getBand
    # ------------------------------------------------------------------------
    def getBand(self) -> numpy.ndarray:
        
        return self._band

    # ------------------------------------------------------------------------
    # __str__
    # ------------------------------------------------------------------------
    def __str__(self):
        
        return ('Input file: ' + str(self._inputFile) + 
                '\nMin. pixel: ' + str(self._minValue) +
                '\nMax. pixel: ' + str(self._maxValue) + 
                '\nNo-data value: ' + str(self._noDataValue) + 
                '\nBand index: ' + str(self._bandId) +
                '\nCorners: ' + str(self.getCorners())
               )
