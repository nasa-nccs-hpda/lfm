
from pathlib import Path

import numpy as np

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

from model.TileDef import TileDef


# ----------------------------------------------------------------------------
# Class Pipeline
# ----------------------------------------------------------------------------
class Pipeline:
    
    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self, outDir: Path, zone: int, zoomLevel: int):
        
        self._outDir: Path = outDir
        self._tileDef: dict = TileDef(zone, zoomLevel)
        
    # ------------------------------------------------------------------------
    # clip
    # ------------------------------------------------------------------------
    def _clip(ll: tuple, 
              ur: tuple, 
              outSRS: osr.SpatialReference,
              ds: gdal.Dataset,
              width: int,
              height: int) -> gdal.Dataset:

        warpOptions = \
            gdal.WarpOptions(outputBounds=[ll[0], ll[1], ur[0], ur[1]],
                             format='MEM',
                             outputBoundsSRS=outSRS,
                             width=width,
                             height=height)

        clipDs: gdal.Dataset = gdal.Warp('', ds, options=warpOptions)

        return clipDs

    # ------------------------------------------------------------------------
    # createCube
    # ------------------------------------------------------------------------
    def _createCube(self,
                    layer: ogr.Layer,
                    ll: tuple,
                    ur: tuple) -> np.ndarray:

        # ---
        # We cannot know the final number of 512 x 512 images in the stack
        # unless we read all the images beforehand; therefore, we cannot 
        # define the ndarray.  Instead, store each 512 x 512 raster in a
        # list, and use it to create the ndarray at the end.
        # ---
        rasterList = []
    
        # ---
        # Read the images and put them in the cube.
        # ---
        for feature in layer:
    
            ds: gdal.Dataset = gdal.Open(feature['location'],
                                         gdalconst.GA_ReadOnly)

            # Verify data type.  Is is ndarray or Dataset?
            clipDs: np.ndarray = _clip(ll, 
                                       ur, 
                                       self._tileDef.srs(), 
                                       ds, 
                                       self._tileDef.width(), 
                                       self._tileDef.height())
    
            raster: np.ndarray = clipDs.ReadAsArray()

            # ---
            # If the raster has one band, the shape will be in two dimensions.
            # If the raster has multiple bands, the shape will be in three
            # dimensions.
            # ---
            numBands = 1 if len(raster.shape) == 2 else raster.shape[0]
        
            if numBands == 1:
            
                rasterList.append(raster)

            else:
            
                for i in range(numBands):
                    rasterList.append(raster[i])

        cube = np.stack(rasterList, axis=0)
        print('shape:', cube.shape)
    
        return cube

    # ------------------------------------------------------------------------
    # query
    # ------------------------------------------------------------------------
    def _query(self, 
               dbPath: Path, 
               llLatLon: tuple, 
               urLatLon: tuple) -> ogr.Layer:

        driver: ogr.Driver = ogr.GetDriverByName('ESRI Shapefile')
        ds: ogr.Dataset = driver.Open(str(dbPath), 0)
        layer: ogr.Layer = ds.GetLayer()
        
        layer.SetSpatialFilterRect(llLatLon[0], 
                                   llLatLon[1], 
                                   urLatLon[0], 
                                   urLatLon[1])

        # ---
        # When you return an ogr.Layer object from a function, the underlying 
        # DataSource that owns the layer may be getting garbage collected, 
        # making the layer invalid.  Attach the datasource to the layer to
        # prevent garbage collection
        # ---
        layer._ds = ds
    
        return layer

    # ------------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------------
    def run(self, tileIndex: tuple[float, float]) -> Path:
        
        # Get the tile corners in LTM.
        ll, ur = self._tileIndex.getTileBbox(tileIndex)
    
        # Query
        llLatLon, urLatLon = Conversions.ltmToLatLon(self._tileDef.zone(), 
                                                     ll, 
                                                     ur, 
                                                     self._tileDef.srs())
            
        layer: ogr.Layer = self._query(TileDef.DB_PATH, llLatLon, urLatLon)

        cubeFile: Path = None
    
        if layer.GetFeatureCount() == 0:
        
            print('Tile does not overlap any images.')

        else:
        
            # Create the data cube.
            cube: np.ndarray = self._createCube(layer, ll, ur)
        
            # ---
            # Write the data cube as a CoG.
            # ---
            cubeFile = self._writeCube(self._outDir, 
                                       self._tileDef.zone(), 
                                       self._tileDef.zoomLevel(), 
                                       tileIndex, 
                                       cube)

        return cubeFile
        
        
