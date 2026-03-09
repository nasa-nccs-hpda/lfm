
from pathlib import Path
from typing import List

import numpy as np

from osgeo import gdal
from osgeo import gdal_array
from osgeo import gdalconst
from osgeo import ogr
from osgeo import osr

from model.Conversions import Conversions
from model.TileDef import TileDef


# ----------------------------------------------------------------------------
# Class Pipeline
# ----------------------------------------------------------------------------
class Pipeline:
    
    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self, tileDbPath: Path, outDir: Path):
        
        self._outDir: Path = outDir
        self._tileDbPath: Path = tileDbPath
        self._tileDef: dict = None
        
    # ------------------------------------------------------------------------
    # clip
    # ------------------------------------------------------------------------
    def _clip(self,
              ll: tuple, 
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
            clipDs: np.ndarray = self._clip(ll, 
                                            ur, 
                                            self._tileDef.srs, 
                                            ds, 
                                            self._tileDef.tileWidth, 
                                            self._tileDef.tileHeight)
    
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
    def _query(self, llLatLon: tuple, urLatLon: tuple) -> ogr.Layer:

        driver: ogr.Driver = ogr.GetDriverByName('ESRI Shapefile')
        ds: ogr.Dataset = driver.Open(str(self._tileDbPath), 0)
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
    # queryTMS
    # ------------------------------------------------------------------------
    def _queryTMS(self, 
                  llLatLon: tuple, 
                  urLatLon: tuple, 
                  zoomLevel: int) -> List[dict]:

        driver: ogr.Driver = ogr.GetDriverByName('GPKG')
        ds: ogr.Dataset = driver.Open(str(TileDef.DB_PATH), 0)
        layer: ogr.Layer = ds.GetLayer()
        layer.SetAttributeFilter('zoom_level = ' + str(zoomLevel))

        layer.SetSpatialFilterRect(llLatLon[0], 
                                   llLatLon[1], 
                                   urLatLon[0], 
                                   urLatLon[1])

        indexes = []
        
        for feature in layer:
            
            index = {'tileY': feature.GetField('tile_row'),
                     'tileX': feature.GetField('tile_col'),
                     'zone': feature.GetField('zone_name'),
                     'zoomLevel': feature.GetField('zoom_level')}
                     
            indexes.append(index)
             
        # ---
        # When you return an ogr.Layer object from a function, the underlying 
        # DataSource that owns the layer may be getting garbage collected, 
        # making the layer invalid.  Attach the datasource to the layer to
        # prevent garbage collection
        # ---
        layer._ds = ds
    
        return indexes

    # ------------------------------------------------------------------------
    # runTileIndex
    # ------------------------------------------------------------------------
    def runTileIndex(self, 
                     tileX: int,
                     tileY: int,
                     zone: int, 
                     zoomLevel: int) -> Path:
        
        print('Processing (' + str(tileX) + ', ' + str(tileY) + \
              ') / zone ' + str(zone) + \
              ' / zoom ' + str(zoomLevel))
        
        self._tileDef: dict = TileDef(zone, zoomLevel)

        # Get the tile corners in LTM.
        ll, ur = self._tileDef.getTileBbox(tileX, tileY)
    
        # Query
        llLatLon, urLatLon = Conversions.ltmToLatLon(self._tileDef.zone, 
                                                     ll, 
                                                     ur, 
                                                     self._tileDef.srs)
            
        layer: ogr.Layer = self._query(llLatLon, urLatLon)

        cubeFile: Path = None
    
        if layer.GetFeatureCount() == 0:
        
            print('Tile does not overlap any images.')

        else:
        
            # Create the data cube.
            cube: np.ndarray = self._createCube(layer, ll, ur)
        
            # Write the data cube as a CoG.
            cubeFile = self._writeCube((tileX, tileY), cube)

        return cubeFile

    # ------------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------------
    def run(self, 
            upperLeft: tuple[float, float], 
            lowerRight: tuple[float, float],
            zoomLevel: int) -> List[Path]:
        
        # Query the AoI with the TMS GeoPackage.
        tileIndexes: List[tuple[int, int]] = self._queryTMS(upperLeft, 
                                                            lowerRight, 
                                                            zoomLevel)
               
        cubeFiles = []
        
        for idx in tileIndexes:
            
            cubeFiles.append(self.runTileIndex(idx['tileX'],
                                               idx['tileY'],
                                               idx['zone'], 
                                               idx['zoomLevel']))
        
        return cubeFiles
        
    # ------------------------------------------------------------------------
    # writeCube
    # ------------------------------------------------------------------------
    def _writeCube(self, 
                   tileIndex: tuple[int, int], 
                   cube: np.ndarray) -> Path:

        outName = 'Cube-LTM' + self._tileDef.zone + \
                  '_Zoom-' + str(self._tileDef.zoomLevel) + \
                  '_Tile-' + str(tileIndex[0]) + '-' + str(tileIndex[1]) + \
                  '.tif'
    
        outFile = self._outDir / outName

        # ---
        # First create a temporary in-memory dataset
        # ---
        dataType = gdal_array.NumericTypeCodeToGDALTypeCode(cube.dtype)
        memDriver = gdal.GetDriverByName('MEM')
    
        memDs = memDriver.Create('', 
                                 cube.shape[1],    # width
                                 cube.shape[2],    # height  
                                 cube.shape[0],    # bands
                                 dataType)
    
        if memDs is None:
            raise RuntimeError('Failed to create dataset in memory.')
    
        memDs.SetSpatialRef(self._tileDef.srs)
        
        geotransform = [
            self._tileDef.pointOfOrigin[0],  # X-coordinate of ul corner
            self._tileDef.cellSize,          # Pixel width (west-east res)
            0,                               # Rotation (usually 0)
            self._tileDef.pointOfOrigin[1],  # Y-coordinate of ul corner  
            0,                               # Rotation (usually 0)
            -self._tileDef.cellSize          # Pixel height (north-south res)
        ]
    
        memDs.SetGeoTransform(geotransform)

        # ---
        # Write data to temporary dataset
        # ---
        for bandIndex in range(cube.shape[0]):
        
            band = memDs.GetRasterBand(bandIndex + 1)
            band.WriteArray(cube[bandIndex])

        # ---
        # Now create COG using CreateCopy.
        # ---
        cogDriver = gdal.GetDriverByName('COG')
    
        cogDs = cogDriver.CreateCopy(str(outFile),
                                     memDs,
                                     options=['BIGTIFF=YES', 'COMPRESS=LZW'])
    
        tempDs = None
        cogDs = None

        return outFile
