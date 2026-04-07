
from pathlib import Path
from typing import List

import numpy as np
import xarray as xr

from osgeo import gdal
from osgeo import gdal_array
from osgeo import gdalconst
from osgeo import ogr
from osgeo import osr

from lfm.model.TmsIntersector import TmsIntersector
from lfm.model.TmsTileDef import TmsTileDef

gdal.UseExceptions()


# ----------------------------------------------------------------------------
# Class Pipeline
#
# TODO: Remove self._tileDef because it acts like a global variable.  Pass it.
# ----------------------------------------------------------------------------
class Pipeline:
    
    MOON_SRS = ('GEOGCRS["Moon (2015) - Sphere / Ocentric",'
                'DATUM["Moon (2015) - Sphere",'
                'ELLIPSOID["Moon (2015) - Sphere",1737400,0,'
                'LENGTHUNIT["metre",1]]],'
                'PRIMEM["Reference Meridian",0,'
                'ANGLEUNIT["degree",0.0174532925199433]],'
                'CS[ellipsoidal,2],'
                'AXIS["geodetic latitude (Lat)",north,'
                'ORDER[1],'
                'ANGLEUNIT["degree",0.0174532925199433]],'
                'AXIS["geodetic longitude (Lon)",east,'
                'ORDER[2],'
                'ANGLEUNIT["degree",0.0174532925199433]],'
                'ID["IAU",30100,2015],'
                'REMARK["Source of IAU Coordinate systems:'
                ' https://doi.org/10.1007/s10569-017-9805-5"]]')

    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self, tileDbPath: Path, outDir: Path, debug: bool = False):
        
        if not tileDbPath.exists():
            raise ValueError('Invalid tile DB path: ', tileDbPath)
        
        if not isinstance(outDir, Path):
            raise TypeError('Output directory must be a Path object.')
            
        self._outDir: Path = outDir
        self._tileDbPath: Path = tileDbPath
        self._debug: bool = debug
        
    # ------------------------------------------------------------------------
    # clip
    # ------------------------------------------------------------------------
    def _clip(self,
              ulx: float,
              uly: float,
              lrx: float,
              lry: float,
              ds: gdal.Dataset,
              width: int,
              height: int) -> gdal.Dataset:

        clipDs: gdal.Dataset = \
            gdal.Translate('',
                           ds,
                           projWin=[ulx, uly, lrx, lry],
                           width=width,
                           height=height,
                           format='MEM')

        return clipDs

    # ------------------------------------------------------------------------
    # createCube
    # ------------------------------------------------------------------------
    def _createCube(self,
                    layer: ogr.Layer,
                    ulx: float,
                    uly: float,
                    lrx: float,
                    lry: float,
                    width: float,
                    height: float) -> dict:  # np.ndarray:

        # ---
        # We cannot know the final number of 512 x 512 images in the stack
        # unless we read all the images beforehand; therefore, we cannot 
        # define the ndarray.  Instead, store each 512 x 512 raster in a
        # list, and use it to create the ndarray at the end.
        # ---
        rasterList = {}
    
        # ---
        # Read the images and put them in the cube.
        # ---
        numProcessed = 0
        nullCount = 0
        rasterCount = 0
        
        for feature in layer:
    
            numProcessed += 1
            
            fileName: Path = Path(feature['location'])

            ds: gdal.Dataset = gdal.Open(fileName, gdalconst.GA_ReadOnly)

            try:
                
                if self._debug:
                    print('Clipping', fileName.name, 'to', ulx, uly, lrx, lry)
                
                clipDs: gdal.Dataset = self._clip(ulx,
                                                  uly,
                                                  lrx,
                                                  lry,
                                                  ds, 
                                                  width, 
                                                  height)

                if self._debug:

                    corners = self._getCorners(clipDs)
                    cUlx = corners['upperLeft'][0]
                    cUly = corners['upperLeft'][1]
                    cLrx = corners['lowerRight'][0]
                    cLry = corners['lowerRight'][1]
                    print('Clip result:', cUlx, cUly, cLrx, cLry)
                    print('Size:', clipDs.RasterXSize, clipDs.RasterYSize)
                    
            except RuntimeError:
                
                print('The image did not clip.  Skipping.')
                continue
                
            raster: np.ndarray = clipDs.ReadAsArray()  # Float32
            
            # ---
            # If the raster has one band, the shape will be in two dimensions.
            # If the raster has multiple bands, the shape will be in three
            # dimensions.
            # ---
            numBands = 1 if len(raster.shape) == 2 else raster.shape[0]
            rasterCount += numBands
        
            if numBands == 1:
            
                ndv = ds.GetRasterBand(1).GetNoDataValue()

                if not (raster == ndv).all():
                    
                    rasterList[fileName.stem] = raster

                else:
                    nullCount += 1
                
            else:
            
                for i in range(numBands):
                    
                    ndv = ds.GetRasterBand(i+1).GetNoDataValue()

                    if not (raster == ndv).all():
                
                        key = fileName.stem + '-' + str(i)
                        rasterList[key] = raster[i]
                        
                    else:
                        nullCount += 1
                    
            if self._debug and numProcessed > 99:
                
                print('Debug cube size reached.  Stopping.')
                break
            
        if self._debug:
            
            print('Null count:', nullCount)
            
        if nullCount == rasterCount:
            
            print('All bands were filled with no-data values.')
            
        print('Total bands:', len(rasterList))
        
        if len(rasterList):

            print('Raster shape:', list(rasterList.values())[0].shape)
    
        return rasterList

    # ------------------------------------------------------------------------
    # getCorners
    # ------------------------------------------------------------------------
    def _getCorners(self, ds: gdal.Dataset) -> dict:

        '''
        This method is a helper used for debugging.
        '''
        gt = ds.GetGeoTransform()
        cols = ds.RasterXSize
        rows = ds.RasterYSize

        # GeoTransform: (originX, pixelWidth, rotX, originY, rotY, pixelHeight)
        def pixelToCoord(col, row):
            x = gt[0] + col * gt[1] + row * gt[2]
            y = gt[3] + col * gt[4] + row * gt[5]
            return (x, y)

        return {'upperLeft':  pixelToCoord(0, 0),
                'lowerRight': pixelToCoord(cols, rows)}

    # ------------------------------------------------------------------------
    # query
    # ------------------------------------------------------------------------
    def _query(self,
               ulLat: float, 
               ulLon: float, 
               lrLat: float, 
               lrLon: float) -> ogr.Layer:

        driver: ogr.Driver = ogr.GetDriverByName('ESRI Shapefile')
        ds: ogr.Dataset = driver.Open(str(self._tileDbPath), 0)
        layer: ogr.Layer = ds.GetLayer()
        
        # minX, minY, maxX, maxY
        layer.SetSpatialFilterRect(ulLon, lrLat, lrLon, ulLat)

        # ---
        # When you return an ogr.Layer object from a function, the underlying 
        # DataSource that owns the layer may be getting garbage collected, 
        # making the layer invalid.  Attach the datasource to the layer to
        # prevent garbage collection
        # ---
        layer._ds = ds
    
        return layer

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
        
        tileDef: dict = TmsTileDef.initFromParams(zone, zoomLevel)

        # Get the tile corners, which are in LTM.
        ulx, uly, lrx, lry = tileDef.getTileBbox(tileX, tileY)
        
        # ---
        # Query the tile-index database.  It is in lat/lon.
        # ---
        ulLat, ulLon = tileDef.ltmToLatLon(ulx, uly)
        lrLat, lrLon = tileDef.ltmToLatLon(lrx, lry)
        layer: ogr.Layer = self._query(ulLat, ulLon, lrLat, lrLon)

        if self._debug:
            
            print('Tile Bbox LTM:', ulx, uly, lrx, lry)
            print('Tile Bbox Lat/Lon:', ulLat, ulLon, lrLat, lrLon)
            print('Layers from Query:', layer.GetFeatureCount())
            
        # ---
        # Create the cube.
        # ---
        cubeFile: Path = None
    
        if layer.GetFeatureCount() == 0:
        
            print('Tile does not overlap any images.')

        else:
        
            cube: dict = self._createCube(layer, 
                                          ulx, 
                                          uly, 
                                          lrx, 
                                          lry, 
                                          tileDef.tileWidth, 
                                          tileDef.tileHeight)
        
            # Write the data cube as a CoG.
            if len(cube):  
        
                cubeFile = \
                    self._writeCube((tileX, tileY), cube, tileDef, ulx, uly)

        return cubeFile

    # ------------------------------------------------------------------------
    # runPoint
    # ------------------------------------------------------------------------
    def runPoint(self, 
                 lat: float, 
                 lon: float, 
                 zone: str, 
                 zoomLevel: int) -> Path:
        
        # Find the tile index for the given point, zone and zoom.
        tileDef = TmsTileDef.initFromParams(zone, zoomLevel)
        tileX, tileY = tileDef.llToTileIndex(lat, lon)
        
        # Run that tile index.
        return self.runTileIndex(tileX, tileY, zone, zoomLevel)
        
    # ------------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------------
    def run(self, 
            ulLat: float,
            ulLon: float,
            lrLat: float,
            lrLon: float,
            zoomLevel: int) -> List[Path]:
        
        # Get all the tile ids in all zones that intersect the bounding box.
        tmsi = TmsIntersector()
        tileIndexes = tmsi.getTids(ulLat, ulLon, lrLat, lrLon, zoomLevel)
        
        if self._debug:
            print('Num Tile Indexes:', len(tileIndexes))
        
        cubeFiles = []
        
        # Make a cube for each tile index.
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
                   cubeDict: dict,
                   tileDef: dict,
                   ulx: float,
                   uly: float) -> Path:

        # Name the file.
        outName = 'Cube-LTM' + tileDef.zone + \
                  '_Zoom-' + str(tileDef.zoomLevel) + \
                  '_Tile-' + str(tileIndex[0]) + '-' + str(tileIndex[1]) + \
                  '.tif'

        outFile = self._outDir / outName
        
        # Get information about the output.
        raster = list(cubeDict.values())[0]
        dataType = gdal_array.NumericTypeCodeToGDALTypeCode(raster.dtype)
        width = raster.shape[0]
        height = raster.shape[1]
        bands = len(cubeDict)
        
        # Create the dataset.
        ds = gdal.GetDriverByName('GTiff').Create(
            str(outFile),
            height,
            width,
            bands,
            dataType,
            options=['BIGTIFF=YES',
                     'TILED=YES',
                     'COMPRESS=LZW'])

        # Set the spatial reference.
        ds.SetSpatialRef(tileDef.srs)

        geotransform = [ulx,
                        tileDef.cellSize,
                        0,
                        uly,
                        0,
                        -tileDef.cellSize]

        ds.SetGeoTransform(geotransform)

        # Write the bands.
        bandIndex = 0

        for bandName, cube in cubeDict.items():

            bandIndex += 1
            band = ds.GetRasterBand(bandIndex)
            band.WriteArray(cube)
            band.SetMetadataItem('Name', bandName)
        
        ds = None
        
        return outFile
        