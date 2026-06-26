from pathlib import Path
from typing import List

import os
import numpy as np
import grp

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

    STATIC_FILE_DB = Path('/explore/nobackup/projects/lfm/staticLinks')

    PROJECT_GROUP = 'j1123'

    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self, tileDbPath: Path, outDir: Path, debug: bool = False,
                 targetProductID: str = None):

        if not tileDbPath.exists():
            raise ValueError('Invalid tile DB path: ', tileDbPath)

        if not isinstance(outDir, Path):
            raise TypeError('Output directory must be a Path object.')

        self._outDir: Path = outDir
        self._tileDbPath: Path = tileDbPath
        self._debug: bool = debug
        self._targetProductID: str = targetProductID  # NEW: Store target product ID

        # ---
        # When you return an ogr.Layer object from a function, the underlying
        # DataSource that owns the layer may be getting garbage collected,
        # making the layer invalid.  Attach the datasource to the layer to
        # prevent garbage collection.  This happens in _query().
        # ---
        self._layer = None

    # ------------------------------------------------------------------------
    # clip
    # ------------------------------------------------------------------------
    def _clip(self,
              ulx: float,
              uly: float,
              lrx: float,
              lry: float,
              srs: osr.SpatialReference,
              ds: gdal.Dataset,
              width: int,
              height: int) -> gdal.Dataset:

        clipDs: gdal.Dataset = gdal.Warp(
                '',
                ds,
                outputBounds=[ulx, lry, lrx, uly],
                dstSRS=srs,
                width=width,
                height=height,
                format='MEM',
                resampleAlg=gdal.GRA_Bilinear,
            )

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
                    srs: osr.SpatialReference,
                    width: float,
                    height: float,
                    is_static: bool = False) -> dict:  # NEW: is_static parameter

        # ---
        # We cannot know the final number of 512 x 512 images in the stack
        # unless we read all the images beforehand; therefore, we cannot
        # define the ndarray.  Instead, store each 512 x 512 raster in a
        # dict, and use it to create the ndarray at the end.
        #
        # Each product id maps to a list of tuples, (bandName, raster).
        # ---
        prodIdDict: dict[str, list] = {}  # One output geotiff per product ID

        # ---
        # Read the images and put them in the cube.
        # ---
        numProcessed = 0
        nullCount = 0
        rasterCount = 0
        skippedCount = 0  # NEW: Track skipped files

        for feature in layer:

            numProcessed += 1

            fileName: Path = Path(feature['location'])

            # Skip non-matching product IDs (only for WAC files, not static)
            if self._targetProductID and not is_static:
                file_product_id = fileName.stem.split('.')[0]
                if file_product_id != self._targetProductID:
                    skippedCount += 1
                    continue  # Skip this file entirely

            ds: gdal.Dataset = gdal.Open(str(fileName), gdalconst.GA_ReadOnly)

            try:

                if self._debug:
                    print('Clipping', fileName.name, 'to', ulx, uly, lrx, lry)

                clipDs: gdal.Dataset = self._clip(ulx,
                                                  uly,
                                                  lrx,
                                                  lry,
                                                  srs,
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

                    print('DS dtype:',
                          gdal.GetDataTypeName(ds.GetRasterBand(1).DataType))

                    print('Clip DS dtype:',
                          gdal.GetDataTypeName( \
                              clipDs.GetRasterBand(1).DataType))

            except RuntimeError as e:

                print('The image', fileName, 'did not clip.  Skipping.')
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

                    # Must do this here to avoid empty prod ids.
                    prodId = fileName.stem.split('.')[0]

                    if prodId not in prodIdDict:
                        prodIdDict[prodId]: list[tuple] = []

                    prodIdDict[prodId].append((fileName.stem, raster, ndv))

                else:
                    nullCount += 1

            else:

                for i in range(numBands):

                    ndv = ds.GetRasterBand(i+1).GetNoDataValue()

                    if not (raster == ndv).all():

                        # Must do this here to avoid empty prod ids.
                        prodId = fileName.stem.split('.')[0]

                        if prodId not in prodIdDict:
                            prodIdDict[prodId]: list[tuple] = []

                        key = fileName.stem + '-' + str(i)
                        prodIdDict[prodId].append((key, raster[i], ndv))

                    else:
                        nullCount += 1

            if self._debug:

                print('Raster count:', rasterCount)

                if numProcessed > 99:

                    print('Debug cube size reached.  Stopping.')
                    break

        if self._debug:
            print('Null count:', nullCount)
            if skippedCount > 0:  # NEW: Report skipped count
                print('Skipped count (non-matching product ID):', skippedCount)

        if nullCount == rasterCount:
            print('All bands were filled with no-data values.')

        modality = "WAC" if not is_static else "Static"
        print(f'Total {modality} product IDs: {len(prodIdDict)}')

        return prodIdDict

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
            lrLon: float,
            dbFile: Path = None) -> ogr.Layer:

        if not dbFile:
            dbFile = self._tileDbPath

        if not dbFile.exists():
            raise FileNotFoundError(f"Tile database file not found: {dbFile}")

        driver: ogr.Driver = ogr.GetDriverByName('ESRI Shapefile')
        ds: ogr.Dataset = driver.Open(str(dbFile), 0)

        if ds is None:
            # Get the last GDAL error
            err = gdal.GetLastErrorMsg()
            err_num = gdal.GetLastErrorNo()

            # Build informative error message
            error_msg = f"Failed to open shapefile: {dbFile}"

            if err:
                error_msg += f"\nGDAL Error ({err_num}): {err}"

            # Check for common issues
            if not os.access(str(dbFile), os.R_OK):
                error_msg += f"\n  → Permission denied: You don't have read access to this file"
                error_msg += f"\n  → Current permissions: {oct(os.stat(dbFile).st_mode)[-3:]}"
            elif not dbFile.stat().st_size > 0:
                error_msg += f"\n  → File is empty or corrupted"
            else:
                error_msg += f"\n  → The file may be corrupted or in an unsupported format"

            raise RuntimeError(error_msg)

        layer: ogr.Layer = ds.GetLayer()

        if layer is None:
            raise RuntimeError(f"No layers found in shapefile: {dbFile}")

        # minX, minY, maxX, maxY
        layer.SetSpatialFilterRect(ulLon, lrLat, lrLon, ulLat)

        layer._ds = ds

        return layer

    # ------------------------------------------------------------------------
    # runTileIndex
    # ------------------------------------------------------------------------
    def runTileIndex(self,
                     tileX: int,
                     tileY: int,
                     zone: int,
                     zoomLevel: int) -> list[Path]:

        print('Processing tile (' + str(tileX) + ', ' + str(tileY) + \
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

        # ---
        # Process the dynamic images.
        # ---
        allCubeFiles: list[Path] = []
        layer: ogr.Layer = self._query(ulLat, ulLon, lrLat, lrLon)

        if self._debug:

            print('Tile Bbox LTM:', ulx, uly, lrx, lry)
            print('Tile Bbox Lat/Lon:', ulLat, ulLon, lrLat, lrLon)
            print('Layers from Query:', layer.GetFeatureCount())

        if layer.GetFeatureCount() == 0:

            print('Tile does not overlap any images.')

        else:

            cube: dict = self._createCube(layer,
                                          ulx,
                                          uly,
                                          lrx,
                                          lry,
                                          tileDef.srs,
                                          tileDef.tileWidth,
                                          tileDef.tileHeight,
                                          is_static=False)  # NEW: WAC files

            # Write the data cube as a CoG.
            if len(cube):

                cubeFiles = self._writeCube((tileX, tileY),
                                            cube,
                                            tileDef,
                                            ulx,
                                            uly)

                allCubeFiles += cubeFiles

        # ---
        # Process the static images.
        # ---
        layer: ogr.Layer = self._query(ulLat,
                                       ulLon,
                                       lrLat,
                                       lrLon,
                                       Pipeline.STATIC_FILE_DB)

        if self._debug:

            print('Tile Bbox LTM:', ulx, uly, lrx, lry)
            print('Tile Bbox Lat/Lon:', ulLat, ulLon, lrLat, lrLon)
            print('Layers from Query:', layer.GetFeatureCount())

        if layer.GetFeatureCount() == 0:

            print('Tile does not overlap any static images.')

        else:

            staticCube: dict = self._createCube(layer,
                                                ulx,
                                                uly,
                                                lrx,
                                                lry,
                                                tileDef.srs,
                                                tileDef.tileWidth,
                                                tileDef.tileHeight,
                                                is_static=True)  # NEW: Static files

            # Write the data cube as a CoG.
            if len(staticCube):

                staticCubeFile = self._writeStaticCube((tileX, tileY),
                                                       staticCube,
                                                       tileDef,
                                                       ulx,
                                                       uly)

                allCubeFiles.append(staticCubeFile)

        return allCubeFiles

    # ------------------------------------------------------------------------
    # runPoint
    # ------------------------------------------------------------------------
    def runPoint(self,
                 lat: float,
                 lon: float,
                 zone: str,
                 zoomLevel: int) -> list[Path]:

        # Find the tile index for the given point, zone and zoom.
        tileDef = TmsTileDef.initFromParams(zone, zoomLevel)
        tileX, tileY = tileDef.llToTileIndex(lat, lon)

        # Run that tile index.
        return self.runTileIndex(tileX,
                                 tileY,
                                 zone,
                                 zoomLevel)

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
            print('Tile Indexes:', tileIndexes)

        cubeFiles = []

        # Make a cube for each tile index.
        for idx in tileIndexes:

            cubeFiles += self.runTileIndex(idx['tileX'],
                                           idx['tileY'],
                                           idx['zone'],
                                           idx['zoomLevel'])

        return cubeFiles

    # ------------------------------------------------------------------------
    # writeStaticCube
    # ------------------------------------------------------------------------
    def _writeStaticCube(self,
                         tileIndex: tuple[int, int],
                         prodIdDict: dict,
                         tileDef: dict,
                         ulx: float,
                         uly: float) -> Path:

        # Name the file.
        outName = 'StaticCube-LTM' + tileDef.zone + \
                  '_Zoom-' + str(tileDef.zoomLevel) + \
                  '_Tile-' + str(tileIndex[0]) + '-' + str(tileIndex[1]) + \
                  '.tif'

        outFile = self._outDir / outName

        # Get information about the output.
        firstRaster = list(prodIdDict.values())[0][0]
        name = firstRaster[0]
        raster = firstRaster[1]

        dataType = gdal_array.NumericTypeCodeToGDALTypeCode(raster.dtype)
        width = raster.shape[0]
        height = raster.shape[1]
        numBands = len(prodIdDict)

        # Create the dataset.
        ds = gdal.GetDriverByName('GTiff').Create(str(outFile),
                                                  height,
                                                  width,
                                                  numBands,
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
        NO_DATA_VAL = -32768

        for pid, rasters in prodIdDict.items():

            for raster in rasters:

                name = raster[0]
                pixels = raster[1]
                noDataValue = raster[2]

                # This would be better in createCube().
                raster = np.where(pixels == noDataValue, NO_DATA_VAL, pixels)

                bandIndex += 1
                band = ds.GetRasterBand(bandIndex)
                band.WriteArray(pixels)
                band.SetMetadataItem('Name', name)
                band.SetNoDataValue(NO_DATA_VAL)

        ds = None

        # 6/26: making cubes group-accessible
        try:
            gid = grp.getgrnam(self.PROJECT_GROUP).gr_gid
            os.chown(outFile, -1, gid)  # -1 means don't change owner
        except (PermissionError, KeyError, OSError) as e:
            print(f"Warning: Could not set group to {self.PROJECT_GROUP}: {e}")

        outFile.chmod(0o664)  # rw-rw-r--

        return outFile

    # ------------------------------------------------------------------------
    # writeCube
    # ------------------------------------------------------------------------
    def _writeCube(self,
                   tileIndex: tuple[int, int],
                   prodIdDict: dict,
                   tileDef: dict,
                   ulx: float,
                   uly: float) -> list[Path]:

        outFiles: list[Path] = []

        for pid, rasters in prodIdDict.items():

            # Name the file.
            outName = 'Cube-LTM' + tileDef.zone + \
                      '_Zoom-' + str(tileDef.zoomLevel) + \
                      '_Tile-' + str(tileIndex[0]) + '-' + \
                      str(tileIndex[1]) + \
                      '_ProdId-' + pid + \
                      '.tif'

            outFile = self._outDir / outName

            # Get information about the output.
            firstRaster = rasters[0]
            name = firstRaster[0]
            raster = firstRaster[1]

            dataType = gdal_array.NumericTypeCodeToGDALTypeCode(raster.dtype)
            width = raster.shape[0]
            height = raster.shape[1]
            numBands = len(rasters)

            # Create the dataset.
            ds = gdal.GetDriverByName('GTiff').Create(
                 str(outFile),
                 height,
                 width,
                 numBands,
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

            # Write the band.
            bandIndex = 0

            for raster in rasters:

                name = raster[0]
                pixels = raster[1]
                noDataValue = raster[2]

                bandIndex += 1
                band = ds.GetRasterBand(bandIndex)
                band.WriteArray(pixels)
                band.SetMetadataItem('Name', name)
                band.SetNoDataValue(noDataValue)

            ds = None

            # 6/26: making cubes group-accessible
            try:
                gid = grp.getgrnam(self.PROJECT_GROUP).gr_gid
                os.chown(outFile, -1, gid)  # -1 means don't change owner
            except (PermissionError, KeyError, OSError) as e:
                print(f"Warning: Could not set group to {self.PROJECT_GROUP}: {e}")

            outFile.chmod(0o664)  # rw-rw-r--

            outFiles.append(outFile)

        return outFiles