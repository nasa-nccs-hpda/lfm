from pathlib import Path
from typing import List

import numpy as np
import warnings

from osgeo import gdal
from osgeo import gdal_array
from osgeo import gdalconst
from osgeo import ogr
from osgeo import osr

from .TmsIntersector import TmsIntersector
from .TmsTileDef import TmsTileDef
from .lunar_crs import (
    LUNAR_GEOGRAPHIC_WKT_PATH,
    load_lunar_geographic_wkt,
)
from .vector_index import open_vector_layer

gdal.UseExceptions()

# ----------------------------------------------------------------------------
# Class Pipeline
# ----------------------------------------------------------------------------
class Pipeline:
    """Legacy WAC/static tiler retained as a migration compatibility adapter.

    New code should use ``TileConfig`` with ``create_tiles_for_index``,
    ``create_tiles_for_point``, or ``create_tiles_for_aoi`` from
    :mod:`lfm.model`.
    """

    MOON_SRS_PATH = LUNAR_GEOGRAPHIC_WKT_PATH
    MOON_SRS = load_lunar_geographic_wkt()

    STATIC_FILE_DB = Path('/explore/nobackup/projects/lfm/staticLinks')

    PROJECT_GROUP = 'j1123'
    STATIC_OUTPUT_NODATA = np.float32(-32768.0)
    STATIC_PRESERVE_SOURCE_NODATA_MARKERS = ('deltacpr', 'deltas1')

    # ------------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------------
    def __init__(self, tileDbPath: Path, outDir: Path, debug: bool = False,
                 targetProductID: str = None):

        warnings.warn(
            "Pipeline(tileDbPath, outDir, ...) is deprecated; use TileConfig "
            "with the create_tiles_for_* API.",
            DeprecationWarning,
            stacklevel=2,
        )

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
        # making the layer invalid. Retain the current datasource on the
        # Pipeline for at least as long as its layer is in use.
        # ---
        self._layer = None
        self._indexDataset = None

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
              height: int,
              srcNodata=None,
              dstNodata=None) -> gdal.Dataset:

        warp_kwargs = {
            'outputBounds': [ulx, lry, lrx, uly],
            'dstSRS': srs,
            'width': width,
            'height': height,
            'format': 'MEM',
            'resampleAlg': gdal.GRA_Bilinear,
        }

        if srcNodata is not None:
            warp_kwargs['srcNodata'] = srcNodata
            if dstNodata is not None:
                warp_kwargs['dstNodata'] = dstNodata
        elif dstNodata is not None:
            warp_kwargs['dstNodata'] = dstNodata

        clipDs: gdal.Dataset = gdal.Warp('', ds, **warp_kwargs)

        return clipDs

    # ------------------------------------------------------------------------
    # getStaticOutputNodata
    # ------------------------------------------------------------------------
    def _getStaticOutputNodata(self,
                               fileName: Path,
                               sourceNoDataValue):

        if sourceNoDataValue is not None:

            name = fileName.name.lower()
            if any(marker in name for marker in
                   Pipeline.STATIC_PRESERVE_SOURCE_NODATA_MARKERS):

                return np.float32(sourceNoDataValue)

        return Pipeline.STATIC_OUTPUT_NODATA

    # ------------------------------------------------------------------------
    # normalizeStaticRaster
    # ------------------------------------------------------------------------
    def _normalizeStaticRaster(self,
                               raster: np.ndarray,
                               sourceNoDataValue,
                               outputNoDataValue) -> np.ndarray:

        normalized = np.asarray(raster).copy()
        outputNoDataValue = np.float32(outputNoDataValue)

        if sourceNoDataValue is not None and sourceNoDataValue != outputNoDataValue:
            normalized = np.where(normalized == sourceNoDataValue,
                                  outputNoDataValue,
                                  normalized)

        normalized = np.where(np.isfinite(normalized),
                              normalized,
                              outputNoDataValue)

        return normalized.astype(raster.dtype, copy=False)

    # ------------------------------------------------------------------------
    # nodataArg
    # ------------------------------------------------------------------------
    @staticmethod
    def _nodataArg(values: list):

        filtered = [v for v in values if v is not None]
        if not filtered:
            return None
        if len(values) == 1:
            return filtered[0]
        return [v if v is not None else None for v in values]

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

                sourceNoDataValues = [
                    ds.GetRasterBand(i + 1).GetNoDataValue()
                    for i in range(ds.RasterCount)
                ]
                outputNoDataValues = None

                if is_static:
                    outputNoDataValues = [
                        self._getStaticOutputNodata(fileName, sourceNoDataValue)
                        for sourceNoDataValue in sourceNoDataValues
                    ]

                if self._debug:
                    print('Clipping', fileName.name, 'to', ulx, uly, lrx, lry)

                clipDs: gdal.Dataset = self._clip(ulx,
                                                  uly,
                                                  lrx,
                                                  lry,
                                                  srs,
                                                  ds,
                                                  width,
                                                  height,
                                                  srcNodata=self._nodataArg(
                                                      sourceNoDataValues)
                                                  if is_static else None,
                                                  dstNodata=self._nodataArg(
                                                      outputNoDataValues)
                                                  if is_static else None)

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

                ndv = sourceNoDataValues[0]
                outNodata = outputNoDataValues[0] if is_static else ndv
                bandRaster = raster

                if is_static:
                    bandRaster = self._normalizeStaticRaster(bandRaster,
                                                             ndv,
                                                             outNodata)

                if not (bandRaster == outNodata).all():

                    # Must do this here to avoid empty prod ids.
                    prodId = fileName.stem.split('.')[0]

                    if prodId not in prodIdDict:
                        prodIdDict[prodId]: list[tuple] = []

                    prodIdDict[prodId].append((fileName.stem,
                                               bandRaster,
                                               outNodata))

                else:
                    nullCount += 1

            else:

                for i in range(numBands):

                    ndv = sourceNoDataValues[i]
                    outNodata = outputNoDataValues[i] if is_static else ndv
                    bandRaster = raster[i]

                    if is_static:
                        bandRaster = self._normalizeStaticRaster(bandRaster,
                                                                 ndv,
                                                                 outNodata)

                    if not (bandRaster == outNodata).all():

                        # Must do this here to avoid empty prod ids.
                        prodId = fileName.stem.split('.')[0]

                        if prodId not in prodIdDict:
                            prodIdDict[prodId]: list[tuple] = []

                        key = fileName.stem + '-' + str(i)
                        prodIdDict[prodId].append((key,
                                                   bandRaster,
                                                   outNodata))

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

        if rasterCount > 0 and nullCount == rasterCount:
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

        self._indexDataset, layer = open_vector_layer(dbFile)

        # minX, minY, maxX, maxY
        layer.SetSpatialFilterRect(ulLon, lrLat, lrLon, ulLat)

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
        noDataValue = float(Pipeline.STATIC_OUTPUT_NODATA)

        for pid, rasters in prodIdDict.items():

            for raster in rasters:

                name = raster[0]
                pixels = raster[1]

                raster = pixels

                bandIndex += 1
                band = ds.GetRasterBand(bandIndex)
                band.WriteArray(raster)
                band.SetMetadataItem('Name', name)
                band.SetNoDataValue(noDataValue)

        ds = None

        # 6/26: making cubes group-accessible
        # try:
        #     # gid = grp.getgrnam(self.PROJECT_GROUP).gr_gid
        #     # os.chown(outFile, -1, gid)  # -1 means don't change owner
        #     cmd = f"chgrp {self.PROJECT_GROUP} {outFile}"
        #     os.system(cmd)
        # except (PermissionError, KeyError, OSError) as e:
        #     print(f"Warning: Could not set group to {self.PROJECT_GROUP}: {e}")

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
            # try:
            #     # gid = grp.getgrnam(self.PROJECT_GROUP).gr_gid
            #     # os.chown(outFile, -1, gid)  # -1 means don't change owner
            #     cmd = f"chgrp {self.PROJECT_GROUP} {outFile}"
            #     os.system(cmd)
            # except (PermissionError, KeyError, OSError) as e:
            #     print(f"Warning: Could not set group to {self.PROJECT_GROUP}: {e}")

            outFile.chmod(0o664)  # rw-rw-r--

            outFiles.append(outFile)

        return outFiles
