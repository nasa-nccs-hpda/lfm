
from pathlib import Path
import tempfile
import unittest

from osgeo import gdal
from osgeo import gdalconst
from osgeo import ogr

from lfm.model.Pipeline import Pipeline
from lfm.model.ResamplingMethod import ResamplingMethod
from lfm.model.TmsIntersector import TmsIntersector
from lfm.model.TmsTileDef import TmsTileDef


# -----------------------------------------------------------------------------
# class PipelineTestCase
#
# python -m unittest lfm.model.tests.test_Pipeline
# python -m unittest lfm.model.tests.test_Pipeline.PipelineTestCase.testInit
# -----------------------------------------------------------------------------
class PipelineTestCase(unittest.TestCase):

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
    # setUp
    # ------------------------------------------------------------------------
    def setUp(self):

        self._imageDir = Path('/explore/nobackup/projects/lfm/' + \
                              'processed_data/Lunar/LRO_WAC_Pho_Sites')
        
        self._tileDbPath = self.getTileDbPath(self._imageDir)
        
    # ------------------------------------------------------------------------
    # getTileDbPath
    # ------------------------------------------------------------------------
    def getTileDbPath(self,
                      imageDir: Path, 
                      dbName: str = 'output_index.shp') -> Path:

        if not imageDir or not imageDir.exists() or not imageDir.is_dir():
            raise ValueError('A valid image directory must be provided.')

        fullPath: Path = imageDir / dbName

        if fullPath.exists():
            return fullPath

        outFile: Path = imageDir / dbName

        # The database does not exist, so create it for the image directory.
        gdal.TileIndex(outFile, 
                       list(imageDir.glob('*.tif')), 
                       outputSRS=PipelineTestCase.MOON_SRS)
                       
        outFile.chmod(666)

        return outFile
      
    # -------------------------------------------------------------------------
    # testInit
    # -------------------------------------------------------------------------
    def testInit(self):
        
        outDir = Path(tempfile.mkdtemp())
        pl = Pipeline(self._tileDbPath, outDir)
        
        self.assertEqual(pl._tileDbPath, self._tileDbPath)
        self.assertEqual(pl._outDir, outDir)

    # -------------------------------------------------------------------------
    # testClip
    # -------------------------------------------------------------------------
    def testClip(self):
        
        # Pipeline clips to the bbox of the tile.
        zone = '42N'
        zoomLevel = 5
        tileDef: dict = TmsTileDef.initFromParams(zone, zoomLevel)
        ulx, uly, lrx, lry = tileDef.getTileBbox(1, 63)
        
        inImage = Path('/explore/nobackup/projects/lfm/' +
                       'processed_data/Lunar/LRO_NAC_Pho_Sites/' +
                       'M1117899885LE.ech.cog.tif')

        ds = gdal.Open(inImage, gdalconst.GA_ReadOnly)
        
        outDir = Path(tempfile.mkdtemp())
        pl = Pipeline(self._tileDbPath, outDir, debug=True)
        
        clipDs: gdal.Dataset = pl._clip(ulx, 
                                        uly, 
                                        lrx, 
                                        lry, 
                                        ds, 
                                        tileDef.tileWidth, 
                                        tileDef.tileHeight,
                                        ResamplingMethod.AVERAGE)
         
        corners = pl._getCorners(clipDs)
        clipUlx = corners['upperLeft'][0]
        clipUly = corners['upperLeft'][1]
        clipLrx = corners['lowerRight'][0]
        clipLry = corners['lowerRight'][1]

        self.assertTrue(abs(ulx - clipUlx) < 1)
        self.assertTrue(abs(uly - clipUly) < 1)
        self.assertTrue(abs(lrx - clipLrx) < 1)
        self.assertTrue(abs(lry - clipLry) < 1)
        
    # -------------------------------------------------------------------------
    # testCreateCube
    # -------------------------------------------------------------------------
    def testCreateCube(self):

        # imageDir = Path('/explore/nobackup/projects/lfm/' +
        #                 'processed_data/Lunar/LRO_NAC_Pho_Sites')

        tileDbPath = self.getTileDbPath(self._imageDir)
        outDir = Path(tempfile.mkdtemp())
        pl = Pipeline(tileDbPath, outDir, debug=False)

        zone = '42N'
        zoomLevel = 5
        tileX = 1
        tileY = 63
        tileDef: dict = TmsTileDef.initFromParams(zone, zoomLevel)
        ulx, uly, lrx, lry = tileDef.getTileBbox(tileX, tileY)
        
        self.assertEqual(ulx, 167551.38789374547)
        self.assertEqual(uly, 38822.04598557763)
        self.assertEqual(lrx, 206373.43387932325)
        self.assertEqual(lry, 0.0)

        ulLat, ulLon = tileDef.ltmToLatLon(ulx, uly)
        lrLat, lrLon = tileDef.ltmToLatLon(lrx, lry)

        self.assertEqual(ulLat, 1.2801057715312034)
        self.assertEqual(ulLon, 149.27864067025496)
        self.assertEqual(lrLat, 0.0)
        self.assertEqual(lrLon, 150.55999953557063)

        layer: ogr.Layer = pl._query(ulLat, ulLon, lrLat, lrLon)
        self.assertEqual(layer.GetFeatureCount(), 1793)
        
        prodIdDict: dict = pl._createCube(layer, 
                                          ulx, 
                                          uly, 
                                          lrx, 
                                          lry, 
                                          tileDef.tileWidth, 
                                          tileDef.tileHeight,
                                          ResamplingMethod.NEAREST)
        
        self.assertEqual(len(prodIdDict), 340)
        self.assertEqual(len(prodIdDict['M1187363083CE']), 7)
        self.assertEqual(prodIdDict['M1187363083CE'][0][1].shape, (512, 512))
        
    # -------------------------------------------------------------------------
    # testQuery
    # -------------------------------------------------------------------------
    def testQuery(self):

        outDir = Path(tempfile.mkdtemp())
        pl = Pipeline(self._tileDbPath, outDir)

        ulLat = 8.464517596269197e-07 
        ulLon = 119.50911264482448 
        lrLat = -1.2665375155552137 
        lrLon = 120.77987584070043

        layer: ogr.layer = pl._query(ulLat, ulLon, lrLat, lrLon)
        self.assertEqual(layer.GetFeatureCount(), 1267)

    # -------------------------------------------------------------------------
    # testRunTileIndex
    # -------------------------------------------------------------------------
    def testRunTileIndex(self):

        tileDbPath = self.getTileDbPath(self._imageDir)
        outDir = Path(tempfile.mkdtemp())
        pl = Pipeline(tileDbPath, outDir, debug=False)

        tileX = 1
        tileY = 63
        zone = '42N'
        zoomLevel = 5

        outFiles: list[Path] = pl.runTileIndex(tileX, 
                                               tileY, 
                                               zone, 
                                               zoomLevel, 
                                               ResamplingMethod.AVERAGE)
        
        self.assertEqual(len(outFiles), 340)
        
        ds = gdal.Open(outFiles[0], gdalconst.GA_ReadOnly)
        self.assertEqual(ds.RasterXSize, 512)
        self.assertEqual(ds.RasterYSize, 512)
        self.assertEqual(ds.RasterCount, 7)

    # -------------------------------------------------------------------------
    # testRunPoint
    # -------------------------------------------------------------------------
    def testRunPoint(self):

        # imageDir = Path('/explore/nobackup/projects/lfm/' +
        #                 'processed_data/Lunar/LRO_NAC_Pho_Sites')

        tileDbPath = self.getTileDbPath(self._imageDir)
        outDir = Path(tempfile.mkdtemp())
        pl = Pipeline(tileDbPath, outDir, debug=False)

        lon = 149.8
        lat = 1.2
        zone = '42N'
        zoomLevel = 5

        outFiles: list[Path] = pl.runPoint(lat, 
                                           lon, 
                                           zone, 
                                           zoomLevel, 
                                           ResamplingMethod.AVERAGE)
        
        self.assertEqual(len(outFiles), 340)
        
        ds = gdal.Open(outFiles[0], gdalconst.GA_ReadOnly)
        self.assertEqual(ds.RasterXSize, 512)
        self.assertEqual(ds.RasterYSize, 512)
        self.assertEqual(ds.RasterCount, 7)
        
    # -------------------------------------------------------------------------
    # testRun
    # -------------------------------------------------------------------------
    def testRun(self):

        # imageDir = Path('/explore/nobackup/projects/lfm/' +
        #                 'processed_data/Lunar/LRO_NAC_Pho_Sites')

        tileDbPath = self.getTileDbPath(self._imageDir)
        outDir = Path(tempfile.mkdtemp())
        pl = Pipeline(tileDbPath, outDir, debug=False)

        ulLat = 1.3
        ulLon = 149.7
        lrLat = 1.1
        lrLon = 149.9
        zoomLevel = 5

        outFiles: list[Path] = pl.run(ulLat, 
                                      ulLon, 
                                      lrLat, 
                                      lrLon, 
                                      zoomLevel, 
                                      ResamplingMethod.AVERAGE)

        self.assertEqual(len(outFiles), 687)
        
        ds = gdal.Open(outFiles[1], gdalconst.GA_ReadOnly)
        self.assertEqual(ds.RasterXSize, 512)
        self.assertEqual(ds.RasterYSize, 512)
        self.assertEqual(ds.RasterCount, 7)
