
from osgeo import osr


# ----------------------------------------------------------------------------
# Class Conversions
# ----------------------------------------------------------------------------
class Conversions():
    
    LUNAR_LAT_LON_PROJ4 = '+proj=longlat +R=1737400 +no_defs'
    
    # ----------------------------------------------------------------------------
    # latLonToLTM
    # ----------------------------------------------------------------------------
    @staticmethod
    def latLonToLTM(lat: float, lon: float, tileDef: dict) -> tuple[float, float]:
    
        # ---
        # Lat/lon SRS
        # ---
        latLonSRS = osr.SpatialReference()
        latLonSRS.ImportFromProj4(Conversions.LUNAR_LAT_LON_PROJ4)
    
        # ---
        # LTM SRS
        # ---
        ltmSRS = osr.SpatialReference()
        ltmSRS.ImportFromWkt(tileDef['crs'])
    
        # Transform the coordinates.
        xform = osr.CoordinateTransformation(latLonSRS, ltmSRS)
        test = xform.TransformPoint(lon, lat)
        easting, northing, height = xform.TransformPoint(lon, lat)

        return easting, northing
    
    # ----------------------------------------------------------------------------
    # ltmToLatLon
    # ----------------------------------------------------------------------------
    @staticmethod
    def ltmToLatLon(zone: str, 
                    ll: tuple, 
                    ur: tuple, 
                    tileDef: dict) -> tuple[float, float]:
    
        # ---
        # Lat/lon SRS
        # ---
        latLonSRS = osr.SpatialReference()
        latLonSRS.ImportFromProj4(Conversions.LUNAR_LAT_LON_PROJ4)
    
        # ---
        # LTM SRS
        # ---
        ltmSRS = osr.SpatialReference()
        ltmSRS.ImportFromWkt(tileDef['crs'])
    
        # Transform the coordinates.
        xform = osr.CoordinateTransformation(ltmSRS, latLonSRS)
        llLatLon = xform.TransformPoint(ll[0], ll[1])
        urLatLon = xform.TransformPoint(ur[0], ur[1])

        return llLatLon, urLatLon
    
    