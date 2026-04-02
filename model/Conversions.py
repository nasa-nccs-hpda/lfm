from osgeo import osr


# ----------------------------------------------------------------------------
# Class Conversions
# ----------------------------------------------------------------------------
class Conversions:
    
    LUNAR_LAT_LON_PROJ4 = '+proj=longlat +R=1737400 +no_defs'
    
    LUNAR_LAT_LON_WKT = 'GEOGCRS["Moon (2015) - Sphere / Ocentric", ' + \
        'DATUM["Moon (2015) - Sphere", ELLIPSOID["Moon (2015) -' + \
        ' Sphere",1737400,0, LENGTHUNIT["metre",1]]], PRIMEM["Reference' + \
        ' Meridian",0, ANGLEUNIT["degree",0.0174532925199433]],' + \
        ' CS[ellipsoidal,2], AXIS["geodetic latitude (Lat)",north,' + \
        ' ORDER[1], ANGLEUNIT["degree",0.0174532925199433]],' + \
        ' AXIS["geodetic longitude (Lon)",east, ORDER[2],' + \
        ' ANGLEUNIT["degree",0.0174532925199433]], ID["IAU",30100,2015],' + \
        ' REMARK["Source of IAU Coordinate systems:' + \
        ' https://doi.org/10.1007/s10569-017-9805-5"]]'
    
    # ------------------------------------------------------------------------
    # latLonToLTM
    # ------------------------------------------------------------------------
    @staticmethod
    def latLonToLTM(lat: float, 
                    lon: float, 
                    outSRS: osr.SpatialReference) -> tuple[float, float]:
    
        # ---
        # Lat/lon SRS
        # ---
        latLonSRS = osr.SpatialReference()
        latLonSRS.ImportFromProj4(Conversions.LUNAR_LAT_LON_PROJ4)
    
        # Transform the coordinates
        # TransformPoint expects (x, y) = (lon, lat) for geographic coords
        xform = osr.CoordinateTransformation(latLonSRS, outSRS)
        easting, northing, height = xform.TransformPoint(lon, lat)

        return easting, northing
    
    # ------------------------------------------------------------------------
    # ltmToLatLon
    # ------------------------------------------------------------------------
    @staticmethod
    def ltmToLatLon(zone: str, 
                    ll: tuple, 
                    ur: tuple, 
                    inSRS: osr.SpatialReference) -> tuple[float, 
                                                          float, 
                                                          float, 
                                                          float]:
    
        # ---
        # Lat/lon SRS
        # ---
        latLonSRS = osr.SpatialReference()
        latLonSRS.ImportFromProj4(Conversions.LUNAR_LAT_LON_PROJ4)
    
        # ---
        # Transform the coordinates
        # TransformPoint returns (lon, lat, height) for geographic coords
        # ---
        xform = osr.CoordinateTransformation(inSRS, latLonSRS)
        llLon, llLat, _ = xform.TransformPoint(ll[0], ll[1])
        urLon, urLat, _ = xform.TransformPoint(ur[0], ur[1])

        return llLat, llLon, urLat, urLon