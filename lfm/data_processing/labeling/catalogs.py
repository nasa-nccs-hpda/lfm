"""Import lunar crater catalogs, projecting and clipping to a raster footprint."""
import csv
import math
from pathlib import Path

import fiona
import numpy as np
from pyproj import CRS, Geod, Transformer
from shapely.affinity import affine_transform
from shapely.geometry import Polygon, shape
from shapely.ops import transform as transform_geometry


def _parts(geometry):
    if geometry.geom_type == 'Polygon':
        yield geometry
    elif hasattr(geometry, 'geoms'):
        for part in geometry.geoms:
            yield from _parts(part)


def import_catalog(path, src, band=1):
    """Read Robbins/generic lunar CSV or CRS-tagged polygon GPKG/SHP.

    CSV: CRATER_ID,LAT_CIRC_IMG,LON_CIRC_IMG,DIAM_CIRC_IMG or
    id,latitude,longitude,diameter_km. Angles are planetocentric, east-positive;
    diameter is km on a 1737.4 km lunar sphere. Vector CRS must be lunar.
    Returns clipped pixel-space records and import diagnostics.
    """
    path = Path(path).expanduser().resolve()
    target = CRS.from_user_input(src.crs)
    if not 1.6e6 < target.ellipsoid.semi_major_metre < 1.9e6:
        raise ValueError('Catalog import requires a lunar raster CRS, not an Earth CRS.')
    moon = CRS.from_proj4('+proj=longlat +R=1737400 +no_defs +type=crs')
    to_map = Transformer.from_crs(moon, target, always_xy=True, force_over=True)
    to_moon = Transformer.from_crs(target, moon, always_xy=True, force_over=True)
    geod = Geod(a=1737400, b=1737400)
    t = src.transform
    footprint = Polygon([t*p for p in [(0,0),(src.width,0),(src.width,src.height),(0,src.height)]])
    center = footprint.centroid
    lon0, lat0 = to_moon.transform(center.x, center.y)
    corners = [to_moon.transform(x,y) for x,y in footprint.exterior.coords]
    scene_radius = max(geod.inv(lon0,lat0,lon,lat)[2] for lon,lat in corners)
    inverse = ~t
    records = []
    stats = dict(read=0, skipped=0, intersecting=0)
    provenance = str(path)

    def add(native, identifier):
        if native.is_empty or not native.is_valid:
            stats['skipped'] += 1
            return
        clipped = native.intersection(footprint)
        pieces = [p for p in _parts(clipped) if p.area > 0]
        if pieces:
            stats['intersecting'] += 1
        for i, part in enumerate(pieces):
            pixel = affine_transform(part,[inverse.a,inverse.b,inverse.d,inverse.e,inverse.c,inverse.f])
            # Roundoff along the scene boundary should not push coordinates outside.
            from shapely.geometry import box
            pixel = pixel.intersection(box(0,0,src.width,src.height))
            for j, poly in enumerate(_parts(pixel)):
                if poly.area <= 0:
                    continue
                seed = poly.representative_point()
                records.append(dict(geometry=poly,seed=(seed.x,seed.y),method='catalog',band=band,
                                    catalog_id=f'{identifier}:{i}:{j}',catalog_source=provenance))

    if path.suffix.lower() == '.csv':
        with path.open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            columns = set(reader.fieldnames or [])
            if {'LAT_CIRC_IMG','LON_CIRC_IMG','DIAM_CIRC_IMG'} <= columns:
                lat_key, lon_key, size_key, id_key = 'LAT_CIRC_IMG','LON_CIRC_IMG','DIAM_CIRC_IMG','CRATER_ID'
            elif {'latitude','longitude','diameter_km'} <= columns:
                lat_key, lon_key, size_key, id_key = 'latitude','longitude','diameter_km','id'
            else:
                raise ValueError('CSV needs Robbins columns or latitude, longitude, diameter_km (optional id).')
            bearings = np.linspace(0,360,128,endpoint=False)
            for row in reader:
                stats['read'] += 1
                try:
                    lat,lon,diameter = (float(row[key]) for key in (lat_key,lon_key,size_key))
                    if not all(map(math.isfinite,(lat,lon,diameter))) or not -90<=lat<=90 or not 0<diameter<10900:
                        raise ValueError()
                except (ValueError,TypeError):
                    stats['skipped'] += 1
                    continue
                radius = diameter*500
                if geod.inv(lon0,lat0,lon,lat)[2] > radius + scene_radius*1.05:
                    continue
                lons,lats,_ = geod.fwd(np.full(128,lon),np.full(128,lat),bearings,np.full(128,radius))
                lons = lon0 + (np.asarray(lons)-lon0+180)%360-180
                x,y = to_map.transform(lons,lats)
                if not np.isfinite([x,y]).all():
                    stats['skipped'] += 1
                    continue
                add(Polygon(zip(x,y)),row.get(id_key) or str(stats['read']))
    elif path.suffix.lower() in ('.gpkg','.shp','.geojson','.json'):
        with fiona.open(path) as collection:
            if not collection.crs_wkt:
                raise ValueError('Vector catalog must declare its lunar CRS.')
            source = CRS.from_wkt(collection.crs_wkt)
            if not 1.6e6 < source.ellipsoid.semi_major_metre < 1.9e6:
                raise ValueError('Vector catalog declares an Earth/unknown CRS. Assign its actual lunar CRS first.')
            project = Transformer.from_crs(source,target,always_xy=True,force_over=True)
            def project_xy(x,y,z=None):
                if source.is_geographic:
                    x=lon0+(np.asarray(x)-lon0+180)%360-180
                return project.transform(x,y)
            for feature in collection:
                stats['read'] += 1
                if not feature['geometry']:
                    stats['skipped'] += 1
                    continue
                native = transform_geometry(project_xy,shape(feature['geometry']))
                if native.geom_type not in ('Polygon','MultiPolygon'):
                    stats['skipped'] += 1
                    continue
                props = feature['properties']
                add(native,props.get('catalog_id') or props.get('CRATER_ID') or feature['id'])
    else:
        raise ValueError('Choose a lunar CSV, GeoPackage, shapefile or CRS-tagged GeoJSON catalog.')
    return records,stats
