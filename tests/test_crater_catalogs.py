import csv
from pathlib import Path

import fiona
import numpy as np
import pytest
import rasterio
from affine import Affine
from pyproj import CRS, Transformer
from shapely.geometry import Polygon, box, mapping

from lfm.labeling.catalogs import import_catalog
from lfm.labeling.craters import CraterLabeler


@pytest.fixture
def lunar_scene(tmp_path):
    # A rotated scene on the far side; negative longitude CSV must wrap to 315E.
    crs=CRS.from_proj4('+proj=eqc +lat_ts=-6 +lon_0=180 +R=1737400 +units=m +type=crs')
    project=Transformer.from_crs(crs.geodetic_crs,crs,always_xy=True,force_over=True)
    x,y=project.transform(315,-6)
    t=Affine(10,2,x-600,1,-10,y+450)
    raster=tmp_path/'moon.tif'
    with rasterio.open(raster,'w',driver='GTiff',width=100,height=100,count=1,dtype='uint8',
                       crs=crs.to_wkt(),transform=t) as ds:
        ds.write(np.ones((100,100),dtype='uint8'),1)
    return raster,crs,t


def test_csv_clipping_wrapping_dedup_deletion_and_provenance(lunar_scene,tmp_path):
    raster,crs,t=lunar_scene
    inverse=Transformer.from_crs(crs,crs.geodetic_crs,always_xy=True,force_over=True)
    catalog=tmp_path/'catalog.csv'
    with catalog.open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['id','latitude','longitude','diameter_km'])
        for id,point in [('inside',(50,50)),('edge',(1,50)),('outside',(500,500))]:
            lon,lat=inverse.transform(*(t*point))
            writer.writerow([id,lat,lon-360,.3])
        writer.writerow(['bad','not-a-number',0,1])
    with rasterio.open(raster) as src:
        records,stats=import_catalog(catalog,src)
        assert len(records)==2
        assert stats['skipped']==1
        assert all(box(0,0,100,100).covers(r['geometry']) for r in records)
        edge=next(r for r in records if r['catalog_id'].startswith('edge:'))
        assert edge['geometry'].bounds[0]==pytest.approx(0,abs=1e-8)
    app=CraterLabeler(tmp_path,tmp_path/'labels',default_raster=raster,default_catalog=catalog)
    try:
        app.load()
        app.import_catalog()
        assert len(app.records)==2
        app.import_catalog()
        assert len(app.records)==2  # repeated import doesn't duplicate
        path=app.last_saved_path
        with fiona.open(path) as ds:
            assert all(f.properties['catalog_source']==str(catalog) for f in ds)
        app.select_record(0)
        app.delete_selected()
        assert len(app.records)==1
        app.undo()
        assert len(app.records)==2
        resumed=CraterLabeler(tmp_path,tmp_path/'labels',default_raster=raster)
        try:
            resumed.load()
            assert all(r['catalog_id'] for r in resumed.records)
        finally:
            resumed.close()
    finally:
        app.close()


def test_vector_catalog_uses_lunar_crs_and_actual_rotated_footprint(lunar_scene,tmp_path):
    raster,crs,t=lunar_scene
    path=tmp_path/'catalog.gpkg'
    native=Polygon([t*p for p in [(-10,40),(40,40),(40,70),(-10,70)]])
    with fiona.open(path,'w',driver='GPKG',crs_wkt=crs.to_wkt(),schema={'geometry':'Polygon','properties':{}}) as dst:
        dst.write({'geometry':mapping(native),'properties':{}})
    with rasterio.open(raster) as src:
        records,stats=import_catalog(path,src)
        assert len(records)==1
        assert records[0]['geometry'].area==pytest.approx(40*30)
    earth=tmp_path/'earth.gpkg'
    with fiona.open(earth,'w',driver='GPKG',crs='EPSG:4326',schema={'geometry':'Polygon','properties':{}}) as dst:
        dst.write({'geometry':mapping(box(0,0,1,1)),'properties':{}})
    with rasterio.open(raster) as src, pytest.raises(ValueError,match='Earth'):
        import_catalog(earth,src)


def test_downloaded_robbins_subset_on_example_raster():
    raster=Path(__file__).resolve().parents[1]/'data/NAC_DTM_NEWCRATER6_M1219245090_80CM.TIF'
    if not raster.exists():
        pytest.skip('Optional local NAC raster is unavailable')
    catalog=Path(__file__).parent/'fixtures/robbins_scene.csv'
    with rasterio.open(raster) as src:
        records,stats=import_catalog(catalog,src)
        assert stats['read']==2 and stats['intersecting']==2
        assert len(records)==2
        assert all(box(0,0,src.width,src.height).covers(r['geometry']) for r in records)
