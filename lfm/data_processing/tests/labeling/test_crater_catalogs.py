import csv
from pathlib import Path
from lfm.data_processing._paths import REPO_ROOT

import fiona
import numpy as np
import pytest
import rasterio
from affine import Affine
from pyproj import CRS, Transformer
from shapely.geometry import Polygon, box, mapping, shape

from lfm.data_processing.labeling.catalogs import import_catalog
from lfm.data_processing.labeling.craters import CraterLabeler


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
    raster=REPO_ROOT/'data/NAC_DTM_NEWCRATER6_M1219245090_80CM.TIF'
    if not raster.exists():
        pytest.skip('Optional local NAC raster is unavailable')
    catalog=Path(__file__).parent/'fixtures/robbins_scene.csv'
    with rasterio.open(raster) as src:
        records,stats=import_catalog(catalog,src)
        assert stats['read']==2 and stats['intersecting']==2
        assert len(records)==2
        assert all(box(0,0,src.width,src.height).covers(r['geometry']) for r in records)


def test_catalog_export_preserves_chip_contract(lunar_scene, tmp_path):
    from lfm.data_processing.labeling.craters import export_gpkg, load_labels
    from lfm.data_processing.chip.chip_notebook_utils import latest_crater_label_path
    raster, crs, transform = lunar_scene
    catalog = tmp_path / 'catalog.gpkg'
    poly = Polygon([transform*p for p in [(10,10),(30,10),(30,30),(10,30)]])
    with fiona.open(catalog, 'w', driver='GPKG', crs_wkt=crs.to_wkt(),
                    schema={'geometry':'Polygon','properties':{'catalog_id':'str'}}) as dst:
        dst.write({'geometry':mapping(poly),'properties':{'catalog_id':'example'}})
    label_dir = tmp_path / 'notebooks/outputs/labels'
    labels = label_dir / (raster.stem + '_label_craters.gpkg')
    with rasterio.open(raster) as src:
        records, _ = import_catalog(catalog, src)
        export_gpkg(labels, records, src.crs, src.transform, src.name)
        resumed = load_labels(labels, src)
    assert latest_crater_label_path(label_dir) == labels
    assert resumed[0]['catalog_id'] == records[0]['catalog_id']
    with fiona.open(labels, layer='craters') as collection:
        assert CRS.from_wkt(collection.crs_wkt) == crs
        feature = next(iter(collection))
        assert feature['properties']['crater_id'] == 1
        exported = shape(feature['geometry'])
        assert exported.hausdorff_distance(poly) < 1e-6
        assert exported.area == pytest.approx(poly.area)


def test_catalog_export_converts_to_chip_instances(lunar_scene, tmp_path):
    pytest.importorskip('osgeo', reason='Run in the HPC container for GDAL chip conversion')
    from lfm.data_processing.labeling.craters import export_gpkg, ellipse
    from lfm.data_processing.chip.chip_instance_labels import convert_crater_labels
    from lfm.data_processing.chip.chip_requests import raster_bounds
    from lfm.data_processing.chip.chip_types import TargetGrid
    raster, crs, transform = lunar_scene
    label = tmp_path / 'catalog_label_craters.gpkg'
    with rasterio.open(raster) as src:
        record = dict(geometry=ellipse((50,50),10), seed=(50,50), method='catalog',
                      band=1, catalog_id='example:0:0', catalog_source='catalog.csv')
        export_gpkg(label, [record], src.crs, src.transform, src.name)
        target = TargetGrid(src.crs.to_wkt(), transform.to_gdal(),
                            raster_bounds(transform.to_gdal(), src.width, src.height),
                            src.width, src.height)
    result = convert_crater_labels(label, target_grid=target)
    assert result.num_craters == 1
    assert result.mask.shape == (100, 100)
    assert set(np.unique(result.mask)) == {0, 1}
    assert result.bboxes.shape == (1, 4)
