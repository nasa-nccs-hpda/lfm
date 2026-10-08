import numpy as np
import pytest
import rasterio
import fiona
from affine import Affine
from rasterio.io import MemoryFile
from shapely.geometry import shape
from lfm.data_processing.labeling.craters import ellipse, grow_region, export_gpkg


def test_growth_barriers_and_seed():
    data=np.full((30,30),100,dtype='float32')
    data[10:20,10:20]=5
    data[15,15]=-999
    with MemoryFile() as mem:
        with mem.open(driver='GTiff',width=30,height=30,count=1,dtype='float32',nodata=-999,
                      transform=Affine.translation(100,100)*Affine.scale(2,-2)) as ds:
            ds.write(data,1)
            poly,limited=grow_region(ds,(12.5,12.5),0,20)
            assert poly.bounds==(10,10,20,20)
            assert poly.area==100  # holes intentionally filled for outer crater outlines
            assert not limited
            with pytest.raises(ValueError,match='nodata'):
                grow_region(ds,(15.5,15.5),1,10)
            poly,limited=grow_region(ds,(12.5,12.5),200,3)
            assert limited
            assert poly.area<50


def test_export_affine_and_no_overwrite(tmp_path):
    poly=ellipse((30,50),10,.6,25)
    transform=Affine(2,.2,100,.1,-3,200)
    records=[dict(geometry=poly,seed=(30,50),method='ellipse',band=1)]
    path=export_gpkg(tmp_path/'test.gpkg',records,rasterio.crs.CRS.from_epsg(3857),transform,'test.tif')
    with fiona.open(path) as ds:
        record=next(iter(ds))
        geom=shape(record.geometry)
        assert geom.area==pytest.approx(poly.area*abs(transform.determinant))
        assert tuple(geom.centroid.coords)[0]==pytest.approx(transform*(30,50))
        assert record.properties['method']=='ellipse'
    with pytest.raises(FileExistsError):
        export_gpkg(path,records,rasterio.crs.CRS.from_epsg(3857),transform,'test.tif')


def test_native_map_coordinates_and_preview():
    import base64
    import io
    from PIL import Image
    from lfm.data_processing.labeling.craters import pixel_to_map, map_to_pixel, raster_preview
    transform=Affine(2,.3,4095000,.2,-3,-176500)
    point=(12.25,20.75)
    location=pixel_to_map(transform,point)
    assert map_to_pixel(transform,location)==pytest.approx(point)
    assert location[0]<0 and location[1]>4000000
    with MemoryFile() as mem:
        with mem.open(driver='GTiff',width=60,height=40,count=1,dtype='float32',
                      crs='EPSG:3857',transform=transform,nodata=-999) as ds:
            data=np.arange(2400,dtype='float32').reshape(40,60)
            data[:3,:]=-999
            ds.write(data,1)
            url,bounds,stretch=raster_preview(ds,max_size=32)
            assert bounds==((ds.bounds.bottom,ds.bounds.left),(ds.bounds.top,ds.bounds.right))
            image=Image.open(io.BytesIO(base64.b64decode(url.split(',')[1])))
            assert max(image.size)<=32
            assert image.mode=='RGBA'
            assert stretch[1]>stretch[0]
            assert raster_preview(ds,bounds=((0,0),(1,1))) is None


def test_leaflet_edit_and_spatial_circle(tmp_path):
    from lfm.data_processing.labeling.craters import CraterLabeler,pixel_to_map
    # A nonsquare, rotated grid catches pixel circles incorrectly used as map circles.
    transform=Affine(2,.4,1000,.2,-3,2000)
    with rasterio.open(tmp_path/'raster.tif','w',driver='GTiff',width=100,height=100,
                       count=1,dtype='float32',crs='EPSG:3857',transform=transform) as ds:
        ds.write(np.ones((100,100),dtype='float32'),1)
    app=CraterLabeler(tmp_path,tmp_path/'labels')
    try:
        app.load()
        assert app.map.scroll_wheel_zoom
        assert app.map.crs['name']=='Simple'
        app.mode.value="Ellipse"
        app.radius.value=20
        app._interaction(type='click',coordinates=pixel_to_map(transform,(50,50)))
        center=transform*(50,50)
        vertices=[transform*p for p in app.draft.exterior.coords]
        assert all(np.hypot(x-center[0],y-center[1])==pytest.approx(20) for x,y in vertices)
        app.mode.value='Edit vertices'
        before=app.draft.area
        handle=app.handles[0]
        handle.location=(handle.location[0]+1,handle.location[1]+1)
        assert app.draft.area!=before
        app.accept()
        app.save()
        with fiona.open(app.last_saved_path) as ds:
            assert len(ds)==1
            assert next(iter(ds)).properties['method']=='ellipse+edited'
    finally:
        app.close()


@pytest.mark.parametrize("blank", [False, True])
def test_edge_circle_and_blank_rejection(tmp_path, blank):
    from lfm.data_processing.labeling.craters import edge_circle, CraterLabeler, pixel_to_map
    from scipy.ndimage import gaussian_filter
    yy,xx=np.indices((200,200))
    rr=np.hypot(xx+.5-100, yy+.5-100)
    image=gaussian_filter((rr<42).astype(float),1)
    image += np.random.default_rng(5).normal(0,.01,image.shape)
    if blank:
        image[:]=1
    path=tmp_path/'circle.tif'
    transform=Affine(2,0,1000,0,-2,2000)
    with rasterio.open(path,'w',driver='GTiff',width=200,height=200,count=1,
                      dtype='float32',crs='EPSG:3857',transform=transform) as ds:
        ds.write(image.astype('float32'),1)
    with rasterio.open(path) as ds:
        if blank:
            with pytest.raises(ValueError,match='contrast'):
                edge_circle(ds,(100,100),40,120)
            return
        poly,radius,info=edge_circle(ds,(100,100),40,120)
        assert radius==pytest.approx(84,abs=3)
        assert info['support']>.8
        assert not info['at_limit']
    app=CraterLabeler(tmp_path,tmp_path/'out')
    try:
        app.load()
        assert app.mode.value=='Edge circle'
        app.mode.value='Circle'
        app.ratio.value=.3  # Circle ignores ellipse settings.
        app._interaction(type='click',coordinates=pixel_to_map(transform,(100,100)))
        app.radius.value=35
        assert app.draft.area==pytest.approx(ellipse((100,100),17.5).area)
        assert app.seed==pytest.approx((100,100))
        app.edge_min.value=40
        app.edge_max.value=120
        app.fit_edges()
        assert app.mode.value=='Edge circle'
        assert app.radius.value==pytest.approx(84,abs=3)
        app.radius.value=90
        assert app.draft_method=='edge_circle+adjusted'
        assert app.draft.area==pytest.approx(ellipse((100,100),45).area)
    finally:
        app.close()


def test_autosave_slider_delete_failure_and_resume(tmp_path, monkeypatch):
    import lfm.data_processing.labeling.craters as module
    from lfm.data_processing.labeling.craters import CraterLabeler, pixel_to_map
    transform=Affine(1,0,1000,0,-1,2000)
    raster=tmp_path/'raster.tif'
    with rasterio.open(raster,'w',driver='GTiff',width=200,height=200,
                       count=1,dtype='float32',crs='EPSG:3857',transform=transform) as ds:
        ds.write(np.ones((200,200),dtype='float32'),1)
    app=CraterLabeler(tmp_path,tmp_path/'labels')
    try:
        app.load()
        app.mode.value='Circle'
        app._interaction(type='click',coordinates=pixel_to_map(transform,(60,60)))
        first=app.last_saved_path
        assert first.exists()
        assert len(app.records)==1  # no Accept/Add button required
        app.radius_slider.value=35
        assert app.radius.value==35
        app.radius.value=40
        assert app.radius_slider.value==40
        with fiona.open(first) as ds:
            assert len(ds)==1
            assert shape(next(iter(ds)).geometry).area==pytest.approx(ellipse((0,0),40).area)
        app._interaction(type='click',coordinates=pixel_to_map(transform,(130,130)))
        assert len(app.records)==2
        before=first.read_bytes()
        def fail(*args,**kwargs):
            raise OSError('disk full')
        with monkeypatch.context() as patch:
            patch.setattr(module.os,'replace',fail)
            app.radius.value=42
            assert 'Autosave failed' in app.status.value
            assert app._draft_dirty
            assert first.read_bytes()==before
        app.save()
        assert len(app.records)==2
        assert not app._draft_dirty
        app.mode.value='Select/delete'
        app._interaction(type='click',coordinates=pixel_to_map(transform,(60,60)))
        assert app._active_index==0
        app.delete_selected()
        with fiona.open(first) as ds:
            assert len(ds)==1
        app.undo()
        assert len(app.records)==2
        resumed=CraterLabeler(tmp_path,tmp_path/'labels')
        try:
            resumed.load()
            assert len(resumed.records)==2
            resumed.select_record(0)
            resumed.delete_selected()
            resumed.select_record(0)
            resumed.delete_selected()
            with fiona.open(first) as ds:
                assert len(ds)==0
        finally:
            resumed.close()
        assert len(list((tmp_path/'labels').glob('*.gpkg')))==1
    finally:
        app.close()


def test_delete_all_confirmation_undo_and_failure(tmp_path, monkeypatch):
    import lfm.data_processing.labeling.craters as module
    transform=Affine(1,0,1000,0,-1,2000)
    raster=tmp_path/'raster.tif'
    with rasterio.open(raster,'w',driver='GTiff',width=200,height=200,
                       count=1,dtype='float32',crs='EPSG:3857',transform=transform) as ds:
        ds.write(np.ones((200,200),dtype='float32'),1)
    app=module.CraterLabeler(tmp_path,tmp_path/'labels')
    try:
        assert app.delete_all_button.disabled
        assert app.delete_all_button.button_style=='danger'
        app.load()
        assert app.delete_all_button.disabled
        app.mode.value='Circle'
        for center in [(60,60),(130,130)]:
            app._interaction(type='click',coordinates=module.pixel_to_map(transform,center))
        path=app.last_saved_path
        before=path.read_bytes()
        history=list(app._history)
        app.delete_all_button.click()
        assert app.delete_all_dialog.layout.display=='flex'
        assert '2 saved craters' in app.delete_all_message.value
        app.cancel_delete_all_button.click()
        app.confirm_delete_all_button.click()  # a cancelled confirmation is inert
        assert path.read_bytes()==before
        assert len(app.records)==2 and app._history==history

        def fail(*args,**kwargs):
            raise OSError('disk full')
        with monkeypatch.context() as patch:
            patch.setattr(module.os,'replace',fail)
            app.radius.value=42  # retain an unsaved edit on write failure
            draft=app.draft
            assert app._draft_dirty
            app.delete_all_button.click()
            app.confirm_delete_all_button.click()
            assert 'disk full' in app.status.value
            assert app.draft is draft and app._draft_dirty
            assert len(app.records)==2 and app._history==history
            assert path.read_bytes()==before

        app.delete_all_button.click()
        app.confirm_delete_all_button.click()
        assert not app.records and app.draft is None and not app._draft_dirty
        assert not app.accepted.layers and not app.handles
        assert app.selection.value is None and app.delete_all_button.disabled
        with fiona.open(path) as ds:
            assert len(ds)==0
        resumed=module.CraterLabeler(tmp_path,tmp_path/'labels')
        try:
            resumed.load()
            assert not resumed.records
        finally:
            resumed.close()
        app.undo()
        assert len(app.records)==2 and not app.delete_all_button.disabled
        with fiona.open(path) as ds:
            assert len(ds)==2

        app.delete_all_button.click()
        app.load()  # reopening/changing rasters invalidates pending confirmation
        app.confirm_delete_all_button.click()
        assert len(app.records)==2
        app.delete_all_button.click()
        app.select_record(0)
        app.radius.value=35  # editing also invalidates confirmation
        app.confirm_delete_all_button.click()
        assert len(app.records)==2
        assert app.delete_all_dialog.layout.display=='none'
    finally:
        app.close()


def test_filechooser_default_and_select_callback(tmp_path):
    from ipyfilechooser import FileChooser
    from types import SimpleNamespace
    from lfm.data_processing.labeling.craters import CraterLabeler
    default=tmp_path/'default.TIF'
    other=tmp_path/'other.tiff'
    for p in (default,other):
        with rasterio.open(p,'w',driver='GTiff',width=10,height=10,count=1,
                           dtype='uint8',crs='EPSG:3857',transform=Affine(1,0,100,0,-1,100)) as ds:
            ds.write(np.ones((10,10),dtype='uint8'),1)
    app=CraterLabeler(tmp_path,tmp_path/'labels',default_raster=default)
    try:
        assert app.mode.value=='Edge circle'
        assert isinstance(app.raster_chooser,FileChooser)
        assert 'Raster image' in app.raster_chooser.title
        assert 'crater vectors' in app.catalog_chooser.title
        assert app.catalog_panel.selected_index is None
        assert app.raster_chooser.selected==str(default)
        app.raster_chooser._show_dialog()
        app.raster_chooser._set_form_values(str(tmp_path),other.name)
        app._load_raster_selection()
        assert app.src.name==str(other)
        assert app.output.value.endswith('other_label_craters.gpkg')
    finally:
        app.close()


def test_click_existing_crater_edits_in_place(tmp_path, monkeypatch):
    from lfm.data_processing.labeling.craters import CraterLabeler, pixel_to_map
    transform=Affine(1,0,1000,0,-1,2000)
    raster=tmp_path/'raster.tif'
    with rasterio.open(raster,'w',driver='GTiff',width=200,height=200,count=1,
                       dtype='uint8',crs='EPSG:3857',transform=transform) as ds:
        ds.write(np.ones((200,200),dtype='uint8'),1)
    app=CraterLabeler(tmp_path,tmp_path/'labels',default_raster=raster)
    try:
        app.load()
        app.mode.value='Circle'
        first=pixel_to_map(transform,(60,60))
        app._interaction(type='click',coordinates=first)
        original=app.records[0]['geometry']
        app._interaction(type='click',coordinates=pixel_to_map(transform,(130,130)))
        app._interaction(type='click',coordinates=first)
        assert len(app.records)==2
        assert app._active_index==0
        assert app.records[0]['geometry'].equals(original)
        app.radius_slider.value=35
        assert len(app.records)==2
        assert app.records[0]['geometry'].area==pytest.approx(ellipse((0,0),35).area)
        import lfm.data_processing.labeling.craters as module
        monkeypatch.setattr(module,'edge_circle',lambda *args: (
            ellipse((60,60),34),34,{'support':1,'at_limit':False}))
        # Real double-click sequences include two click events before dblclick.
        for event in ('click','click','dblclick'):
            app._interaction(type=event,coordinates=first)
        assert len(app.records)==2
        assert app.mode.value=='Edge circle'
        assert not app.handles
        assert app.draft_method=='edge_circle'
        app.mode.value='Edit vertices'
        assert app.handles
        app._vertex_changed(0,pixel_to_map(transform,(94,60)))
        with fiona.open(app.last_saved_path) as ds:
            assert len(ds)==2
        app.mode.value='Navigate'
        app._interaction(type='click',coordinates=first)
        assert len(app.records)==2
    finally:
        app.close()


def test_zoom_preview_reads_source_detail(tmp_path):
    import base64
    import io
    from PIL import Image
    from lfm.data_processing.labeling.craters import raster_preview
    raster=tmp_path/'detail.tif'
    pixels=(np.indices((1600,1600)).sum(axis=0)%2*255).astype('uint8')
    with rasterio.open(raster,'w',driver='GTiff',width=1600,height=1600,count=1,
                       dtype='uint8',crs='EPSG:3857',transform=Affine(1,0,0,0,-1,1600)) as ds:
        ds.write(pixels,1)
    with rasterio.open(raster) as src:
        overview,_,_=raster_preview(src,stretch=(0,255))
        detail,bounds,_=raster_preview(src,bounds=((1400,100),(1500,200)),stretch=(0,255),max_size=2800)
    def decode(url):
        return np.array(Image.open(io.BytesIO(base64.b64decode(url.split(',',1)[1]))))
    assert decode(overview).shape[:2]==(1400,1400)
    assert bounds==((1400,100),(1500,200))
    np.testing.assert_array_equal(decode(detail)[:,:,0],pixels[100:200,100:200])


def test_resume_develop_labels_without_catalog_fields(tmp_path):
    from lfm.data_processing.labeling.craters import load_labels
    raster = tmp_path / 'scene.tif'
    transform = Affine(1,0,1000,0,-1,2000)
    with rasterio.open(raster,'w',driver='GTiff',width=100,height=100,count=1,
                       dtype='uint8',crs='EPSG:3857',transform=transform) as dst:
        dst.write(np.ones((100,100),dtype='uint8'),1)
    label = tmp_path / 'scene_label_craters.gpkg'
    # Exact pre-integration schema: optional catalog provenance did not exist.
    schema = {'geometry':'Polygon', 'properties':{'crater_id':'int','method':'str',
        'source':'str','band':'int','seed_col':'float','seed_row':'float','area_native':'float'}}
    from shapely.affinity import affine_transform
    from shapely.geometry import mapping
    pixel = ellipse((50,50),10)
    native = affine_transform(pixel,[1,0,0,-1,1000,2000])
    with rasterio.open(raster) as src:
        with fiona.open(label,'w',driver='GPKG',layer='craters',crs_wkt=src.crs.to_wkt(),schema=schema) as dst:
            dst.write({'geometry':mapping(native),'properties':{'crater_id':1,'method':'circle',
                'source':str(raster),'band':1,'seed_col':50.,'seed_row':50.,'area_native':native.area}})
        records = load_labels(label, src)
    assert len(records) == 1
    assert records[0]['catalog_id'] == records[0]['catalog_source'] == ''
    assert records[0]['geometry'].equals_exact(pixel,1e-6)
