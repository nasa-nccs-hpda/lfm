import numpy as np
import pytest
import rasterio
import fiona
from affine import Affine
from rasterio.io import MemoryFile
from shapely.geometry import shape
from lfm.labeling.craters import ellipse, grow_region, export_gpkg


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
    from lfm.labeling.craters import pixel_to_map, map_to_pixel, raster_preview
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
    from lfm.labeling.craters import CraterLabeler,pixel_to_map
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
    from lfm.labeling.craters import edge_circle, CraterLabeler, pixel_to_map
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


def test_persistent_autosave_and_failed_accept(tmp_path, monkeypatch):
    import lfm.labeling.craters as module
    from lfm.labeling.craters import CraterLabeler, pixel_to_map
    transform=Affine(1,0,1000,0,-1,2000)
    with rasterio.open(tmp_path/'raster.tif','w',driver='GTiff',width=200,height=200,
                       count=1,dtype='float32',crs='EPSG:3857',transform=transform) as ds:
        ds.write(np.ones((200,200),dtype='float32'),1)
    app=CraterLabeler(tmp_path,tmp_path/'labels')
    try:
        app.load()
        app.mode.value='Circle'
        base=app.output.value
        app._interaction(type='click',coordinates=pixel_to_map(transform,(60,60)))
        app.accept()
        first=app.last_saved_path
        assert first.exists() and first.name=='raster_label_craters.gpkg'
        assert app.output.value==base
        with fiona.open(first) as ds:
            assert len(ds)==1
        app._interaction(type='click',coordinates=pixel_to_map(transform,(130,130)))
        app.accept()
        second=app.last_saved_path
        assert second==first
        with fiona.open(second) as ds:
            assert len(ds)==2
        with fiona.open(first) as ds:
            assert len(ds)==2
        app.save()
        assert app.last_saved_path==first
        assert app.output.value==base
        app._interaction(type='click',coordinates=pixel_to_map(transform,(100,100)))
        draft=app.draft
        last_saved=app.last_saved_path
        def fail(*args,**kwargs):
            raise OSError('disk full')
        with monkeypatch.context() as patch:
            patch.setattr(module,'export_gpkg',fail)
            with pytest.raises(RuntimeError,match='draft is retained'):
                app.accept()
        assert len(app.records)==2
        assert app.draft is draft
        assert app.last_saved_path==last_saved
        with fiona.open(last_saved) as ds:
            assert len(ds)==2
        app.accept()
        assert len(app.records)==3
        with fiona.open(app.last_saved_path) as ds:
            assert len(ds)==3
        resumed=CraterLabeler(tmp_path,tmp_path/'labels')
        try:
            resumed.load()
            assert len(resumed.records)==3
            assert resumed.output.value==base
            resumed.undo()
            with fiona.open(resumed.last_saved_path) as ds:
                assert len(ds)==2
        finally:
            resumed.close()
        assert len(list((tmp_path/'labels').glob('*.gpkg')))==1
    finally:
        app.close()


def test_raster_dropdown_and_default_path(tmp_path):
    from lfm.labeling.craters import CraterLabeler
    default=tmp_path/'default.TIF'
    default.touch()
    (tmp_path/'other.tiff').touch()
    app=CraterLabeler(tmp_path,tmp_path/'labels',default_raster=default)
    try:
        assert app.mode.value=='Edge circle'
        assert app.files.value==str(default)
        assert len(app.files.options)==2
        app.files.value=str(tmp_path/'other.tiff')
        assert app.path.value==str(tmp_path/'other.tiff')
        app.refresh_rasters()
        assert app.files.value==str(tmp_path/'other.tiff')
    finally:
        app.close()


def test_file_browser_navigation_selection_and_cancel(tmp_path):
    from lfm.labeling.craters import CraterLabeler
    default=tmp_path/'default.TIF'
    default.touch()
    nested=tmp_path/'another_folder'
    nested.mkdir()
    target=nested/'crater.TIFF'
    target.touch()
    (nested/'notes.txt').touch()
    app=CraterLabeler(tmp_path,tmp_path/'labels',default_raster=default)
    try:
        app.widget.children[0].click()  # Browse files button
        assert app.browser_panel.layout.display==''
        assert app.folder.value==str(tmp_path)
        app.browser_entries.value=str(nested)
        app._browser_open()
        assert app.folder.value==str(nested)
        assert [v for _,v in app.browser_entries.options]==[str(target)]
        app.browser_filter.value='missing'
        assert len(app.browser_entries.options)==0
        app.browser_filter.value='CRATER'
        app.browser_entries.value=str(target)
        app._browser_open()
        assert app.path.value==str(target)
        assert app.browser_panel.layout.display=='none'
        app._browse_files()
        app._browser_up()
        assert app.folder.value==str(tmp_path)
        app._browser_cancel()
        assert app.path.value==str(target)
        app.folder.value=str(tmp_path/'unavailable')
        app._list_browser()
        assert 'Cannot open this folder' in app.browser_message.value
        app._browser_default()
        assert app.folder.value==str(tmp_path)
    finally:
        app.close()
