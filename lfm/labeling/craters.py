"""Pixel-space annotation with native-CRS GeoPackage export.

Raster reads are bounded; the full-resolution source is never loaded in memory.
"""
from collections import deque
from pathlib import Path
import math
import os
import tempfile

import fiona
import numpy as np
import rasterio
from affine import Affine
from rasterio.features import shapes
from rasterio.windows import Window
from shapely.affinity import affine_transform, scale
from shapely.geometry import Polygon, mapping, shape


def ellipse(center, radius, ratio=1.0, angle=0.0, vertices=96):
    """Return an ellipse in source pixel coordinates (radius is semi-major axis)."""
    if radius <= 0 or ratio <= 0:
        raise ValueError('Radius and axis ratio must be positive.')
    t = np.linspace(0, 2 * np.pi, vertices, endpoint=False)
    a = np.deg2rad(angle)
    x, y = radius * np.cos(t), radius * ratio * np.sin(t)
    return Polygon(np.column_stack((center[0] + x*np.cos(a) - y*np.sin(a),
                                    center[1] + x*np.sin(a) + y*np.cos(a))))


def grow_region(src, center, tolerance, radius, band=1):
    """4-connected seed-intensity flood, bounded by a native-pixel radius.

    Tolerance is in original band units. Nodata and nonfinite pixels are barriers.
    Returns (pixel polygon, reached distance/window limit).
    """
    if not 1 <= radius <= 512 or tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError('Growth radius must be 1–512 pixels; tolerance must be finite and nonnegative.')
    col, row = int(math.floor(center[0])), int(math.floor(center[1]))
    if not (0 <= col < src.width and 0 <= row < src.height):
        raise ValueError('Seed is outside the raster.')
    r = int(math.ceil(radius))
    x0, y0 = max(0, col-r), max(0, row-r)
    x1, y1 = min(src.width, col+r+1), min(src.height, row+r+1)
    data = src.read(band, window=Window(x0, y0, x1-x0, y1-y0), masked=True)
    valid = ~np.ma.getmaskarray(data) & np.isfinite(data.data)
    sy, sx = row-y0, col-x0
    if not valid[sy, sx]:
        raise ValueError('Seed falls on nodata; choose a valid pixel.')
    yy, xx = np.indices(data.shape)
    distance = np.hypot(xx-sx, yy-sy)
    allowed = valid & (distance <= radius) & (np.abs(data.data.astype(float)-float(data[sy, sx])) <= tolerance)
    connected = np.zeros(data.shape, dtype=np.uint8)
    connected[sy, sx] = 1
    queue = deque([(sy, sx)])
    while queue:
        y, x = queue.popleft()
        for ny, nx in ((y-1,x), (y+1,x), (y,x-1), (y,x+1)):
            if 0 <= ny < data.shape[0] and 0 <= nx < data.shape[1] and allowed[ny,nx] and not connected[ny,nx]:
                connected[ny,nx] = 1
                queue.append((ny,nx))
    geom = next(shape(g) for g, value in shapes(connected, mask=connected.astype(bool),
                transform=Affine.translation(x0,y0), connectivity=4) if value == 1)
    # Crater labels describe the outer rim, so fill intensity islands/holes.
    poly = Polygon(geom.exterior).simplify(0.75, preserve_topology=True)
    limited = bool(np.any(connected.astype(bool) & (distance >= max(0, radius-1))))
    limited |= bool(connected[0,:].any() or connected[-1,:].any() or connected[:,0].any() or connected[:,-1].any())
    return poly, limited


def edge_circle(src, center, min_radius, max_radius, band=1):
    """Fit a fixed-center circle to radial edges in a bounded source window.

    Radii use native CRS units. Scores combine angular sectors so one bright
    arc cannot dominate. This is an edge proposal, not a rim classifier.
    """
    from scipy.ndimage import gaussian_filter, map_coordinates, minimum_filter
    if not (np.isfinite(min_radius) and np.isfinite(max_radius) and 0 < min_radius < max_radius):
        raise ValueError('Edge search requires 0 < minimum radius < maximum radius.')
    col,row=center
    if not (0 <= col < src.width and 0 <= row < src.height):
        raise ValueError('Seed is outside the raster.')
    inverse=~src.transform
    dx=max_radius*math.hypot(inverse.a,inverse.b)+4
    dy=max_radius*math.hypot(inverse.d,inverse.e)+4
    x0,y0=max(0,math.floor(col-dx)),max(0,math.floor(row-dy))
    x1,y1=min(src.width,math.ceil(col+dx)),min(src.height,math.ceil(row+dy))
    width,height=x1-x0,y1-y0
    scale=max(width/1024,height/1024,1)
    ow,oh=max(1,math.ceil(width/scale)),max(1,math.ceil(height/scale))
    data=src.read(band,window=Window(x0,y0,width,height),out_shape=(oh,ow),masked=True)
    valid=~np.ma.getmaskarray(data) & np.isfinite(data.data)
    if not valid.any():
        raise ValueError('No valid pixels in the edge search window.')
    values=data.data[valid].astype(float)
    lo,hi=np.percentile(values,[2,98])
    if hi-lo<=max(abs(lo),1)*1e-10:
        raise ValueError('No usable contrast for an edge fit. Use Circle mode.')
    normalized=np.where(valid,np.clip((data.data-lo)/(hi-lo),0,1),0)
    smoothed=gaussian_filter(normalized,1.2)
    safe=minimum_filter(valid.astype('uint8'),size=9,mode='constant',cval=0)>0
    sample_transform=src.transform*Affine.translation(x0,y0)*Affine.scale(width/ow,height/oh)
    inv=~sample_transform
    pixel_size=min(np.linalg.svd(np.array([[sample_transform.a,sample_transform.b],
                                         [sample_transform.d,sample_transform.e]]),compute_uv=False))
    radii=np.linspace(min_radius,max_radius,min(512,max(3,math.ceil((max_radius-min_radius)/pixel_size)+1)))
    angles=np.linspace(0,2*np.pi,180,endpoint=False)
    cx,cy=src.transform*center
    def sample(offset):
        x=cx+(radii[:,None]+offset)*np.cos(angles)
        y=cy+(radii[:,None]+offset)*np.sin(angles)
        cols=inv.a*x+inv.b*y+inv.c-.5
        rows=inv.d*x+inv.e*y+inv.f-.5
        coords=np.array([rows,cols])
        intensity=map_coordinates(smoothed,coords,order=1,mode='constant',cval=0)
        mask=map_coordinates(safe.astype(float),coords,order=0,mode='constant',cval=0)>.5
        return intensity,mask
    outside,ov=sample(pixel_size*1.5)
    inside,iv=sample(-pixel_size*1.5)
    valid_samples=ov & iv
    gradient=np.abs(outside-inside)
    # Require nearly complete valid coverage, avoiding nodata/crop boundary fits.
    eligible=valid_samples.mean(axis=1)>=.9
    sector_values=np.where(valid_samples,gradient,0).reshape(len(radii),12,15).mean(axis=2)
    # Discard the strongest two sectors; retain broad, possibly partial rim support.
    scores=np.sort(sector_values,axis=1)[:,:10].mean(axis=1)
    scores[~eligible]=-np.inf
    best=int(np.argmax(scores))
    if not np.isfinite(scores[best]) or scores[best]<.01:
        raise ValueError('No supported circular edge in this range. Adjust the range or use Circle mode.')
    radius=float(radii[best])
    native=ellipse((cx,cy),radius)
    poly=affine_transform(native,[inverse.a,inverse.b,inverse.d,inverse.e,inverse.c,inverse.f])
    support=float(np.mean(sector_values[best]>.02))
    return poly,radius,{'support':support,'at_limit':best in (0,len(radii)-1),
                        'sample_scale':scale}


def export_gpkg(path, records, crs, transform, source, overwrite=False):
    """Publish a complete GeoPackage atomically; replacement is opt-in."""
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() != '.gpkg':
        raise ValueError('Choose an output filename ending in .gpkg.')
    if path.exists() and not overwrite:
        raise FileExistsError(f'{path} exists. Choose a new filename.')
    if crs is None:
        raise ValueError('A source CRS is required.')
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = {'geometry':'Polygon', 'properties':{
        'crater_id':'int', 'method':'str', 'source':'str', 'band':'int',
        'seed_col':'float', 'seed_row':'float', 'area_native':'float',
        'catalog_id':'str', 'catalog_source':'str'}}
    fd, tmp = tempfile.mkstemp(suffix='.gpkg', dir=path.parent)
    os.close(fd)
    os.unlink(tmp)
    try:
        with fiona.open(tmp, 'w', driver='GPKG', layer='craters', crs_wkt=crs.to_wkt(), schema=schema) as dst:
            for i, rec in enumerate(records, 1):
                poly = rec['geometry']
                if poly.is_empty or not poly.is_valid or poly.area <= 0 or poly.geom_type != 'Polygon':
                    raise ValueError(f'Crater {i} has invalid geometry.')
                geo = affine_transform(poly, [transform.a,transform.b,transform.d,transform.e,transform.c,transform.f])
                dst.write({'geometry':mapping(geo), 'properties':{
                    'crater_id':i, 'method':rec['method'], 'source':str(source), 'band':rec['band'],
                    'seed_col':rec['seed'][0], 'seed_row':rec['seed'][1], 'area_native':geo.area,
                    'catalog_id':rec.get('catalog_id',''), 'catalog_source':rec.get('catalog_source','')}})
        if overwrite:
            os.replace(tmp, path)
        else:
            os.link(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def load_labels(path, src):
    """Resume this tool's existing layer without dropping unrelated content."""
    path=Path(path)
    if not path.exists():
        return []
    if fiona.listlayers(path)!=['craters']:
        raise ValueError('Output contains other layers; choose a separate label output directory.')
    records=[]
    inverse=~src.transform
    with fiona.open(path,layer='craters') as layer:
        if rasterio.crs.CRS.from_wkt(layer.crs_wkt)!=src.crs:
            raise ValueError('Existing labels have a different CRS.')
        required={'source','method','band','seed_col','seed_row'}
        if not required.issubset(layer.schema['properties']):
            raise ValueError('Existing output is not a crater-labeling GeoPackage.')
        for feature in layer:
            props=feature['properties']
            if Path(props['source']).resolve()!=Path(src.name).resolve():
                raise ValueError('Existing labels belong to a different raster.')
            native=shape(feature['geometry'])
            if native.geom_type!='Polygon' or native.is_empty or not native.is_valid:
                raise ValueError('Existing labels contain an invalid polygon.')
            poly=affine_transform(native,[inverse.a,inverse.b,inverse.d,inverse.e,inverse.c,inverse.f])
            records.append(dict(geometry=poly,seed=(props['seed_col'],props['seed_row']),
                                method=props['method'],band=props['band'],
                                catalog_id=props.get('catalog_id') or '',catalog_source=props.get('catalog_source') or ''))
    return records


def pixel_to_map(transform, point):
    """Leaflet Simple locations are (native northing, native easting)."""
    x, y = transform * tuple(point)
    return (y, x)


def map_to_pixel(transform, location):
    y, x = location
    return (~transform) * (x, y)


def raster_preview(src, bounds=None, band=1, stretch=None, max_size=1400):
    """Render a bounded north-up native-CRS window to an in-memory PNG.

    The source may be rotated: a same-CRS VRT aligns the display, while labels
    retain the original source transform. No Earth CRS or HTTP server is used.
    """
    import base64
    import io
    from PIL import Image
    from rasterio.vrt import WarpedVRT
    from rasterio.transform import from_bounds
    from rasterio.windows import from_bounds as window_from_bounds
    left, bottom, right, top = src.bounds
    if bounds:
        (south, west), (north, east) = bounds
        left, bottom, right, top = max(left, west), max(bottom, south), min(right, east), min(top, north)
    if left >= right or bottom >= top:
        return None
    # North-up grids can be read directly. Rotated grids need a display VRT.
    from contextlib import nullcontext
    if src.transform.b or src.transform.d or src.transform.a <= 0 or src.transform.e >= 0:
        res = min(math.hypot(src.transform.a, src.transform.d), math.hypot(src.transform.b, src.transform.e))
        width = max(1, math.ceil((src.bounds.right-src.bounds.left)/res))
        height = max(1, math.ceil((src.bounds.top-src.bounds.bottom)/res))
        context = WarpedVRT(src, crs=src.crs, width=width, height=height,
                            transform=from_bounds(*src.bounds,width,height), add_alpha=True)
    else:
        context = nullcontext(src)
    with context as grid:
        win = window_from_bounds(left,bottom,right,top,transform=grid.transform)
        factor = max(win.width/max_size,win.height/max_size,1)
        data = grid.read(band,window=win,out_shape=(max(1,math.ceil(win.height/factor)),
                         max(1,math.ceil(win.width/factor))),masked=True)
    data = np.ma.masked_invalid(data)
    if stretch is None:
        vals=data.compressed()
        lo,hi=np.percentile(vals,[2,98]) if vals.size else (0,1)
        stretch=(float(lo),float(hi) if hi>lo else float(lo+1))
    lo,hi=stretch
    gray=(np.clip((data.filled(lo).astype(float)-lo)/(hi-lo),0,1)*255).astype('uint8')
    rgba=np.dstack((gray,gray,gray,(~np.ma.getmaskarray(data)*255).astype('uint8')))
    buf=io.BytesIO()
    Image.fromarray(rgba).save(buf,format='PNG')
    return ('data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode(),
            ((bottom,left),(top,right)),stretch)


class CraterLabeler:
    """ipyleaflet map in the raster's native projected coordinate plane."""
    def __init__(self, data_dir, output_dir, default_raster=None, default_catalog=None):
        import ipywidgets as w
        import ipyleaflet as L
        self.L=L
        self.src=None
        self.records=[]
        self._active_index=None
        self._draft_dirty=False
        self._history=[]
        self._sync_selection=False
        self.seed=self.draft=None
        self.draft_method=''
        self.handles=[]
        self._refresh_handle=None
        self._closed=False
        self.stretch=None
        self.output_dir=Path(output_dir).expanduser().resolve()
        self.default_raster=Path(default_raster).expanduser().resolve() if default_raster else None
        from ipyfilechooser import FileChooser
        initial=self.default_raster or next(iter(sorted(Path(data_dir).glob('*.tif'))),Path(data_dir)/'')
        self.path=w.Text(value=str(initial))  # internal selected path; chooser is the visible control
        browse_dir=initial.parent if initial.suffix else initial
        while not browse_dir.is_dir() and browse_dir!=browse_dir.parent:
            browse_dir=browse_dir.parent
        self.raster_chooser=FileChooser(str(browse_dir),filename=initial.name if initial.suffix else '',
            title='<b>Raster image — GeoTIFF (.tif / .tiff)</b>',
            select_default=initial.is_file(),filter_pattern=['*.tif','*.tiff'],
            select_desc='Browse raster…',change_desc='Browse raster…',layout=w.Layout(width='95%'))
        self.raster_chooser.register_callback(self._raster_chosen)
        catalog=Path(default_catalog).expanduser() if default_catalog else Path(data_dir)
        catalog_dir=catalog.parent if catalog.suffix else catalog
        while not catalog_dir.is_dir() and catalog_dir!=catalog_dir.parent:
            catalog_dir=catalog_dir.parent
        self.catalog_chooser=FileChooser(str(catalog_dir),filename=catalog.name if catalog.suffix else '',
            title='<b>Existing crater vectors / catalog — GPKG, SHP, GeoJSON or CSV</b>',
            select_default=catalog.is_file(),filter_pattern=['*.csv','*.gpkg','*.shp','*.geojson'],
            select_desc='Browse catalog…',change_desc='Browse catalog…',layout=w.Layout(width='95%'))
        self.band=w.BoundedIntText(value=1,min=1,max=1,description='Band:')
        self.mode=w.ToggleButtons(options=['Navigate','Circle','Ellipse','Edge circle','Region grow','Edit vertices','Select/delete'],value='Edge circle',description='Mode:')
        self.radius=w.BoundedFloatText(value=30,min=.001,max=1e12,step=1,description='Radius:')
        self.radius_slider=w.FloatSlider(value=30,min=.001,max=512,step=1,description='Size:',continuous_update=False,readout=False)
        self.radius_slider.observe(lambda c:setattr(self.radius,'value',c['new']),names='value')
        self._setting_radius=False
        self.edge_min=w.BoundedFloatText(value=5,min=.001,max=1e12,description='Min radius:')
        self.edge_max=w.BoundedFloatText(value=200,min=.001,max=1e12,description='Max radius:')
        self.ratio=w.FloatSlider(value=1,min=.1,max=1,step=.01,description='Axis ratio:',continuous_update=False)
        self.angle=w.FloatSlider(value=0,min=-180,max=180,description='Angle °:',continuous_update=False)
        self.ratio.disabled=self.angle.disabled=True
        self.tolerance=w.BoundedFloatText(value=5,min=0,max=1e20,description='Tolerance:')
        self.growth=w.IntSlider(value=100,min=1,max=512,description='Limit px:',continuous_update=False)
        self.output=w.Text(value=str(self.output_dir / (Path(self.path.value).stem+'_label_craters.gpkg')),description='Labels:',disabled=True,layout=w.Layout(width='95%'))
        self.last_saved_path=None
        self.saved_status=w.HTML(value='Autosave is on: clicks and size/vertex edits update the same GeoPackage automatically.')
        self.status=w.HTML(value='Browse to a GeoTIFF to load it. The chooser browses the notebook server filesystem.')
        self.selection=w.Dropdown(options=[],description='Crater:',layout=w.Layout(width='95%'))
        self.selection.observe(self._selection_changed,names='value')
        self.coordinates=w.HTML()
        self.count=w.HTML(value='0 accepted craters')
        self.map=L.Map(crs=L.projections.Simple,layers=(),center=(0,0),zoom=0,
                       min_zoom=-30,max_zoom=30,scroll_wheel_zoom=True,
                       double_click_zoom=False,layout=w.Layout(height='650px',width='100%'))
        self.overview=L.ImageOverlay(name='Raster overview',url='',bounds=((0,0),(1,1)))
        self.detail=L.ImageOverlay(name='Raster detail',url='',bounds=((0,0),(1,1)))
        self.accepted=L.LayerGroup(name='Accepted craters')
        self.draft_layer=L.Polygon(locations=[],color='cyan',fill_opacity=.08,weight=2,smooth_factor=0,name='Draft crater')
        self.edit_layer=L.LayerGroup(name='Vertex handles')
        for layer in (self.overview,self.detail,self.accepted,self.draft_layer,self.edit_layer):
            self.map.add(layer)
        self.map.add(L.LayersControl(position='topright'))
        self.map.add(L.FullScreenControl())
        self.map.add_class('crater-raster-map')
        self.map.on_interaction(self._interaction)
        self.map.observe(self._schedule_refresh,names='bounds')
        def button(label,callback):
            b=w.Button(description=label)
            b.on_click(lambda _:self._guard(callback))
            return b
        def row(children):
            return w.HBox(children,layout=w.Layout(flex_flow="row wrap"))
        self.radius.style.description_width="initial"
        self.catalog_panel=w.Accordion(children=[w.VBox([
            w.HTML('Optional: load existing crater outlines after choosing the raster. Select a file, then import it.'),
            self.catalog_chooser,button('Import clipped catalog',self.import_catalog)])],selected_index=None)
        self.catalog_panel.set_title(0,'2. Existing crater vectors / catalog (optional)')
        # Smooth enlargement of photographic imagery; forcing pixelated rendering
        # magnifies block boundaries without recovering any additional detail.
        self.widget=w.VBox([w.HTML('<style>.crater-raster-map .leaflet-image-layer {image-rendering: auto;}</style>'),w.HTML('<b>1. Raster to label</b> — choose the background image. It loads when selected.'),self.raster_chooser,
            row([button('Load raster',self._load_raster_selection),self.band,button('Refresh view',self.refresh),button('Full extent',self.full_extent)]),
            self.catalog_panel,
            self.mode,w.HTML('Click a center to fit a crater; double-click to refit its circular edge. '
                'Click an existing crater to resize or delete it. Vertex editing is only enabled with Edit vertices.'),
            row([self.radius_slider,self.radius,self.ratio,self.angle]),
            row([self.edge_min,self.edge_max,button('Fit circle to edges',self.fit_edges)]),
            row([self.tolerance,self.growth,button('Regrow / reset',self.regenerate)]),
            self.map,self.coordinates,
            self.selection,row([button('Delete selected',self.delete_selected),button('Undo change',self.undo)]),
            self.count,self.output,button('Export GeoPackage',self.save),self.saved_status,self.status])
        self.radius.observe(lambda c:self._guard(self._radius_changed),names='value')
        for control in (self.ratio,self.angle):
            control.observe(lambda c:self._guard(self.regenerate) if self.mode.value=='Ellipse' and self.seed else None,names='value')
        self.mode.observe(lambda c:self._guard(self._mode_changed),names='value')
        self.band.observe(lambda c:self._guard(self._band_changed),names='value')

    def _load_raster_selection(self):
        # ipyfilechooser 0.6 exposes the pending form only through these internals.
        # Commit the highlighted filename, so Load does not reopen the previous file.
        chooser=self.raster_chooser
        if chooser._gb.layout.display!='none':
            candidate=Path(chooser._pathlist.value)/chooser._filename.value
            if not candidate.is_file() or candidate.suffix.lower() not in ('.tif','.tiff'):
                raise ValueError('Choose a GeoTIFF file in the raster browser, then click Load raster.')
            chooser._apply_selection()
        self._raster_chosen(chooser)

    def _raster_chosen(self,chooser):
        def choose():
            path=Path(chooser.selected or '')
            if not path.is_file():
                raise ValueError('Select an existing GeoTIFF.')
            self.path.value=str(path)
            self.load()
        self._guard(choose)

    def _guard(self,fn):
        from html import escape
        try:
            fn()
        except Exception as exc:
            self.status.value=f'<b>Error:</b> {escape(str(exc))}'

    def load(self):
        if self._draft_dirty:
            self._persist_draft()
        candidate=rasterio.open(Path(self.path.value).expanduser())
        if candidate.crs is None or not candidate.crs.is_projected:
            candidate.close()
            raise ValueError('Use a GeoTIFF with a projected lunar CRS (native easting/northing).')
        output_path=self.output_dir / (Path(candidate.name).stem+'_label_craters.gpkg')
        try:
            records=load_labels(output_path,candidate)
        except Exception:
            candidate.close()
            raise
        if self.src:
            self.src.close()
        self.discard()
        self.src=candidate
        self.records=records
        self._history=[]
        self.output.value=str(output_path)
        self.last_saved_path=output_path if output_path.exists() else None
        self.saved_status.value=f'Resumed {len(records)} saved craters.' if output_path.exists() else 'Autosave will create the raster-specific label file on your first valid crater click.'
        self.band.max=candidate.count
        self.band.value=1
        unit=candidate.crs.linear_units
        res=math.sqrt(abs(candidate.transform.determinant))
        self.radius.min=min(.001,res)
        self.radius.max=1e12
        self.radius.value=30*res
        self.radius.step=res
        self.radius_slider.min=self.radius.min
        self.radius_slider.max=max(512*res,self.radius.value)
        self.radius_slider.step=res
        self.radius_slider.description=f"Radius ({unit}):"
        self.edge_min.value=5*res
        self.edge_max.value=200*res
        self.radius.description=f'Radius ({unit}):'
        self._set_overview()
        self.full_extent()
        self._draw()
        self.status.value=f'Loaded {candidate.width:,} × {candidate.height:,} pixels. Scroll to zoom; drag to pan; click a crater center.'
        self.coordinates.value=f'Native lunar easting / northing ({unit}); ellipse radius uses {unit}.'

    def _set_overview(self):
        preview=raster_preview(self.src,band=self.band.value)
        self.overview.url,self.overview.bounds,self.stretch=preview
        self.detail.url=''

    def full_extent(self):
        if self.src:
            b=self.src.bounds
            self.map.center=((b.bottom+b.top)/2,(b.left+b.right)/2)
            self.map.zoom=math.floor(math.log2(min(800/(b.right-b.left),550/(b.top-b.bottom))))
            self.detail.url=''
            self._schedule_refresh()

    def _schedule_refresh(self,change=None):
        if self._refresh_handle:
            self._refresh_handle.cancel()
        if self.src and not self._closed:
            import asyncio
            try:
                loop=asyncio.get_running_loop()
            except RuntimeError:
                return
            self._refresh_handle=loop.call_later(.25,lambda:self._guard(self.refresh))

    def refresh(self):
        if not self.src or self._closed:
            return
        preview=raster_preview(self.src,bounds=self.map.bounds or None,band=self.band.value,stretch=self.stretch,max_size=2800)
        if preview:
            self.detail.url,self.detail.bounds,_=preview
        else:
            self.detail.url=''

    def _band_changed(self):
        if self._draft_dirty:
            self._persist_draft()
        self.discard()
        if self.src:
            self._set_overview()
            self.refresh()

    def _interaction(self,**event):
        if not self.src or event.get('type') not in ('click','dblclick'):
            return
        location=event.get('coordinates')
        if location is None:
            return
        y,x=location
        self.coordinates.value=f'Easting: {x:,.3f} · Northing: {y:,.3f} ({self.src.crs.linear_units})'
        hits=self._hits_at(location)
        if hits and self.mode.value!='Navigate':
            def edit():
                index=min(hits,key=lambda i:self.records[i]['geometry'].area)
                if index!=self._active_index:
                    self.select_record(index,preserve_mode=True)
                if event['type']=='dblclick' and self.mode.value in ('Circle','Ellipse','Edge circle','Region grow'):
                    self._guard(self.fit_edges)
            self._guard(edit)
            return
        if self.mode.value=='Select/delete':
            self._guard(lambda:self._select_at(location))
            return
        if self.mode.value not in ('Circle','Ellipse','Edge circle','Region grow'):
            return
        def make():
            col,row=map_to_pixel(self.src.transform,location)
            if not (0<=col<self.src.width and 0<=row<self.src.height):
                raise ValueError('Click inside the raster.')
            sample=self.src.read(self.band.value,window=Window(int(col),int(row),1,1),masked=True)
            if np.ma.is_masked(sample[0,0]) or not np.isfinite(sample[0,0]):
                raise ValueError('Seed falls on nodata.')
            if self._draft_dirty:
                self._persist_draft()
            self.discard()
            self.seed=(col,row)
            self.regenerate()
        self._guard(make)

    def _mode_changed(self):
        self.ratio.disabled=self.angle.disabled=self.mode.value!='Ellipse'
        self._remove_handles()
        if self._setting_radius:
            return
        if self.mode.value=='Edit vertices':
            if self.draft is None:
                raise ValueError('Click a center in Circle, Ellipse, Edge circle or Region grow mode first.')
            self.vertices=list(self.draft.exterior.coords)[:-1]
            for i,point in enumerate(self.vertices):
                marker=self.L.Marker(location=pixel_to_map(self.src.transform,point),draggable=True,
                    icon=self.L.DivIcon(html='<div style="width:8px;height:8px;background:white;border:1px solid #008b8b;border-radius:50%"></div>',icon_size=[10,10],icon_anchor=[5,5]))
                marker.observe(lambda c,index=i:self._guard(lambda:self._vertex_changed(index,c['new'])),names='location')
                self.handles.append(marker)
            self.edit_layer.layers=tuple(self.handles)
            self.status.value='Drag the white vertex handles onto the crater rim. Valid edits save automatically.'
        elif self.mode.value in ('Circle','Ellipse','Edge circle','Region grow') and self.seed:
            self.regenerate()

    def _vertex_changed(self,index,location):
        self.vertices[index]=map_to_pixel(self.src.transform,location)
        self.draft=Polygon(self.vertices)
        self.draft_method=self.draft_method.split('+')[0]+'+edited'
        self._persist_draft()

    def regenerate(self):
        if not self.src or self.seed is None:
            return
        if self.mode.value not in ('Circle','Ellipse','Edge circle','Region grow'):
            raise ValueError('Choose Circle, Ellipse, Edge circle or Region grow to regenerate the outline.')
        self._remove_handles()
        if self.mode.value=='Edge circle':
            self.fit_edges()
            return
        if self.mode.value in ('Circle','Ellipse'):
            # Construct in physical map units so nonsquare/rotated pixels do not distort circles.
            center=self.src.transform*self.seed
            native=ellipse(center,self.radius.value,self.ratio.value if self.mode.value=='Ellipse' else 1,self.angle.value)
            t=~self.src.transform
            self.draft=affine_transform(native,[t.a,t.b,t.d,t.e,t.c,t.f])
            self.draft_method='ellipse' if self.mode.value=='Ellipse' else 'circle'
            message='Type a radius and press Enter to resize from the same center. Changes save automatically; click the next crater center when ready.'
        else:
            self.draft,limited=grow_region(self.src,self.seed,self.tolerance.value,self.growth.value,self.band.value)
            self.draft_method='region_grow'
            message='Review the intensity-based outline; edit vertices or change tolerance and click Regrow / reset.'
            if limited:
                message='Growth reached its distance/window boundary. '+message
        self.status.value=message
        self._persist_draft()

    def _radius_changed(self):
        self.radius_slider.max=max(self.radius_slider.max,self.radius.value)
        self.radius_slider.value=self.radius.value
        if self._setting_radius or not self.seed:
            return
        if self.mode.value in ('Circle','Ellipse'):
            self.regenerate()
        elif self.mode.value=='Edge circle':
            center=self.src.transform*self.seed
            native=ellipse(center,self.radius.value)
            t=~self.src.transform
            self.draft=affine_transform(native,[t.a,t.b,t.d,t.e,t.c,t.f])
            self.draft_method='edge_circle+adjusted'
            self._persist_draft()
            self.status.value='Edge proposal resized manually. Click Fit circle to edges to refit.'
        elif self.mode.value in ('Select/delete','Edit vertices') and self.draft is not None:
            # Scale the selected geometry instead of creating a new circle or losing an edited rim.
            old_radius=math.sqrt(self.draft.area*abs(self.src.transform.determinant)/math.pi)
            factor=self.radius.value/old_radius
            self.draft=scale(self.draft,xfact=factor,yfact=factor,origin=self.seed)
            self.draft_method=self.draft_method.split('+')[0]+'+adjusted'
            self._persist_draft()
            if self.mode.value=='Edit vertices':
                self._mode_changed()

    def fit_edges(self):
        if not self.src or self.seed is None:
            raise ValueError('Click a crater center first, then fit the circle to edges.')
        # On failure retain the existing draft; never replace it with a failed fit.
        poly,radius,info=edge_circle(self.src,self.seed,self.edge_min.value,self.edge_max.value,self.band.value)
        self._remove_handles()
        self._setting_radius=True
        try:
            self.radius.value=radius
            self.mode.value='Edge circle'
        finally:
            self._setting_radius=False
        self.draft=poly
        self.draft_method='edge_circle'
        self._draw()
        self.status.value=(f'Edge proposal: radius {radius:.2f} {self.src.crs.linear_units}; '
            f'{info["support"]:.0%} of angular sectors have contrast (not a confidence score). '
            'Review the rim; type a radius to refine or click Fit circle to edges again.'+
            (' Best fit is at a search limit; widen the radius range.' if info['at_limit'] else ''))
        self._persist_draft()

    def _remove_handles(self):
        self.edit_layer.layers=()
        for marker in self.handles:
            marker.close()
        self.handles=[]

    def _draw(self):
        def locations(poly):
            return [pixel_to_map(self.src.transform,p) for p in poly.exterior.coords]
        old=self.accepted.layers
        self.accepted.layers=tuple(self.L.Polygon(locations=locations(rec['geometry']),color='lime',
            weight=2,fill_opacity=.08) for i,rec in enumerate(self.records) if i!=self._active_index)
        for layer in old:
            layer.close()
        self.draft_layer.locations=locations(self.draft) if self.draft is not None else []
        self.count.value=f'{len(self.records)} saved craters. Cyan is selected; green is saved. Click a crater to resize; double-click to fit the circular edge.'
        self._sync_selection=True
        try:
            self.selection.options=[('Select a saved crater',None)]+[
                (f"{i+1}: {rec.get('catalog_id') or rec['method']}",i) for i,rec in enumerate(self.records)]
            self.selection.value=self._active_index
        finally:
            self._sync_selection=False

    def _persist_draft(self):
        self._draft_dirty=True
        if self.draft is None or not self.draft.is_valid or self.draft.area<=0:
            raise ValueError('Outline is invalid; last saved version is unchanged. Fix the outline or undo.')
        xmin,ymin,xmax,ymax=self.draft.bounds
        if xmin < -1e-6 or ymin < -1e-6 or xmax>self.src.width+1e-6 or ymax>self.src.height+1e-6:
            raise ValueError('Outline extends outside the raster; resize it to save.')
        previous=self.records[self._active_index] if self._active_index is not None else {}
        record={**previous,'geometry':self.draft,'seed':self.seed,'method':self.draft_method,'band':self.band.value}
        records=list(self.records)
        index=self._active_index
        if index is None:
            index=len(records)
            records.append(record)
        else:
            records[index]=record
        try:
            self._commit(records)
        except Exception as exc:
            raise RuntimeError(f'Autosave failed; last saved labels are unchanged and the current outline is retained. {exc}') from exc
        self._active_index=index
        self._draft_dirty=False
        self._draw()

    def _commit(self,records):
        self._save_labels(records)
        self._history.append(list(self.records))
        self._history=self._history[-30:]
        self.records=records

    def accept(self):
        """Compatibility helper; the interface saves without this button."""
        if self._draft_dirty:
            self._persist_draft()
        self.discard()

    def discard(self):
        self._remove_handles()
        self.seed=self.draft=None
        self._active_index=None
        self._draft_dirty=False
        self._draw()

    def _selection_changed(self,change):
        if not self._sync_selection and change['new'] is not None:
            self._guard(lambda:self.select_record(change['new']))

    def select_record(self,index,preserve_mode=True):
        if self._draft_dirty:
            self._persist_draft()
        if not 0<=index<len(self.records):
            raise ValueError('Select an existing crater.')
        self.discard()
        record=self.records[index]
        self._active_index=index
        self.seed=record['seed']
        self.draft=record['geometry']
        self.draft_method=record['method']
        self._setting_radius=True
        try:
            self.radius.value=math.sqrt(self.draft.area*abs(self.src.transform.determinant)/math.pi)
            if not preserve_mode:
                self.mode.value='Select/delete'
        finally:
            self._setting_radius=False
        self._draw()
        self.status.value='Crater selected. Resize with the slider or radius box; double-click in a drawing mode to refit the circular edge. Choose Edit vertices explicitly for handles. Edits update this same saved crater. Choose a drawing mode to add another.'

    def _hits_at(self,location):
        from shapely.geometry import Point
        point=Point(map_to_pixel(self.src.transform,location))
        return [i for i,r in enumerate(self.records) if r['geometry'].covers(point)]

    def _select_at(self,location):
        hits=self._hits_at(location)
        if not hits:
            raise ValueError('Click inside a saved crater, or choose one from the Crater list.')
        self.select_record(min(hits,key=lambda i:self.records[i]['geometry'].area))

    def delete_selected(self):
        if self._active_index is None:
            if self.draft is not None:
                self.discard()
                return
            raise ValueError('Choose a crater from the list or click it in Select/delete mode.')
        records=[r for i,r in enumerate(self.records) if i!=self._active_index]
        self._commit(records)
        self.discard()
        self.status.value='Crater deleted and saved. Undo change restores it.'

    def undo(self):
        if self._draft_dirty:
            index=self._active_index
            self.discard()
            if index is not None:
                self.select_record(index)
            return
        if self._history:
            records=self._history[-1]
            self._save_labels(records)
            self._history.pop()
            self.records=records
            self.discard()

    def import_catalog(self):
        from .catalogs import import_catalog
        if not self.src:
            raise ValueError('Choose a raster before importing a catalog.')
        if self._draft_dirty:
            self._persist_draft()
        path=self.catalog_chooser.selected
        if not path or not Path(path).is_file():
            raise ValueError('Choose an existing catalog first.')
        imported,stats=import_catalog(path,self.src,self.band.value)
        existing={(r.get('catalog_source'),r.get('catalog_id')) for r in self.records if r.get('catalog_id')}
        additions=[r for r in imported if (r['catalog_source'],r['catalog_id']) not in existing]
        if additions:
            self._commit([*self.records,*additions])
        self.discard()
        self.status.value=(f"Imported and saved {len(additions)} clipped crater polygons; "
            f"{stats['intersecting']} catalog craters intersect this scene; {stats['skipped']} invalid entries skipped. "
            'Select/delete mode lets you remove existing craters before redrawing them.')

    def save(self):
        if not self.src:
            raise ValueError('Load a raster first.')
        if self._draft_dirty:
            self._persist_draft()
        else:
            self._save_labels(self.records)
        self.status.value='Saved to the raster-specific GeoPackage.'

    def _save_labels(self,records):
        from html import escape
        path=Path(self.output.value)
        export_gpkg(path,records,self.src.crs,self.src.transform,self.src.name,
                    overwrite=self.last_saved_path==path)
        self.last_saved_path=path
        self.saved_status.value=f'<b>Saved {len(records)} craters:</b> {escape(str(path))}'
        return path

    def close(self):
        self._closed=True
        if self._refresh_handle:
            self._refresh_handle.cancel()
        self._remove_handles()
        self.map.unobserve(self._schedule_refresh,names='bounds')
        self.map.on_interaction(self._interaction,remove=True)
        if self.src:
            self.src.close()
        self.map.close()
        self.widget.close()
