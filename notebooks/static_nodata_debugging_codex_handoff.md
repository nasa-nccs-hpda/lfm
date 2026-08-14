# Static Lunar Data NoData / `-inf` Debugging Handoff

## Objective

We are debugging invalid values in a set of multi-band lunar GeoTIFF chips used for an ML workflow.

The immediate ML goal is to compute per-channel training-set statistics (mean/std) for z-score normalization of the **static modality**, which occupies **band indices 7–69 inclusive (0-indexed)** in 70-channel image chips. Before computing those statistics, we need to understand and correctly handle invalid/fill values.

The main issue discovered is that some generated chip bands contain large numbers of `-inf` pixels even though the corresponding original source GeoTIFFs do not contain `-inf`. Evidence so far strongly suggests that some original source **NoData/fill pixels are being converted to `-inf` somewhere in the static-data preprocessing/chip-generation pipeline**.

Codex should inspect the code that reads, reprojects/resamples, normalizes/transforms, stacks, and writes these static rasters and identify exactly where this conversion occurs.

---

## Relevant data layout

### Current generated dataset

The current chips are 70-channel GeoTIFFs arranged in train/val/test splits, for example:

```text
/explore/nobackup/projects/lfm/model_inputs/300_300_inputs/
    fm_all_static_all_wac_iseg/
        train/chips/*.tif
        val/chips/*.tif
        test/chips/*.tif
```

Example chip:

```text
/explore/nobackup/projects/lfm/model_inputs/300_300_inputs/
fm_all_static_all_wac_iseg/train/chips/
M1096558039CE_r7650_c750_input_wac_chip.tif
```

The chips were verified to have **70 channels**.

Static data is represented by:

```python
STATIC_BANDS_0IDX = np.arange(7, 70)
```

which is 63 static channels.

Remember that Rasterio uses 1-indexed band numbers:

```python
STATIC_BANDS_RASTERIO = (STATIC_BANDS_0IDX + 1).tolist()
```

---

### Older generated static dataset

We also compared against older generated chips:

```text
/explore/nobackup/projects/lfm/model_inputs/300_300_inputs/
all_static_all_wac/inst_seg/chips/*.tif
```

The invalid-value pattern already exists in this older dataset, so the issue predates the current train/val/test reorganization.

---

## Original source static GeoTIFFs

Two source filenames relevant to the problematic bands include:

```text
GlobeNoPolesDeltaCPR_v2-offsetto49d.iau.tif
GlobeNoPolesDeltaS1_v2.iau.tif
```

One confirmed source path is:

```text
/explore/nobackup/projects/lfm/processed_data/Lunar/Static_final/
mini_rf/GlobeNoPolesDeltaCPR_v2-offsetto49d.iau.tif
```

The CPR source raster is enormous:

```text
121293 x 57118 pixels
Float32
```

Its CRS is a custom lunar equirectangular projection:

```text
Moon (2015) - Sphere / Ocentric / Equirectangular, clon = 0
```

Generated chips are instead in local lunar Transverse Mercator (LTM) projections.

Example chip CRS:

```text
Moon (2015) - Sphere / Ocentric / Tranverse Mercator
central_meridian = -8
```

Therefore, direct pixel-window comparisons between the source and chips require coordinate transformation/reprojection.

---

## Source NoData value

For the CPR source:

```text
NoData Value = -3.40282306073709653e+38
```

Rasterio reports approximately:

```python
-3.4028230607370965e+38
```

This is very close to the minimum representable `float32` value.

An important observation from generated chips was a slightly different reported NoData/fill value:

```text
-3.40282265508890445e+38
```

These values are **not numerically identical**.

One current hypothesis is that preprocessing code attempted to use a common fill/NoData value across many source GeoTIFFs but did not exactly match the fill value of these particular source rasters.

Codex should verify how source NoData is detected and how destination NoData is assigned.

---

## Invalid-value scan of generated chips

We wrote multiprocessing diagnostics that iterate over all static bands and count:

- `NaN`
- declared NoData
- `-inf`
- `+inf`

### Older generated data

```text
Band  7: NaN=0, nodata=2,530,899, -inf=0,       +inf=0
Band  8: NaN=0, nodata=95,        -inf=543,486, +inf=0
Band  9: NaN=0, nodata=66,        -inf=452,649, +inf=0
Band 20: NaN=0, nodata=365,044,   -inf=0,       +inf=0
Band 21: NaN=0, nodata=7,592,     -inf=0,       +inf=0
Band 46: NaN=0, nodata=7,592,     -inf=0,       +inf=0
Band 47: NaN=0, nodata=159,222,   -inf=0,       +inf=0
Band 48: NaN=0, nodata=159,222,   -inf=0,       +inf=0
Band 49: NaN=0, nodata=159,222,   -inf=0,       +inf=0
Band 50: NaN=0, nodata=159,222,   -inf=0,       +inf=0
Band 51: NaN=0, nodata=159,222,   -inf=0,       +inf=0
Band 52: NaN=0, nodata=159,222,   -inf=0,       +inf=0
Band 53: NaN=0, nodata=159,222,   -inf=0,       +inf=0
```

### Current generated data

```text
Band  7: NaN=0, nodata=2,040,410, -inf=0,       +inf=0
Band  8: NaN=0, nodata=73,        -inf=430,042, +inf=0
Band  9: NaN=0, nodata=50,        -inf=354,523, +inf=0
Band 20: NaN=0, nodata=271,507,   -inf=0,       +inf=0
Band 21: NaN=0, nodata=5,705,     -inf=0,       +inf=0
Band 46: NaN=0, nodata=5,705,     -inf=0,       +inf=0
Band 47: NaN=0, nodata=151,378,   -inf=0,       +inf=0
Band 48: NaN=0, nodata=151,378,   -inf=0,       +inf=0
Band 49: NaN=0, nodata=151,378,   -inf=0,       +inf=0
Band 50: NaN=0, nodata=151,378,   -inf=0,       +inf=0
Band 51: NaN=0, nodata=151,378,   -inf=0,       +inf=0
Band 52: NaN=0, nodata=151,378,   -inf=0,       +inf=0
Band 53: NaN=0, nodata=151,378,   -inf=0,       +inf=0
```

Key observations:

1. There are no NaNs or `+inf` values in these scans.
2. Most problematic bands contain only declared NoData.
3. **Bands 8 and 9 (0-indexed) contain hundreds of thousands of `-inf` pixels.**
4. The same pattern is present in both old and current generated datasets.
5. Therefore, the current dataset-copy/re-splitting operation did not introduce the problem.

---

## Full scan of the CPR source raster

The original CPR GeoTIFF was scanned block-by-block without loading the entire raster into memory.

Result:

```python
{
    "nodata": 3086838926,
    "nan": 0,
    "neginf": 0,
    "posinf": 0,
}
```

Therefore:

- the original CPR source contains a very large amount of NoData;
- it contains **zero `-inf` pixels**;
- it contains **zero `+inf` pixels**;
- it contains **zero NaN pixels**.

This establishes that the `-inf` values seen in the generated chips are introduced downstream of the original source file.

The valid CPR source data appears to be approximately in the range:

```text
[-0.5, 3.0]
```

Negative CPR values are somewhat unusual but are not automatically invalid. Do not treat all negative values as missing.

---

## Direct source-vs-chip comparison

We identified the first current chip containing a `-inf` value in the investigated CPR-related band:

```text
/explore/nobackup/projects/lfm/model_inputs/300_300_inputs/
fm_all_static_all_wac_iseg/train/chips/
M1096558039CE_r7650_c750_input_wac_chip.tif
```

For that chip:

```text
Band 7: 0 -inf pixels
Band 8: 37 -inf pixels
```

The chip is 300x300 at 100 m resolution.

Because the source and chip use different projections, the CPR source was lazily warped onto the chip's exact grid using Rasterio `WarpedVRT` with nearest-neighbor resampling.

Comparison result:

```text
Band 8
  Chip -inf pixels:                  37
  Source nodata pixels on chip grid: 279
  -inf AND source nodata:            37
  -inf BUT source valid:             0
  Fraction of -inf matching nodata:  100.00%
```

This is strong evidence that the `-inf` pixels in this sample originate from locations that were NoData in the original source raster.

Important nuance:

- 37 chip `-inf` pixels corresponded to source NoData.
- The reprojected source showed 279 NoData pixels in the chip footprint.
- Therefore, not every source NoData pixel became `-inf`.
- Some NoData pixels remain represented another way, or behavior depends on interpolation/processing conditions.

The immediate debugging target is therefore **how source NoData pixels are handled through static preprocessing**.

---

## Current working hypothesis

The likely bug is in the static-raster preprocessing/chipping pipeline rather than in the source data.

Potential mechanisms to investigate:

### 1. Incorrect equality comparison against a common fill value

If code uses a hard-coded/shared fill value such as:

```python
COMMON_NODATA = ...
mask = data == COMMON_NODATA
```

but the source TIFF has:

```text
-3.4028230607370965e+38
```

while the chosen common value is closer to:

```text
-3.40282265508890445e+38
```

then the source NoData pixels may fail to be recognized before subsequent processing.

Check whether a value was:

- typed manually;
- serialized/deserialized at different precision;
- cast through `float32`/`float64`;
- taken from a different static raster;
- used as a universal NoData value for heterogeneous source datasets.

Prefer reading:

```python
src.nodata
```

for each source raster rather than assuming all static rasters share exactly the same value.

---

### 2. Arithmetic applied to NoData before masking

Because the source fill value is extremely close to the negative `float32` limit, arithmetic can overflow it to `-inf`.

For example, operations such as:

```python
data * scale
data - offset
data / value
np.log(...)
np.log10(...)
```

may convert an unmasked fill value into an infinity.

The safe conceptual pattern is:

```python
src_nodata = src.nodata

valid = (
    np.isfinite(data)
    & (data != src_nodata)
)

output = np.full(data.shape, dst_nodata, dtype=np.float32)

output[valid] = transform(data[valid])
```

rather than transforming all pixels and attempting to restore NoData afterward.

Codex should inspect whether any transforms are applied specifically to the layers corresponding to generated bands 8 and 9.

---

### 3. Reprojection/resampling NoData handling

The source data is equirectangular while the chips are local Transverse Mercator, so some reprojection/resampling step necessarily occurs.

Inspect calls involving:

- `rasterio.warp.reproject`
- `WarpedVRT`
- GDAL warp APIs
- `gdal.Warp`
- `gdal.Translate`
- xarray/rioxarray reprojection
- custom interpolation/resampling

Verify that the source-specific NoData is explicitly passed through, e.g.:

```python
src_nodata=src.nodata
dst_nodata=<desired destination nodata>
```

Also inspect the chosen resampling method and behavior around NoData boundaries.

---

### 4. Dtype conversion

Inspect all conversions between:

```text
float64
float32
integer types
```

especially near reading/writing and stacking.

Because the fill value sits near `float32`'s minimum finite value, small representation differences matter.

---

## What Codex should search for in the codebase

Find the code responsible for creating the original multi-static-band chips from the source GeoTIFFs.

Search for combinations of terms such as:

```text
GlobeNoPolesDeltaCPR
GlobeNoPolesDeltaS1
mini_rf
Static_final
nodata
NoData
fill
fill_value
src_nodata
dst_nodata
reproject
Warp
WarpedVRT
resampling
stack
static
chip
float32
-3.40282
```

The most important question is:

> Where does a CPR source pixel equal to that TIFF's declared NoData value cease being treated as NoData, and what operation converts a subset of those pixels to `-inf`?

---

## Useful diagnostic code pattern

A worker used to count invalid pixels per static channel has roughly this structure:

```python
def inspect_chip(file_name):
    with rasterio.open(file_name) as src:
        nodata = src.nodata

        data = src.read(
            indexes=STATIC_BANDS_RASTERIO,
            masked=False,
        ).astype(np.float64)

    n_nan = np.isnan(data).sum(axis=(1, 2))
    n_posinf = np.isposinf(data).sum(axis=(1, 2))
    n_neginf = np.isneginf(data).sum(axis=(1, 2))

    if nodata is None:
        n_nodata = np.zeros(data.shape[0], dtype=np.int64)
    else:
        n_nodata = (data == nodata).sum(axis=(1, 2))

    return {
        "nan": n_nan.astype(np.int64),
        "nodata": n_nodata.astype(np.int64),
        "posinf": n_posinf.astype(np.int64),
        "neginf": n_neginf.astype(np.int64),
    }
```

This has already established the generated-data patterns described above.

---

## QGIS / spatial inspection workflow

We have also been inspecting source imagery directly in QGIS.

Because the source TIFF is extremely large, the plan is to extract small windows rather than copy/download the full raster.

Desired diagnostic source windows:

1. A **1000x1000** source window containing **25%–75% NoData**.
2. A **1000x1000** source window containing **>0% and <25% NoData**, preferably with NoData spatially scattered throughout the window.

The search is performed using Rasterio window reads from top-left to bottom-right, using non-overlapping 1000x1000 windows.

For scattered NoData, candidate windows can be divided into a 5x5 grid and required to contain NoData in multiple grid cells.

The selected windows are exported as native-projection GeoTIFFs for QGIS inspection.

No reprojection should be applied to these source inspection extracts unless needed later; preserving exact source pixels is desirable for debugging.

---

## PROJ environment issue observed

During one Rasterio/WarpedVRT diagnostic, the notebook emitted:

```text
ERROR 1: PROJ: internal_proj_identify:
.../share/proj/proj.db contains DATABASE.LAYOUT.VERSION.MINOR = 5
whereas a number >= 6 is expected.
It comes from another PROJ installation.
```

The diagnostic still produced a plausible spatial comparison, but this indicates multiple incompatible PROJ installations are visible in the environment.

This appears separate from the `-inf` problem because the `-inf` values already exist in previously generated chips, but it should be fixed before relying heavily on new reprojection work.

Codex should avoid conflating this environment issue with the NoData conversion bug unless evidence connects them.

---

## ML normalization context

Once the static values are verified/fixed, we need per-band mean/std statistics over the **training split only** for z-score normalization.

The intended statistics are:

- one global mean per static channel;
- one global standard deviation per static channel;
- calculated over all valid training pixels;
- every valid pixel weighted equally;
- NoData, NaN, `+inf`, and `-inf` excluded.

There are 63 static channels:

```text
0-indexed bands 7 through 69
```

The final outputs should be NumPy arrays:

```python
static_mean.shape == (63,)
static_std.shape == (63,)
```

and their values will later be inserted into modality configuration YAML under a structure analogous to:

```yaml
scaler_dict:
  static: std

stats:
  static:
    mean:
      # 63 values
    std:
      # 63 values

num_channels: 63
```

Do **not** treat the currently observed `-inf` values as legitimate values in the normalization calculation.

Ideally, resolve the preprocessing bug first. At minimum, stats must exclude both declared NoData and all non-finite values.

---

## Requested next steps for Codex

Please inspect the available static-data preprocessing/chip-generation code and:

1. Identify the exact code path used to process `GlobeNoPolesDeltaCPR_v2-offsetto49d.iau.tif` and `GlobeNoPolesDeltaS1_v2.iau.tif`.
2. Determine which generated 0-indexed chip bands correspond to each source raster.
3. Trace source NoData handling from raster read through reprojection/resampling, optional transformation, stacking, and GeoTIFF write.
4. Identify any hard-coded or shared fill/NoData value.
5. Compare that value against `src.nodata` for each source raster.
6. Identify arithmetic performed before invalid pixels are masked.
7. Check whether `src_nodata` and `dst_nodata` are explicitly supplied during reprojection.
8. Check dtype conversions that could alter values near the `float32` minimum.
9. Explain the most likely mechanism by which source NoData becomes `-inf`.
10. Propose a minimal fix that preserves legitimate negative source values.
11. Add or propose a regression test that verifies:
    - valid source pixels remain finite;
    - source NoData is represented consistently as destination NoData;
    - no generated static chip contains unexpected `NaN`, `+inf`, or `-inf`;
    - legitimate negative CPR values are preserved.
12. Avoid changing unrelated processing unless necessary.

Before implementing a broad fix, report the relevant code path and explain the specific failure mechanism so it can be validated against the data observations above.

---

## Strongest confirmed evidence

The most important facts to preserve while debugging are:

1. **Original CPR source:** billions of NoData pixels, but zero NaN/`-inf`/`+inf`.
2. **Generated old and current chips:** bands 8 and 9 contain large numbers of `-inf`.
3. **The problem predates the latest dataset split/copy operation.**
4. **One directly compared chip:** every one of its 37 band-8 `-inf` pixels corresponded to source NoData after aligning the source to the chip grid.
5. **Source and generated NoData values observed are slightly different.**
6. Therefore, current evidence favors a **NoData/fill handling bug during static preprocessing**, not inherently invalid values in the original CPR source imagery.
