# Static NoData Tiling Refactor Plan

## Scope

This plan is limited to the tiling/datacube generation code under `model/`, primarily `model/Pipeline.py`. Chip-making code under `model/chip_making/` is intentionally out of scope for this refactor and should be investigated separately.

## Problem Summary

Static datacube generation currently applies NoData handling after resampling rather than during resampling. For static bands with band-specific NoData values, especially CPR/S1 bands, this allows NoData sentinel values to participate in bilinear interpolation. The result is a large population of finite, invalid negative values clustered around the source NoData sentinel instead of a clean shared NoData value.

The current static write path also computes a normalized NoData array but writes the original pixel array, so even exact NoData replacement is not applied.

Large positive artifacts are suspected to come from later chip-making resampling or another downstream operation and should be validated separately.

## Constraints

- Keep bilinear resampling for static tiling. Nearest-neighbor resampling created artifacts in earlier testing and should not be the default fix.
- Preserve the shared static output NoData value as a float value: `-32768.0`.
- Do not add hard failure diagnostics to the production tiling path yet. Diagnostics will be run separately after datacube creation.
- Add regression tests eventually, but implementation can start with a targeted refactor and manual validation.

## Proposed Refactor Steps

1. Make static clipping/warping NoData-aware

   Update the Pipeline clipping path so `gdal.Warp` receives explicit `srcNodata` and `dstNodata` values for static rasters. This keeps bilinear resampling but prevents source NoData from contributing to interpolation.

   Static clipping should be able to pass band-specific values similar to:

   ```python
   srcNodata=<band-specific source nodata>
   dstNodata=<band-specific output nodata>
   resampleAlg=gdal.GRA_Bilinear
   ```

   Keep existing WAC behavior stable unless there is a clear reason to change it. Static and WAC clipping can share implementation, but static data needs explicit NoData handling because the static bands do not all share one source NoData value.

2. Preserve the intended multiple-NoData policy

   Do not force every static NoData value to `-32768.0`. Use the following policy:

   - Normalize the Pipeline/shared static NoData value to float `-32768.0`.
   - Preserve the known CPR/S1 band-specific source NoData sentinel values for static bands `08` and `09`.
   - Document those preserved CPR/S1 sentinels as additional NoData values that downstream training/data-loading policy must mask.

   For CPR/S1 bands, the warp should treat the CPR/S1 sentinel as NoData during resampling and preserve that sentinel as the destination NoData value. For static bands that already use the Pipeline/shared NoData convention, the destination NoData should be `-32768.0`.

3. Normalize static arrays before storing them

   After warp, normalize remaining exact NoData values according to the policy above before appending arrays to the static cube dictionary. The ordering still matters: masking/normalization should happen immediately after clipping/warping and before the raster is stored or written.

   Avoid broad numeric thresholds in the production tiling path unless the team explicitly approves them. Exact known NoData values should drive the refactor; post-run diagnostics can continue checking for unexpected large-magnitude artifacts.

4. Fix static write bug

   In `_writeStaticCube()`, write the normalized array, not the original pixels:

   ```python
   band.WriteArray(raster)
   ```

   instead of:

   ```python
   band.WriteArray(pixels)
   ```

5. Preserve output metadata

   Static output bands should continue to set:

   ```python
   band.SetNoDataValue(-32768.0)
   band.SetMetadataItem("Name", name)
   ```

   The output dataset should remain float-compatible so `-32768.0` is represented as a float NoData value.

   With the revised multiple-NoData policy, this metadata is not the complete NoData contract for CPR/S1 bands. The preserved CPR/S1 sentinels must also be documented in the data dictionary/model input policy so training and diagnostics can mask them intentionally.

6. Keep output ordering stable

   Do not change static band discovery, filtering, product grouping, or band ordering as part of this refactor. The first pass should only change NoData-aware warp/normalization behavior.

## Validation Plan

1. Run the existing filtered static cube diagnostic AOIs with the refactored tiling code.

2. Compare `static_08` and `static_09` value counts before and after refactor at the static datacube stage.

3. Expected datacube-stage result:

   - No `-inf`.
   - No non-exact e30/e37/e38 finite artifacts.
   - Exact CPR/S1 source NoData sentinel values may remain in static bands `08` and `09` only.
   - Pipeline/shared NoData should appear as float `-32768.0`.
   - Valid negative CPR/S1 values should remain unchanged.

4. After the tiling output is clean, inspect `model/chip_making/` separately to determine whether downstream reprojection creates the large positive artifacts.

## Future Tests

Add targeted regression tests once the implementation shape is stable:

- A static raster with a CPR/S1-like NoData sentinel adjacent to valid pixels.
- A bilinear warp into a tile-sized output.
- Assertions that output contains valid values plus the documented NoData values, with no `inf`, `nan`, or non-exact e30-scale artifacts.
- A test that `_writeStaticCube()` writes the normalized array and records `-32768.0` as the band NoData value.

## Regression Testing Workflow

Use two complementary regression setups:

1. HPC integration comparison

   Keep two clean HPC working copies checked out to different branches:

   ```text
   lfm_main  -> main/develop/prod baseline
   lfm_dev   -> static NoData refactor branch
   ```

   Run the same AOIs, tile database, staticLinks, Apptainer container, and diagnostic commands in both repos. Write outputs to separate directories, for example:

   ```text
   /explore/nobackup/people/ajkerr1/static_regression/main
   /explore/nobackup/people/ajkerr1/static_regression/dev
   ```

   Compare machine-readable summaries for:

   - Static band names and ordering.
   - Raster shape, CRS, transform, and NoData metadata.
   - Exact source NoData counts.
   - Shared `-32768.0` NoData counts.
   - `inf` and `nan` counts.
   - `abs(value) > 1e30` counts.
   - Min, max, mean, and std for CPR/S1 static bands.
   - Optional pixelwise differences over valid pixels.

   The baseline run should reproduce the current bad behavior. The refactor run should remove `-inf` and non-exact e30-scale artifacts at the static datacube stage while preserving valid negative CPR/S1 values and documented exact CPR/S1 NoData sentinels.

2. Synthetic pytest regression

   Add a small synthetic test after the refactor has settled. This test should construct a tiny raster with valid values adjacent to a CPR/S1-like NoData sentinel, run the Pipeline clipping/static write path, and assert that bilinear resampling does not produce sentinel-scale finite artifacts.

   The synthetic test should be fast enough for local or CI-style runs, while the two-repo HPC comparison remains the end-to-end confidence check.
