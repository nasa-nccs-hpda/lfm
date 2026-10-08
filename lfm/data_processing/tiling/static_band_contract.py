"""Canonical band order and NoData constants for the static lunar modality."""

STATIC_BAND_NAMES = (
    "LDRM_32_N_FLOAT.iau",
    "GlobeNoPolesDeltaCPR_v2-offsetto49d.iau",
    "GlobeNoPolesDeltaS1_v2.iau",
    "WAC_EMP_321NM.iau",
    "WAC_EMP_360NM.iau",
    "WAC_EMP_415NM.iau",
    "WAC_EMP_566NM.iau",
    "WAC_EMP_604NM.iau",
    "WAC_EMP_643NM.iau",
    "WAC_EMP_689NM.iau",
    "WAC_GLOBAL.iau",
    "WAC_TIO2.iau",
    "jggrx_1800f_me_dist_meters_20km_cog",
    "hpar_global128ppd_v1c_dateline_cut.iau3",
    "RA_SAM_70Sto70N.iau7",
    "diviner_tbol_snapshot_000E",
    "diviner_tbol_snapshot_015E",
    "diviner_tbol_snapshot_030E",
    "diviner_tbol_snapshot_045E",
    "diviner_tbol_snapshot_060E",
    "diviner_tbol_snapshot_075E",
    "diviner_tbol_snapshot_090E",
    "diviner_tbol_snapshot_105E",
    "diviner_tbol_snapshot_120E",
    "diviner_tbol_snapshot_135E",
    "diviner_tbol_snapshot_150E",
    "diviner_tbol_snapshot_165E",
    "diviner_tbol_snapshot_180E",
    "diviner_tbol_snapshot_195E",
    "diviner_tbol_snapshot_210E",
    "diviner_tbol_snapshot_225E",
    "diviner_tbol_snapshot_240E",
    "diviner_tbol_snapshot_255E",
    "diviner_tbol_snapshot_270E",
    "diviner_tbol_snapshot_285E",
    "diviner_tbol_snapshot_300E",
    "diviner_tbol_snapshot_315E",
    "diviner_tbol_snapshot_330E",
    "diviner_tbol_snapshot_345E",
    "TREG_ANOM_70Sto70N.iau7",
    "Lunar_Kaguya_MIMap_MineralDeconv_ClinopyroxenePercent_50N50S.iau2",
    "Lunar_Kaguya_MIMap_MineralDeconv_FeOWeightPercent_50N50S.iau2",
    "Lunar_Kaguya_MIMap_MineralDeconv_OlivinePercent_50N50S.iau2",
    "Lunar_Kaguya_MIMap_MineralDeconv_OpticalMaturityIndex_50N50S.iau2",
    "Lunar_Kaguya_MIMap_MineralDeconv_OrthopyroxenePercent_50N50S.iau2",
    "Lunar_Kaguya_MIMap_MineralDeconv_PlagioclaseGrainSizeMicrons_50N50S.iau2",
    "Lunar_Kaguya_MIMap_MineralDeconv_PlagioclasePercent_50N50S.iau2",
    "kaguya_mi_derived_30ppd_mpfe.iau",
    "kaguya_mi_derived_30ppd_npfe.iau",
    "kaguya_mi_derived_30ppd_smfe.iau",
    "Lunar_Kaguya_MIMap_Band1_MV1_414nm_65N65S_512ppd.iau2",
    "Lunar_Kaguya_MIMap_Band2_MV2_749nm_65N65S_512ppd.iau2",
    "Lunar_Kaguya_MIMap_Band3_MV3_901nm_65N65S_512ppd.iau2",
    "Lunar_Kaguya_MIMap_Band4_MV4_950nm_65N65S_512ppd.iau2",
    "Lunar_Kaguya_MIMap_Band5_MV5_1001nm_65N65S_512ppd.iau2",
    "Lunar_Kaguya_MIMap_Band7_MN2_1049nm_65N65S_512ppd.iau2",
    "Lunar_Kaguya_MIMap_Band8_MN3_1248nm_65N65S_512ppd.iau2",
    "Lunar_Kaguya_MIMap_Band9_MN4_1548nm_65N65S_512ppd.iau2",
    "lola_kaguya_60mpp_asp",
    "lola_kaguya_60mpp_cos",
    "lola_kaguya_60mpp_elv",
    "lola_kaguya_60mpp_sin",
    "lola_kaguya_60mpp_slp",
)

# Every band in a written static LTM cube uses this destination sentinel.
STATIC_OUTPUT_NODATA = -32768.0

# These are source-only overrides. They are masked during the warp and
# converted to STATIC_OUTPUT_NODATA in the destination cube.
MINIRF_SOURCE_NODATA = -3.4028230607370965e38
MINIRF_SOURCE_NODATA_BANDS = (
    "GlobeNoPolesDeltaCPR_v2-offsetto49d.iau",
    "GlobeNoPolesDeltaS1_v2.iau",
)


__all__ = [
    "MINIRF_SOURCE_NODATA",
    "MINIRF_SOURCE_NODATA_BANDS",
    "STATIC_BAND_NAMES",
    "STATIC_OUTPUT_NODATA",
]
