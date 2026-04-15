
from pathlib import Path


NAMES = [
    'lola_kaguya_60mpp.asp.vrt',
    'lola_kaguya_60mpp.cos.vrt',
    'lola_kaguya_60mpp.elv.vrt',
    'lola_kaguya_60mpp.sin.vrt',
    'lola_kaguya_60mpp.slp.vrt',
    'mini_rf/GlobeNoPolesDeltaCPR_v2-offsetto49d.iau.tif',
    'mini_rf/GlobeNoPolesDeltaS1_v2.iau.tif',
    'mini_rf/NorthDeltaCPR_mean80n_v5-offsetto49d.iau.tif',
    'mini_rf/NorthDeltaS1_mean80n_v5.iau.tif',
    'mini_rf/SouthDeltaCPR_mean80s_v3-offsetto49d.iau.tif',
    'mini_rf/SouthDeltaS1_mean80s_v3.iau.tif',
    'LROC/WAC/wac_emp/WAC_EMP_321NM.iau.tif',
    'LROC/WAC/wac_emp/WAC_EMP_360NM.iau.tif',
    'LROC/WAC/wac_emp/WAC_EMP_415NM.iau.tif',
    'LROC/WAC/wac_emp/WAC_EMP_566NM.iau.tif',
    'LROC/WAC/wac_emp/WAC_EMP_604NM.iau.tif',
    'LROC/WAC/wac_emp/WAC_EMP_643NM_304P.merge.iau.tif',
    'LROC/WAC/wac_emp/WAC_EMP_643NM.iau.tif',
    'LROC/WAC/wac_emp/WAC_EMP_689NM.iau.tif',
    'LROC/WAC/wac_glob_morf_mos/WAC_GLOBAL.iau.tif',
    'LROC/WAC/wac_glob_morf_mos/WAC_GLOBAL_P900N0000_100M.eqc.iau2.tif',
    'LROC/WAC/wac_glob_morf_mos/WAC_GLOBAL_P900S0000_100M.eqc.iau2.tif',
    'LROC/WAC/wac_titan/WAC_TIO2.iau.tif',
    'LP/LP_hydrogen_north.iau.tif',
    'LP/LP_hydrogen_south.iau.tif',
    'LOLA/albedo/LDAM_50N_1000M_FLOAT.iau.tif',
    'LOLA/albedo/LDAM_50S_1000M_FLOAT.iau.tif',
    'Kaguya/MI/mineralogy/Lunar_Kaguya_MIMap_MineralDeconv_ClinopyroxenePercent_50N50S.iau2.tif',
    'Kaguya/MI/mineralogy/Lunar_Kaguya_MIMap_MineralDeconv_FeOWeightPercent_50N50S.iau2.tif',
    'Kaguya/MI/mineralogy/Lunar_Kaguya_MIMap_MineralDeconv_OlivinePercent_50N50S.iau2.tif',
    'Kaguya/MI/mineralogy/Lunar_Kaguya_MIMap_MineralDeconv_OpticalMaturityIndex_50N50S.iau2.tif',
    'Kaguya/MI/mineralogy/Lunar_Kaguya_MIMap_MineralDeconv_OrthopyroxenePercent_50N50S.iau2.tif',
    'Kaguya/MI/mineralogy/Lunar_Kaguya_MIMap_MineralDeconv_PlagioclaseGrainSizeMicrons_50N50S.iau2.tif',
    'Kaguya/MI/mineralogy/Lunar_Kaguya_MIMap_MineralDeconv_PlagioclasePercent_50N50S.iau2.tif',
    'Kaguya/MI/space_weathering/kaguya_mi_derived_30ppd_mpfe.iau.tif',
    'Kaguya/MI/space_weathering/kaguya_mi_derived_30ppd_npfe.iau.tif',
    'Kaguya/MI/space_weathering/kaguya_mi_derived_30ppd_smfe.iau.tif',
    'Kaguya/MI/multispectral/Lunar_Kaguya_MIMap_Band1_MV1_414nm_65N65S_512ppd.iau2.tif',
    'Kaguya/MI/multispectral/Lunar_Kaguya_MIMap_Band2_MV2_749nm_65N65S_512ppd.iau2.tif',
    'Kaguya/MI/multispectral/Lunar_Kaguya_MIMap_Band3_MV3_901nm_65N65S_512ppd.iau2.tif',
    'Kaguya/MI/multispectral/Lunar_Kaguya_MIMap_Band4_MV4_950nm_65N65S_512ppd.iau2.tif',
    'Kaguya/MI/multispectral/Lunar_Kaguya_MIMap_Band5_MV5_1001nm_65N65S_512ppd.iau2.tif',
    'Kaguya/MI/multispectral/Lunar_Kaguya_MIMap_Band7_MN2_1049nm_65N65S_512ppd.iau2.tif',
    'Kaguya/MI/multispectral/Lunar_Kaguya_MIMap_Band8_MN3_1248nm_65N65S_512ppd.iau2.tif',
    'Kaguya/MI/multispectral/Lunar_Kaguya_MIMap_Band9_MN4_1548nm_65N65S_512ppd.iau2.tif',
    'Kaguya/SP/mineralogy/gridded_feo_mosaic_north_pole.iau2.tif',
    'Kaguya/SP/mineralogy/gridded_feo_mosaic_south_pole.iau2.tif',
        'Kaguya/SP/mineralogy/gridded_high_calcium_pyroxene_mosaic_north_pole.iau2.tif',
        'Kaguya/SP/mineralogy/gridded_high_calcium_pyroxene_mosaic_south_pole.iau2.tif',
        'Kaguya/SP/mineralogy/gridded_low_calcium_pyroxene_mosaic_north_pole.iau2.tif',
        'Kaguya/SP/mineralogy/gridded_low_calcium_pyroxene_mosaic_south_pole.iau2.tif',
        'Kaguya/SP/mineralogy/gridded_nanophase_iron_mosaic_north_pole.iau2.tif',
        'Kaguya/SP/mineralogy/gridded_nanophase_iron_mosaic_south_pole.iau2.tif',
    'Kaguya/SP/mineralogy/gridded_olivine_mosaic_north_pole.iau2.tif',
    'Kaguya/SP/mineralogy/gridded_olivine_mosaic_south_pole.iau2.tif',
    'Kaguya/SP/mineralogy/gridded_omat_mosaic_north_pole.iau2.tif',
    'Kaguya/SP/mineralogy/gridded_omat_mosaic_south_pole.iau2.tif',
    'Kaguya/SP/mineralogy/gridded_plagioclase_mosaic_north_pole.iau2.tif',
    'Kaguya/SP/mineralogy/gridded_plagioclase_mosaic_south_pole.iau2.tif',
    'gravity//jggrx_1800f_me_dist_meters_20km_cog.tif',
    'gravity//outputNorthPole_20km.tif',
    'gravity//outputSouthPole_20km.tif',
    'Diviner/Dice/polar_north_80_zit-100.iau.tif',
    'Diviner/Dice/polar_south_80_zit-100.iau.tif',
    'Diviner/Hpar/hpar_global128ppd_v1c_dateline_cut.iau3.tif',
    'Diviner/Rock/RA_SAM_70Sto70N.iau7.tif',
    'Diviner/Tbol/diviner_tbol_snapshot_000E.tif',   
    'Diviner/Tbol/diviner_tbol_snapshot_015E.tif',
    'Diviner/Tbol/diviner_tbol_snapshot_030E.tif',
    'Diviner/Tbol/diviner_tbol_snapshot_045E.tif',
    'Diviner/Tbol/diviner_tbol_snapshot_060E.tif',
    'Diviner/Tbol/diviner_tbol_snapshot_075E.tif',
    'Diviner/Tbol/diviner_tbol_snapshot_090E.tif',
    'Diviner/Tbol/diviner_tbol_snapshot_105E.tif',
    'Diviner/Tbol/diviner_tbol_snapshot_120E.tif',
    'Diviner/Tbol/diviner_tbol_snapshot_135E.tif',
    'Diviner/Tbol/diviner_tbol_snapshot_150E.tif',
    'Diviner/Tbol/diviner_tbol_snapshot_165E.tif',
    'Diviner/Tbol/diviner_tbol_snapshot_180E.tif',
    'Diviner/Tbol/diviner_tbol_snapshot_195E.tif',
    'Diviner/Tbol/diviner_tbol_snapshot_210E.tif',
    'Diviner/Tbol/diviner_tbol_snapshot_225E.tif',
    'Diviner/Tbol/diviner_tbol_snapshot_240E.tif',
    'Diviner/Tbol/diviner_tbol_snapshot_255E.tif',
    'Diviner/Tbol/diviner_tbol_snapshot_270E.tif',
    'Diviner/Tbol/diviner_tbol_snapshot_285E.tif',
    'Diviner/Tbol/diviner_tbol_snapshot_300E.tif',
    'Diviner/Tbol/diviner_tbol_snapshot_315E.tif',
    'Diviner/Tbol/diviner_tbol_snapshot_330E.tif',
    'Diviner/Tbol/diviner_tbol_snapshot_345E.tif',
    'Diviner/Tbol/polar_north_80_summer_tbol-slon01.iau.tif',
    'Diviner/Tbol/polar_north_80_summer_tbol-slon02.iau.tif',
    'Diviner/Tbol/polar_north_80_summer_tbol-slon03.iau.tif',
    'Diviner/Tbol/polar_north_80_summer_tbol-slon04.iau.tif',
    'Diviner/Tbol/polar_north_80_summer_tbol-slon05.iau.tif',
    'Diviner/Tbol/polar_north_80_summer_tbol-slon06.iau.tif',
    'Diviner/Tbol/polar_north_80_summer_tbol-slon07.iau.tif',
    'Diviner/Tbol/polar_north_80_summer_tbol-slon08.iau.tif',
    'Diviner/Tbol/polar_north_80_summer_tbol-slon09.iau.tif',
    'Diviner/Tbol/polar_north_80_summer_tbol-slon10.iau.tif',
    'Diviner/Tbol/polar_north_80_summer_tbol-slon11.iau.tif',
    'Diviner/Tbol/polar_north_80_summer_tbol-slon12.iau.tif',
    'Diviner/Tbol/polar_north_80_summer_tbol-slon13.iau.tif',
    'Diviner/Tbol/polar_north_80_summer_tbol-slon14.iau.tif',
    'Diviner/Tbol/polar_north_80_summer_tbol-slon15.iau.tif',
    'Diviner/Tbol/polar_north_80_summer_tbol-slon16.iau.tif',
    'Diviner/Tbol/polar_north_80_summer_tbol-slon17.iau.tif',
    'Diviner/Tbol/polar_north_80_summer_tbol-slon18.iau.tif',
    'Diviner/Tbol/polar_north_80_summer_tbol-slon19.iau.tif',
    'Diviner/Tbol/polar_north_80_summer_tbol-slon20.iau.tif',
    'Diviner/Tbol/polar_north_80_summer_tbol-slon21.iau.tif',
    'Diviner/Tbol/polar_north_80_summer_tbol-slon22.iau.tif',
    'Diviner/Tbol/polar_north_80_summer_tbol-slon23.iau.tif',
    'Diviner/Tbol/polar_north_80_summer_tbol-slon24.iau.tif',
    'Diviner/Tbol/polar_north_80_winter_tbol-slon01.iau.tif',
    'Diviner/Tbol/polar_north_80_winter_tbol-slon02.iau.tif',
    'Diviner/Tbol/polar_north_80_winter_tbol-slon03.iau.tif',
    'Diviner/Tbol/polar_north_80_winter_tbol-slon04.iau.tif',
    'Diviner/Tbol/polar_north_80_winter_tbol-slon05.iau.tif',
    'Diviner/Tbol/polar_north_80_winter_tbol-slon06.iau.tif',
    'Diviner/Tbol/polar_north_80_winter_tbol-slon07.iau.tif',
    'Diviner/Tbol/polar_north_80_winter_tbol-slon08.iau.tif',
    'Diviner/Tbol/polar_north_80_winter_tbol-slon09.iau.tif',
    'Diviner/Tbol/polar_north_80_winter_tbol-slon10.iau.tif',
    'Diviner/Tbol/polar_north_80_winter_tbol-slon11.iau.tif',
    'Diviner/Tbol/polar_north_80_winter_tbol-slon12.iau.tif',
    'Diviner/Tbol/polar_north_80_winter_tbol-slon13.iau.tif',
    'Diviner/Tbol/polar_north_80_winter_tbol-slon14.iau.tif',
    'Diviner/Tbol/polar_north_80_winter_tbol-slon15.iau.tif',
    'Diviner/Tbol/polar_north_80_winter_tbol-slon16.iau.tif'
    'Diviner/Treg/TREG_ANOM_70Sto70N.iau7.tif'
]
    
# ------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------
def main():

    # Process command-line args.
    desc = 'Use this application create symbolic links to static files.'

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-i',
                        type=Path,
                        required=True,
                        help='The path to the input directory')
    
    parser.add_argument('-o',
                        type=Path,
                        required=True,
                        help='The path to the output directory')
    
    args = parser.parse_args()
    
    inputDir = args.i
    outputDir = args.o
    
    if not inputDir or not inputDir.exists() or not inputDir.is_dir():
        raise ValueError('Invalid input directory.')

    if not outputDir or not outputDir.exists() or not outputDir.is_dir():
        raise ValueError('Invalid output directory.')
        
    for name in cls.NAMES:
    
        fromPath: Path = inputDir / name
        toPath: Path = outputDir
        
        if not fromPath.exists():
            raise RuntimeError('Invalid input: ' + str(fromPath))
            
        fromPath.symlink_to(toPath, target_is_directory=True)
        
        break
            