# Copyright 2024 EPFL and Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import hashlib
import os

from ast import literal_eval
from dataclasses import asdict, dataclass, field
from functools import partial, reduce
from pathlib import Path
from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf

from terramind.data.geomap_class_map import MAPPING_GEOMAP_CLASSES
from terramind.data.io_utils import convert_to_object
from terramind.data import modality_transforms as mt
from terramind.models.decoder_embeddings import ImageTokenDecoderEmbedding, SequenceDecoderEmbedding
from terramind.models.encoder_embeddings import (
    ImageEncoderEmbedding,
    ImageTokenEncoderEmbedding,
    SequenceEncoderEmbedding,
)


TOKENIZATION_CONFIG_FILE = "tokenization_config.yaml"
MODALITY_INFO_FILE = "modality_info.yaml"

STATIC_MAPS_FIELDS = (
    "ALBEDO",
    "AVG_ILLUM",
    "DICE",
    "GRAVITY",
    "HPAR",
    "HYDROGEN",
    "ROCK_ABUNDANCE",
    "ROUGHNESS",
    "SP_MINERALOGY_feo",
    "SP_MINERALOGY_high_calcium_pyroxene",
    "SP_MINERALOGY_low_calcium_pyroxene",
    "SP_MINERALOGY_nanophase_iron",
    "SP_MINERALOGY_olivine",
    "SP_MINERALOGY_omat",
    "SP_MINERALOGY_plagioclase",
    "SW_FE_mpfe",
    "SW_FE_npfe",
    "SW_FE_smfe",
    "TBOL_POLES_closest",
    "TBOL_closest",
    "TIO2",
    "TREG",
    "WAC_NORM_REF_321",
    "WAC_NORM_REF_360",
    "WAC_NORM_REF_415",
    "WAC_NORM_REF_566",
    "WAC_NORM_REF_604",
    "WAC_NORM_REF_643",
    "WAC_NORM_REF_689",
)


@dataclass
class ModalityInfoImg:
    """Information for untokenized image modalities.

    Args:
        encoder_embedding: encoder embedding class.
        encoder_kwargs: kwargs for encoder embedding initialization.
        decoder_embedding: decoder embedding class (None for raw images).
        decoder_kwargs: kwargs for decoder embedding initialization.
        type: Modality type ("img").
        pretokenized: whether modality is pretokenized.
        data_range: expected data range.
        one_hot_encoding: number of classes for one-hot encoded modalities.
    """
    encoder_embedding: Any | None = ImageEncoderEmbedding
    encoder_kwargs: dict = field(default_factory=dict)
    decoder_embedding: Any | None = None
    decoder_kwargs: dict = field(default_factory=dict)
    type: Literal["img"] = "img"
    pretokenized: bool = False
    data_range: tuple[float, float] | None = None
    one_hot_encoding: int | None = None


@dataclass
class ModalityInfoImgTokenized(ModalityInfoImg):
    """Information for tokenized image modalities."""
    encoder_embedding: Any | None = ImageTokenEncoderEmbedding
    decoder_embedding: Any | None = ImageTokenDecoderEmbedding
    pretokenized: bool = True


@dataclass
class ModalityInfoSeq:
    """Information for sequence modalities."""
    encoder_embedding: type = SequenceEncoderEmbedding
    encoder_kwargs: dict = field(default_factory=dict)
    decoder_embedding: type = SequenceDecoderEmbedding
    decoder_kwargs: dict = field(default_factory=dict)
    type: Literal["seq", "seq_emb", "seq_token"] = "seq"
    image_size: int | None = None  # Only relevant for bboxes - image size bboxes are based on optical modality


def _generate_uint15_hash(seed_str):
    """Generates a hash of the seed string as an unsigned int15 integer."""
    return int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest(), 16) % (2**15)


def compute_modality_id(mod_name: str, modality_info: dict) -> int:
    """Compute a unique, deterministic ID for a modality based on its name and key properties.

    Args:
        mod_name: The modality name (e.g., "tok_dtm").
        modality_info: The modality_info dict built at runtime.

    Returns:
        A uint15 hash (0-32767) unique to this modality configuration
    """
    id_components = [mod_name]  # Start with modality name only and add configurable properties

    if modality_info.get("pretokenized") is not None:
        id_components += ["c" + str(modality_info.get("codebook_size", "x")),
                          "n" + str(modality_info.get("num_codebooks", "x"))]

    if modality_info["type"] == "img":
        id_components += ["p" + str(modality_info.get("patch_size", "")),
                          "i" + str(modality_info.get("input_size", "")),
                          str(modality_info.get("tokenizer", "untok"))]

    id_string = "-".join(id_components)
    hash_value = _generate_uint15_hash(id_string)

    return hash_value


MODALITY_INFO = {
    # image modalities - pixel
    "vis": asdict(ModalityInfoImg(data_range=(0.0, 1.0))),
    "uv": asdict(ModalityInfoImg(data_range=(0.0, 1.0))),
    "dtm": asdict(ModalityInfoImg(data_range=(-9500.0, 10800.0))),
    "slope": asdict(ModalityInfoImg(data_range=(0.0, 90.0))),
    "aspect": asdict(ModalityInfoImg(data_range=(-1.0, 1.0))),
    "geomap": asdict(ModalityInfoImg(one_hot_encoding=16)),
    "crater_masks": asdict(ModalityInfoImg()),
    "aspect_3m": asdict(ModalityInfoImg(data_range=(-1.0, 1.0))),
    "dtm_3m": asdict(ModalityInfoImg(data_range=(-9500.0, 10800.0))),
    "nac": asdict(ModalityInfoImg(data_range=(0., 1.))),
    # "psr": asdict(ModalityInfoImg()),
    "slope_3m": asdict(ModalityInfoImg(data_range=(0.0, 90.0))),
    # "rock_abundance": asdict(ModalityInfoImg()),
    # "treg": asdict(ModalityInfoImg()),
    # "hpar": asdict(ModalityInfoImg()),
    # "tio2": asdict(ModalityInfoImg()),
    # "roughness": asdict(ModalityInfoImg()),
    # "minirf_cpr": asdict(ModalityInfoImg()),
    # "minirf_s1": asdict(ModalityInfoImg()),
    "wac_mosaic": asdict(ModalityInfoImg()),
    # "mi_mineralogy": asdict(ModalityInfoImg()),
    # "mi_norm_ref": asdict(ModalityInfoImg()),
    # "sw_fe": asdict(ModalityInfoImg()),
    # "wac_norm_ref": asdict(ModalityInfoImg()),
    # "wac_nr_643_hr": asdict(ModalityInfoImg()),
    # "sp_mineralogy": asdict(ModalityInfoImg()),
    # "avg_illum": asdict(ModalityInfoImg()),
    # "albedo": asdict(ModalityInfoImg()),
    # "dice": asdict(ModalityInfoImg()),
    # "tbol_poles": asdict(ModalityInfoImg()),

    # Sequence modalities

    "metadata": asdict(ModalityInfoSeq(
        encoder_kwargs={"vocab_size": 62_917},
        decoder_kwargs={"vocab_size": 62_917},
    )),

    "static_maps": asdict(ModalityInfoSeq(
        encoder_kwargs={"vocab_size": 62_917},
        decoder_kwargs={"vocab_size": 62_917},
    )),

    "crater_bboxes": asdict(ModalityInfoSeq(
        image_size=512,
        encoder_kwargs={"vocab_size": 62_917},
        decoder_kwargs={"vocab_size": 62_917},
    )),

    # Tokenized image modalities
    "tok_vis": asdict(ModalityInfoImgTokenized()),
    "tok_uv": asdict(ModalityInfoImgTokenized()),
    "tok_dtm": asdict(ModalityInfoImgTokenized()),
    "tok_slope": asdict(ModalityInfoImgTokenized()),
    "tok_aspect": asdict(ModalityInfoImgTokenized()),
    "tok_geomap": asdict(ModalityInfoImgTokenized()),
    "tok_vis_604": asdict(ModalityInfoImgTokenized()),
    "tok_aspect_3m": asdict(ModalityInfoImgTokenized()),
    "tok_dtm_3m": asdict(ModalityInfoImgTokenized()),
    "tok_nac": asdict(ModalityInfoImgTokenized()),
    "tok_psr": asdict(ModalityInfoImgTokenized()),
    "tok_slope_3m": asdict(ModalityInfoImgTokenized()),
    # "tok_rock_abundance": asdict(ModalityInfoImgTokenized()),
    # "tok_treg": asdict(ModalityInfoImgTokenized()),
    # "tok_hpar": asdict(ModalityInfoImgTokenized()),
    # "tok_tio2": asdict(ModalityInfoImgTokenized()),
    # "tok_roughness": asdict(ModalityInfoImgTokenized()),
    # "tok_minirf_cpr": asdict(ModalityInfoImgTokenized()),
    # "tok_minirf_s1": asdict(ModalityInfoImgTokenized()),
    "tok_wac_mosaic": asdict(ModalityInfoImgTokenized()),
    # "tok_mi_mineralogy": asdict(ModalityInfoImgTokenized()),
    # "tok_mi_norm_ref": asdict(ModalityInfoImgTokenized()),
    # "tok_sw_fe": asdict(ModalityInfoImgTokenized()),
    # "tok_wac_norm_ref": asdict(ModalityInfoImgTokenized()),
    # "tok_wac_nr_643_hr": asdict(ModalityInfoImgTokenized()),
    # "tok_sp_mineralogy": asdict(ModalityInfoImgTokenized()),
    # "tok_avg_illum": asdict(ModalityInfoImgTokenized()),
    # "tok_albedo": asdict(ModalityInfoImgTokenized()),
    # "tok_dice": asdict(ModalityInfoImgTokenized()),
    # "tok_tbol_poles": asdict(ModalityInfoImgTokenized()),
}

MODALITY_TRANSFORMS = {
    "mask_valid": partial(mt.MaskTransform, mask_pool_size=1),

    # Lunar untokenized modalities
    "vis": mt.UntokLunarTransform,
    "vis_604": mt.UntokLunarTransform,
    "uv": mt.UntokLunarTransform,
    "dtm": mt.UntokLunarTransform,
    "slope": mt.UntokLunarTransform,
    "aspect": mt.UntokLunarTransform,
    "geomap": partial(mt.UntokGeomapTransform, class_mapping=MAPPING_GEOMAP_CLASSES),
    "crater_masks": mt.UntokCraterMasksTransform,
    "aspect_3m": mt.UntokLunarTransform,
    "dtm_3m": mt.UntokLunarTransform,
    "nac": mt.UntokLunarTransform,
    # "psr": mt.UntokLunarTransform,
    "slope_3m": mt.UntokLunarTransform,
    # "rock_abundance": mt.UntokLunarTransform,
    # "treg": mt.UntokLunarTransform,
    # "hpar": mt.UntokLunarTransform,
    # "tio2": mt.UntokLunarTransform,
    # "roughness": mt.UntokLunarTransform,
    # "minirf_cpr": mt.UntokLunarTransform,
    # "minirf_s1": mt.UntokLunarTransform,
    "wac_mosaic": mt.UntokLunarTransform,
    # "mi_mineralogy": mt.UntokLunarTransform,
    # "mi_norm_ref": mt.UntokLunarTransform,
    # "sw_fe": mt.UntokLunarTransform,
    # "wac_norm_ref": mt.UntokLunarTransform,
    # "wac_nr_643_hr": mt.UntokLunarTransform,
    # "sp_mineralogy": mt.UntokLunarTransform,
    # "avg_illum": mt.UntokLunarTransform,
    # "albedo": mt.UntokLunarTransform,
    # "dice": mt.UntokLunarTransform,
    # "tbol_poles": mt.UntokLunarTransform,

    # Lunar tokenized transforms
    "tok_vis": mt.TokTransform,
    "tok_vis_604": mt.TokTransform,
    "tok_uv": mt.TokTransform,
    "tok_dtm": mt.TokTransform,
    "tok_slope": mt.TokTransform,
    "tok_aspect": mt.TokTransform,
    "tok_geomap": mt.TokTransform,
    "tok_crater_masks": mt.TokTransform,
    "tok_aspect_3m": mt.TokTransform,
    "tok_dtm_3m": mt.TokTransform,
    "tok_nac": mt.TokTransform,
    # "tok_psr": mt.TokTransform,
    "tok_slope_3m": mt.TokTransform,
    # "tok_rock_abundance": mt.TokTransform,
    # "tok_treg": mt.TokTransform,
    # "tok_hpar": mt.TokTransform,
    # "tok_tio2": mt.TokTransform,
    # "tok_roughness": mt.TokTransform,
    # "tok_minirf_cpr": mt.TokTransform,
    # "tok_minirf_s1": mt.TokTransform,
    "tok_wac_mosaic": mt.TokTransform,
    # "tok_mi_mineralogy": mt.TokTransform,
    # "tok_mi_norm_ref": mt.TokTransform,
    # "tok_sw_fe": mt.TokTransform,
    # "tok_wac_norm_ref": mt.TokTransform,
    # "tok_wac_nr_643_hr": mt.TokTransform,
    # "tok_sp_mineralogy": mt.TokTransform,
    # "tok_avg_illum": mt.TokTransform,
    # "tok_albedo": mt.TokTransform,
    # "tok_dice": mt.TokTransform,
    # "tok_tbol_poles": mt.TokTransform,

    # Text modalities
    "metadata": mt.MetadataTransform,
    "static_maps": partial(mt.StaticMapsTransform, selected_vars=list(STATIC_MAPS_FIELDS)),
    "crater_bboxes": mt.BBoxTransform,
}

MODALITY_TRANSFORMS_TOK = {
    "vis": mt.LunarTransform,
    "vis_604": mt.LunarTransform,
    "uv": mt.LunarTransform,
    "dtm": mt.LunarTransform,
    "slope": mt.LunarTransform,
    "aspect": mt.LunarTransform,
    "geomap": partial(mt.GeomapTransform, class_mapping=MAPPING_GEOMAP_CLASSES),
    "crater_masks": mt.CraterMasksTransform,
    "aspect_3m": mt.LunarTransform,
    "dtm_3m": mt.LunarTransform,
    "nac": mt.LunarTransform,
    # "psr": mt.LunarTransform,
    "slope_3m": mt.LunarTransform,
    # "rock_abundance": mt.LunarTransform,
    # "treg": mt.LunarTransform,
    # "hpar": mt.LunarTransform,
    # "tio2": mt.LunarTransform,
    # "roughness": mt.LunarTransform,
    # "minirf_cpr": mt.LunarTransform,
    # "minirf_s1": mt.LunarTransform,
    "wac_mosaic": mt.LunarTransform,
    # "mi_mineralogy": mt.LunarTransform,
    # "mi_norm_ref": mt.LunarTransform,
    # "sw_fe": mt.LunarTransform,
    # "wac_norm_ref": mt.LunarTransform,
    # "wac_nr_643_hr": mt.LunarTransform,
    # "sp_mineralogy": mt.LunarTransform,
    # "avg_illum": mt.LunarTransform,
    # "albedo": mt.LunarTransform,
    # "dice": mt.LunarTransform,
    # "tbol_poles": mt.LunarTransform,

}


def setup_modality_transform(
        domains: list[str], stats: dict | DictConfig,
        scaler_dict: dict | DictConfig | None = None,
        pre_resize: int | None = None,
    ):

    scaler_dict = {} if scaler_dict is None else convert_to_object(scaler_dict)
    stats = convert_to_object(stats)
    modality_transform = {
        mod: MODALITY_TRANSFORMS_TOK[mod](**stats[mod],
                                          one_hot_encoding=MODALITY_INFO[mod].get("one_hot_encoding"),
                                          pre_resize=pre_resize,
                                          scaler=scaler_dict.get(mod))
        for mod in domains
    }
    return modality_transform


def setup_modality_transform_tm(domains: list[str], modality_info: dict):
    modality_transform = dict()
    for mod in domains:
        is_tokenized = modality_info[mod].get("pretokenized", False)
        mod_type = modality_info[mod]["type"]

        if mod_type == "img":
            if is_tokenized:
                modality_transform[mod] = MODALITY_TRANSFORMS[mod](num_codebooks=modality_info[mod]["num_codebooks"])
            else:
                stats = modality_info[mod]["stats"]
                scaler_dict = modality_info[mod]["scaler_dict"]
                scaler_dict = {} if scaler_dict is None else scaler_dict
                pre_resize = modality_info[mod]["pre_resize"]
                modality_transform[mod] = MODALITY_TRANSFORMS[mod](
                    **stats[mod],
                    one_hot_encoding=modality_info[mod].get("one_hot_encoding"),
                    pre_resize=pre_resize,
                    scaler=scaler_dict.get(mod),
                )
        # Special case for craters that needs optical modality info
        elif mod == "crater_bboxes":
            vis_info = modality_info.get("tok_vis") or modality_info.get("vis")
            if vis_info is None:
                raise ValueError("Modality *crater_bboxes* needs optical *vis* info to be properly loaded.")
            modality_transform[mod] = MODALITY_TRANSFORMS[mod](original_size=modality_info[mod]["image_size"],
                                                               pre_resize=vis_info["pre_resize"])
        else:
            modality_transform[mod] = MODALITY_TRANSFORMS[mod]()
    return modality_transform


def _load_tokenization_config(data_root: str, mod_path: str, base_mod_dict: dict, add_tokenizer_info: bool) -> dict:
    """Load and parse tokenization config for a pretokenized modality."""

    yaml_path = os.path.join(data_root, mod_path, TOKENIZATION_CONFIG_FILE)
    token_info = OmegaConf.load(yaml_path)

    input_size = token_info.input_size
    patch_size = token_info.patch_size
    num_patches = (input_size // patch_size) ** 2
    parent_domain = token_info.domain
    num_channels = len(token_info.stats[parent_domain]["channels"])

    # Handle one-hot encoding (changes number of channels)
    one_hot_encoding = base_mod_dict[parent_domain].get("one_hot_encoding")
    if one_hot_encoding is not None:
        num_channels = one_hot_encoding

    # Parse codebook_size - handle list and string formats
    codebook_size = token_info.codebook_size
    if isinstance(codebook_size, str):
        if codebook_size.startswith("[") and codebook_size.endswith("]"):
            codebook_size = literal_eval(codebook_size)   # heterogeneous codebooks: "[8640, 1024]"
        else:
            codebook_size = reduce(lambda x, y: x * y, [int(x) for x in codebook_size.split("-")], 1)  # "8-8-8-6-5"

    token_dict = {"input_size": input_size,
                  "patch_size": patch_size,
                  "num_channels": num_channels,
                  "codebook_size": codebook_size,
                  "num_codebooks": token_info.num_codebooks,
                  "max_tokens": num_patches,
                  "parent_domain": parent_domain,
                  "tokenizer": token_info.tokenizer,
                  "pre_resize": token_info.pre_resize,
                  "crop_settings": convert_to_object(token_info.crop_settings),   # Convert ListConfig to list object
                  "scaler_dict": convert_to_object(token_info.scaler_dict),
                  "stats": convert_to_object(token_info.stats)}

    # Add tokenizer paths if requested
    if add_tokenizer_info:
        token_dict["tokenizer_weights"] = token_info.get("ckpt_path")
        token_dict["tokenizer_config"] = token_info.get("cfg_path")

    return token_dict


def setup_modality_info_tm(cfg: DictConfig, add_tokenizer_info: bool = False) -> tuple[dict, int, list]:
    """Setup modality info for TerraMind by merging fixed and variable properties.

    Args:
        cfg: DictConfig of the format JobConfigTM.
        add_tokenizer_info: if True, add tokenizer and config paths from tokenization_config.yaml to "img" modalities.
            Useful when loading model with tokenizers.

    Returns:
        dict of modality_info and loaded common input size.
    """

    def _check_dict_equality(dict_list: list[dict], keys: list[str] | None):
        keys = keys if keys is not None else list(dict_list[0].keys())
        for k in keys:
            value_list = [d[k] for d in dict_list]
            assert all(s == value_list[0] for s in value_list[1:]), f"All modalities must have the same {k}."

    modality_info = {}
    dict_check = []
    parent_mod_info = {}

    for dataset_cfg in cfg.data.train_datasets.values():
        for mod_name, mod_cfg in dataset_cfg.domains.items():

            # Skip if modality was already setup or if not in all_domains
            if (mod_name in modality_info) or (mod_name not in cfg.data.all_domains):
                continue

            modality_info[mod_name] = {
                **MODALITY_INFO[mod_name],  # Fixed properties
                "path": mod_cfg.path,
                "min_tokens": mod_cfg.min_tokens,
                "max_tokens": mod_cfg.max_tokens,
            }

            if MODALITY_INFO[mod_name]["type"] in ["seq", "seq_emb", "seq_token"]:
                modality_info[mod_name]["keep"] = mod_cfg.keep

            # Load tokenization info for pretokenized image modalities
            if (MODALITY_INFO[mod_name]["type"] == "img") and MODALITY_INFO[mod_name].get("pretokenized"):
                # Load tokenization config
                token_config = _load_tokenization_config(data_root=dataset_cfg.data_root,
                                                         mod_path=mod_cfg.path,
                                                         base_mod_dict=MODALITY_INFO,
                                                         add_tokenizer_info=add_tokenizer_info)
                parent_domain = token_config["parent_domain"]

                # Update tokenized modality info
                modality_info[mod_name].update(token_config)

                common_params = ["input_size", "pre_resize", "crop_settings", "scaler_dict", "stats", "num_channels"]
                common_dict = {k: token_config[k] for k in common_params}
                dict_check.append(common_dict)

                # Save parameters for parent modality (if it's in all_domains)
                if parent_domain in cfg.data.all_domains:
                    add_params = {}
                    if dataset_cfg.domains.get(parent_domain, {}).get("max_tokens") is None:
                        add_params["max_tokens"] = token_config["max_tokens"]
                    ps = dataset_cfg.domains.get(parent_domain, {}).get("patch_size")
                    add_params["patch_size"] = token_config["patch_size"] if ps is None else ps

                    parent_mod_info[parent_domain] = {**common_dict, **add_params}

    # Apply updates to untokenized (parent) modalities
    for mod_name, mod_info in parent_mod_info.items():
        modality_info[mod_name].update(mod_info)

    # Add ID based on modality properties
    for mod_name, mod_info in modality_info.items():
        mod_info["id"] = compute_modality_id(mod_name=mod_name, modality_info=mod_info)

    # Sort modalities as in all_domains
    modality_info = {mod: modality_info[mod] for mod in cfg.data.all_domains}

    # Ensure all modalities have the same input size and crop settings at the moment
    # TODO: generalize for multiple image sizes
    common_input_size = dict_check[0]["input_size"]
    common_crop_settings = dict_check[0]["crop_settings"]
    _check_dict_equality(dict_list=dict_check, keys=["input_size", "crop_settings"])

    return modality_info, common_input_size, common_crop_settings


def save_modality_info(modality_info: dict, output_dir: str):
    serialized = {}
    for mod, info in modality_info.items():
        serialized[mod] = {}
        for k, v in info.items():
            if k in ["encoder_embedding", "decoder_embedding"] and v is not None:
                serialized[mod][k] = f"{v.__module__}.{v.__name__}"  # Class to string
            elif isinstance(v, (str, int, float, bool, list, tuple, dict, type(None))):
                serialized[mod][k] = v

    OmegaConf.save(OmegaConf.create(serialized), Path(output_dir) / MODALITY_INFO_FILE)
    print(f"Saved modality_info to {Path(output_dir) / MODALITY_INFO_FILE}")
