"""Utilities for the data behavior refactor audit notebook."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

RUNNER_CODE = r"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def tensor_summary(value, *, max_unique=20):
    tensor = value.detach().cpu()
    summary = {
        "kind": "tensor",
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
    }
    if tensor.numel() == 0:
        return summary
    numeric = tensor.float() if not tensor.is_floating_point() else tensor
    finite = torch.isfinite(numeric)
    summary["nonfinite_count"] = int((~finite).sum().item())
    if torch.any(finite):
        valid = numeric[finite]
        summary.update(
            {
                "min": round(float(valid.min().item()), 8),
                "max": round(float(valid.max().item()), 8),
                "mean": round(float(valid.mean().item()), 8),
                "std": round(float(valid.std(unbiased=False).item()), 8),
            }
        )
    if tensor.ndim <= 2 or tensor.numel() <= 4096:
        unique = torch.unique(tensor)
        if unique.numel() <= max_unique:
            summary["unique_values"] = [float(x) if tensor.is_floating_point() else int(x) for x in unique.tolist()]
        else:
            summary["unique_count"] = int(unique.numel())
            summary["unique_head"] = [float(x) if tensor.is_floating_point() else int(x) for x in unique[:max_unique].tolist()]
    return summary


def array_summary(value):
    return tensor_summary(torch.as_tensor(value))


def summarize(value, *, depth=0):
    if depth > 5:
        return {"kind": "max_depth", "repr": repr(type(value))}
    if isinstance(value, torch.Tensor):
        return tensor_summary(value)
    if isinstance(value, np.ndarray):
        out = array_summary(value)
        out["kind"] = "ndarray"
        return out
    if isinstance(value, (str, os.PathLike)):
        path = str(value)
        return {"kind": "path", "name": Path(path).name, "suffix": Path(path).suffix, "path": path}
    if isinstance(value, dict):
        return {str(key): summarize(val, depth=depth + 1) for key, val in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return {
            "kind": type(value).__name__,
            "length": len(value),
            "items": [summarize(item, depth=depth + 1) for item in list(value)[:8]],
        }
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return {"kind": type(value).__name__, "repr": repr(value)[:200]}


def get_dataset(datamodule, split):
    return getattr(datamodule, f"{split}_dataset", None)


def get_loader(datamodule, split):
    if split == "train":
        return datamodule.train_dataloader()
    if split == "val":
        return datamodule.val_dataloader()
    if split == "test":
        return datamodule.test_dataloader()
    raise ValueError(split)


def audit_case(repo_root, data_root, case, n_samples):
    sys.path.insert(0, str(repo_root))
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    print(f"[audit] importing {case['module']}.{case['class']}", file=sys.stderr, flush=True)
    module = importlib.import_module(case["module"])
    cls = getattr(module, case["class"])
    kwargs = dict(case.get("kwargs", {}))
    kwargs["data_root"] = str(data_root)
    print(f"[audit] constructing {case['name']}", file=sys.stderr, flush=True)
    dm = cls(**kwargs)
    result = {
        "case": case["name"],
        "module": case["module"],
        "class": case["class"],
        "datamodule_type": f"{type(dm).__module__}.{type(dm).__name__}",
        "weight_assignments": None,
        "splits": {},
    }
    try:
        print(f"[audit] setup {case['name']}", file=sys.stderr, flush=True)
        dm.setup(None)
    except Exception as exc:
        result["setup_error"] = {"type": type(exc).__name__, "message": str(exc)}
        return result
    print(f"[audit] setup complete {case['name']}", file=sys.stderr, flush=True)
    if hasattr(dm, "weight_assignments"):
        result["weight_assignments"] = getattr(dm, "weight_assignments")
    for split in ("train", "val", "test"):
        print(f"[audit] summarize {case['name']} {split}", file=sys.stderr, flush=True)
        split_result = {}
        dataset = get_dataset(dm, split)
        if dataset is None:
            result["splits"][split] = {"present": False}
            continue
        split_result["present"] = True
        split_result["dataset_type"] = f"{type(dataset).__module__}.{type(dataset).__name__}"
        try:
            split_result["dataset_len"] = int(len(dataset))
        except Exception as exc:
            split_result["dataset_len_error"] = {"type": type(exc).__name__, "message": str(exc)}
            result["splits"][split] = split_result
            continue
        samples = []
        for idx in range(min(n_samples, split_result["dataset_len"])):
            try:
                samples.append(summarize(dataset[idx]))
            except Exception as exc:
                samples.append({"sample_index": idx, "error": {"type": type(exc).__name__, "message": str(exc)}})
        split_result["samples"] = samples
        try:
            print(
                f"[audit] dataloader batch {case['name']} {split}",
                file=sys.stderr,
                flush=True,
            )
            random.seed(1000 + len(split))
            np.random.seed(1000 + len(split))
            torch.manual_seed(1000 + len(split))
            batch = next(iter(get_loader(dm, split)))
            split_result["batch"] = summarize(batch)
        except Exception as exc:
            split_result["batch_error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["splits"][split] = split_result
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--case-json", required=True)
    parser.add_argument("--n-samples", type=int, default=3)
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    data_root = Path(args.data_root).resolve()
    case = json.loads(args.case_json)
    with contextlib.redirect_stdout(sys.stderr):
        result = audit_case(repo_root, data_root, case, args.n_samples)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
"""


def build_audit_cases(
    target_size: int,
    *,
    image_glob: str = "*.tif",
    label_glob: str = "*_label.npz",
    image_suffix: str = "_input_wac_chip",
    label_suffix: str = "_label",
) -> list[dict[str, Any]]:
    """Return the default audit cases for an instance-label split dataset."""
    common_file_kwargs = {
        "image_glob": image_glob,
        "label_glob": label_glob,
        "image_suffix": image_suffix,
        "label_suffix": label_suffix,
    }
    return [
        {
            "name": "toy_semantic",
            "enabled": True,
            "base_module": (
                "lfm.toy_model.sem_seg.lightning_wrappers."
                "toy_sem_seg_from_instance_datamodule"
            ),
            "base_class": "ToySemSegFromInstanceDataModule",
            "oop_module": (
                "lfm.toy_model.sem_seg.lightning_wrappers."
                "toy_sem_seg_from_instance_datamodule"
            ),
            "oop_class": "ToySemSegFromInstanceDataModule",
            "kwargs": {
                "batch_size": 2,
                "num_workers": 0,
                "target_size": [target_size, target_size],
                "spatial_transform": "crop",
                "image_file_type": ".tif",
                "label_file_type": ".npz",
                "label_npz_key": "mask",
                "binarize_label": True,
                "image_suffix": image_suffix,
                "label_suffix": label_suffix,
                "normalize_inputs": False,
                "scale_inputs": True,
                "max_train_samples": 8,
                "max_val_samples": 8,
                "max_test_samples": 8,
            },
        },
        {
            "name": "toy_instance_mask2former",
            "enabled": True,
            "base_module": (
                "lfm.toy_model.inst_seg.lightning_wrappers."
                "toy_instance_seg_datamodule"
            ),
            "base_class": "ToyInstanceSegSplitDataModule",
            "oop_module": (
                "lfm.toy_model.inst_seg.lightning_wrappers."
                "toy_instance_seg_datamodule"
            ),
            "oop_class": "ToyInstanceSegSplitDataModule",
            "kwargs": {
                "batch_size": 2,
                "num_workers": 0,
                "target_size": target_size,
                "normalize_inputs": False,
                "scale_inputs": True,
                "mask_shift": [0, 0],
                "max_train_samples": 8,
                "max_val_samples": 8,
                "max_test_samples": 8,
                **common_file_kwargs,
            },
        },
        {
            "name": "toy_instance_mask_rcnn",
            "enabled": True,
            "base_module": (
                "lfm.toy_model.inst_seg.lightning_wrappers."
                "toy_dino_mask_rcnn_datamodule"
            ),
            "base_class": "ToyDinoMaskRCNNSplitDataModule",
            "oop_module": (
                "lfm.toy_model.inst_seg.lightning_wrappers."
                "toy_dino_mask_rcnn_datamodule"
            ),
            "oop_class": "ToyDinoMaskRCNNSplitDataModule",
            "kwargs": {
                "batch_size": 2,
                "num_workers": 0,
                "target_size": target_size,
                "normalize_inputs": False,
                "scale_inputs": True,
                "mask_shift": [0, 0],
                "max_train_samples": 8,
                "max_val_samples": 8,
                "max_test_samples": 8,
                **common_file_kwargs,
            },
        },
        {
            "name": "graha_semantic",
            "enabled": True,
            "base_module": "lfm.full_model.sem_seg.semantic_from_instance_datamodule",
            "base_class": "LunarSemanticFromInstanceDatamodule",
            "oop_module": "lfm.full_model.sem_seg.semantic_from_instance_datamodule",
            "oop_class": "LunarSemanticFromInstanceDatamodule",
            "kwargs": {
                "batch_size": 2,
                "num_workers": 0,
                "crop_size": target_size,
                "means": None,
                "stds": None,
                "binarize_mask": True,
                "max_train_samples": 8,
                "max_val_samples": 8,
                "max_test_samples": 8,
                "no_data_replace": 0.0,
                "no_label_replace": None,
                **common_file_kwargs,
            },
        },
        {
            "name": "graha_instance_object_detection",
            "enabled": True,
            "base_module": "lfm.full_model.inst_seg.instance_mask_datamodule",
            "base_class": "LunarObjectDetectionInstanceMaskDatamodule",
            "oop_module": "lfm.full_model.inst_seg.instance_mask_datamodule",
            "oop_class": "LunarObjectDetectionInstanceMaskDatamodule",
            "kwargs": {
                "batch_size": 2,
                "num_workers": 0,
                "crop_size": target_size,
                "means": None,
                "stds": None,
                "target_box_format": "xyxy",
                "max_train_samples": 8,
                "max_val_samples": 8,
                "max_test_samples": 8,
                "no_data_replace": 0.0,
                "no_label_replace": None,
                "mask_shift": [0, 0],
                **common_file_kwargs,
            },
        },
    ]


def enabled_audit_cases(audit_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [case for case in audit_cases if case.get("enabled", True)]


def write_runner(output_dir: Path) -> Path:
    runner_path = Path(output_dir) / "lfm_data_audit_runner.py"
    runner_path.write_text(RUNNER_CODE, encoding="utf-8")
    return runner_path


def run_repo_case(
    repo_label: str,
    repo_root: Path,
    data_root: Path,
    python_exe: str,
    case: dict[str, Any],
    runner_path: Path,
    n_samples: int,
    timeout_seconds: int | None = 300,
) -> dict[str, Any]:
    side = "base" if repo_label == "base" else "oop"
    kwargs = dict(case.get("kwargs", {}))
    kwargs.update(case.get(f"{side}_kwargs", {}))
    runner_case = {
        "name": case["name"],
        "module": case[f"{side}_module"],
        "class": case[f"{side}_class"],
        "kwargs": kwargs,
    }
    cmd = [
        str(python_exe),
        str(runner_path),
        "--repo-root",
        str(repo_root),
        "--data-root",
        str(data_root),
        "--case-json",
        json.dumps(runner_case),
        "--n-samples",
        str(n_samples),
    ]
    try:
        completed = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "repo_label": repo_label,
            "repo_root": str(repo_root),
            "case": case["name"],
            "returncode": "timeout",
            "timeout_seconds": timeout_seconds,
            "stderr_tail": (exc.stderr or "")[-4000:],
            "stdout_tail": (exc.stdout or "")[-4000:],
        }
    payload = {
        "repo_label": repo_label,
        "repo_root": str(repo_root),
        "case": case["name"],
        "returncode": completed.returncode,
        "stderr_tail": completed.stderr[-4000:],
    }
    if completed.returncode != 0:
        payload["stdout_tail"] = completed.stdout[-4000:]
        return payload
    try:
        payload["summary"] = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        payload["json_error"] = str(exc)
        payload["stdout_tail"] = completed.stdout[-4000:]
    return payload


def run_audit_cases(
    *,
    audit_cases: list[dict[str, Any]],
    base_repo_root: Path,
    oop_repo_root: Path,
    base_data_root: Path,
    oop_data_root: Path,
    base_python: str,
    oop_python: str,
    runner_path: Path,
    n_samples: int,
    timeout_seconds: int | None = 300,
) -> dict[str, dict[str, Any]]:
    audit_results: dict[str, dict[str, Any]] = {"base": {}, "oop": {}}
    for case in audit_cases:
        print(f"running {case['name']} base...")
        audit_results["base"][case["name"]] = run_repo_case(
            "base",
            base_repo_root,
            base_data_root,
            base_python,
            case,
            runner_path,
            n_samples,
            timeout_seconds,
        )
        print(f"running {case['name']} oop...")
        audit_results["oop"][case["name"]] = run_repo_case(
            "oop",
            oop_repo_root,
            oop_data_root,
            oop_python,
            case,
            runner_path,
            n_samples,
            timeout_seconds,
        )
    return audit_results


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    rows = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            rows.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            rows.update(flatten(value, f"{prefix}[{index}]"))
    else:
        rows[prefix] = obj
    return rows


def values_match(a: Any, b: Any, *, atol: float = 1e-6) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= atol
        except Exception:
            return False
    return a == b


def compare_audit_results(
    audit_results: dict[str, dict[str, Any]],
    audit_cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    comparison_rows = []
    for case in audit_cases:
        name = case["name"]
        base_payload = audit_results["base"][name]
        oop_payload = audit_results["oop"][name]
        if base_payload.get("returncode") != 0 or oop_payload.get("returncode") != 0:
            comparison_rows.append(
                {
                    "case": name,
                    "path": "SUBPROCESS_STATUS",
                    "base": base_payload.get("returncode"),
                    "oop": oop_payload.get("returncode"),
                    "match": False,
                }
            )
            continue
        base_summary = base_payload.get("summary", {})
        oop_summary = oop_payload.get("summary", {})
        base_flat = flatten(base_summary)
        oop_flat = flatten(oop_summary)
        all_keys = sorted(set(base_flat) | set(oop_flat))
        for key in all_keys:
            if key in {"module", "class", "datamodule_type"} or key.endswith(
                ".dataset_type"
            ):
                continue
            base_value = base_flat.get(key, "<MISSING>")
            oop_value = oop_flat.get(key, "<MISSING>")
            match = values_match(base_value, oop_value)
            if not match:
                comparison_rows.append(
                    {
                        "case": name,
                        "path": key,
                        "base": base_value,
                        "oop": oop_value,
                        "match": match,
                    }
                )
    return comparison_rows
