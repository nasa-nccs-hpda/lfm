"""Lightning callbacks for instance segmentation test-suite workflows."""

from __future__ import annotations

from pathlib import Path

from lightning.pytorch.callbacks import Callback

from lfm.all_models.inst_seg.instance_test_suite import run_instance_test_suite
from lfm.all_models.all_tasks.utils import (
    plot_instance_cache_predictions,
    save_graha_instance_prediction_cache,
)


class GrahaInstancePlotCallback(Callback):
    """Save Graha instance validation plots at epoch end."""

    def __init__(
        self,
        output_dir: Path,
        *,
        n_samples: int,
        every_n_epochs: int,
        score_threshold: float,
    ) -> None:
        self.output_dir = output_dir
        self.n_samples = n_samples
        self.every_n_epochs = every_n_epochs
        self.score_threshold = score_threshold

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch
        if self.every_n_epochs <= 0 or (epoch + 1) % self.every_n_epochs != 0:
            return
        cache_dir = save_graha_instance_prediction_cache(
            task=pl_module,
            datamodule=trainer.datamodule,
            output_dir=self.output_dir,
            model_name="graha",
            split="val",
            n_samples=self.n_samples,
            score_threshold=self.score_threshold,
            setup_datamodule=False,
        )
        plot_instance_cache_predictions(
            cache_dir,
            self.output_dir / "plots" / "single_model" / "full_model",
            model_name="graha",
            n_samples=self.n_samples,
            filename=f"validation_epoch_{epoch + 1:03d}.png",
        )


class InstanceEpochTestSuiteCallback(Callback):
    """Run an instance test suite at epoch end and save arrays/metrics."""

    def __init__(
        self,
        *,
        output_dir: Path,
        model_name: str,
        split: str,
        n_samples: int,
        every_n_epochs: int,
        score_threshold: float,
        image_processor=None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.split = split
        self.n_samples = n_samples
        self.every_n_epochs = every_n_epochs
        self.score_threshold = score_threshold
        self.image_processor = image_processor

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch + 1
        if self.every_n_epochs <= 0 or epoch % self.every_n_epochs != 0:
            return
        run_instance_test_suite(
            task=pl_module,
            datamodule=trainer.datamodule,
            output_dir=self.output_dir,
            model_name=self.model_name,
            split=self.split,
            n_samples=self.n_samples,
            suite_name=f"epoch_{epoch:03d}",
            score_threshold=self.score_threshold,
            epoch=epoch,
            image_processor=self.image_processor,
        )
