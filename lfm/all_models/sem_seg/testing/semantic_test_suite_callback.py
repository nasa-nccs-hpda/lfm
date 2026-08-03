"""Lightning callback for semantic segmentation epoch test suites."""

from __future__ import annotations

from pathlib import Path

from lightning.pytorch.callbacks import Callback

from lfm.all_models.sem_seg.testing.semantic_test_suite import run_semantic_test_suite


class SemanticEpochTestSuiteCallback(Callback):
    """Run a semantic test suite at epoch end and save arrays/metrics."""

    def __init__(
        self,
        *,
        output_dir: Path,
        model_name: str,
        split: str,
        n_samples: int,
        every_n_epochs: int,
        ignore_index: int | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.split = split
        self.n_samples = n_samples
        self.every_n_epochs = every_n_epochs
        self.ignore_index = ignore_index

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch + 1
        if self.every_n_epochs <= 0 or epoch % self.every_n_epochs != 0:
            return
        aggregate, saved = run_semantic_test_suite(
            datamodule=trainer.datamodule,
            task=pl_module,
            output_dir=self.output_dir,
            model_name=self.model_name,
            split=self.split,
            n_samples=self.n_samples,
            suite_name=f"epoch_{epoch:03d}",
            epoch=epoch,
            ignore_index=self.ignore_index,
        )
        print(
            f"[{self.model_name}] epoch {epoch:03d} test suite: "
            f"F1={aggregate['foreground_f1']:.4f}, IoU={aggregate['iou']:.4f}, "
            f"samples={saved}",
            flush=True,
        )
