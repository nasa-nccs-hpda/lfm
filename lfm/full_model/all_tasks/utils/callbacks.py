"""Lightning callbacks used by full-model plotting workflows."""

from __future__ import annotations

from pathlib import Path

from lightning.pytorch.callbacks import Callback

from lfm.full_model.sem_seg.semantic_plotting import plot_validation_predictions


class ValidationPlotCallback(Callback):
    """Save a lightweight val pred plot at the end of each epoch."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        n_samples: int = 5,
        every_n_epochs: int = 1,
        plots_subdir: str | Path = "plots",
        display_method: str = "minmax",
        dpi: int = 150,
    ) -> None:

        self.output_dir = Path(output_dir)
        self.n_samples = n_samples
        self.every_n_epochs = every_n_epochs
        self.plots_subdir = Path(plots_subdir)
        self.display_method = display_method
        self.dpi = dpi

    def on_validation_epoch_end(self, trainer, pl_module) -> None:

        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch
        if self.every_n_epochs <= 0 or (epoch + 1) % self.every_n_epochs != 0:
            return
        if trainer.datamodule is None:
            return

        plot_validation_predictions(
            task=pl_module,
            datamodule=trainer.datamodule,
            output_dir=self.output_dir,
            n_samples=self.n_samples,
            filename=f"validation_epoch_{epoch + 1:03d}.png",
            plots_subdir=self.plots_subdir,
            display_method=self.display_method,
            dpi=self.dpi,
            setup_datamodule=False,
        )
