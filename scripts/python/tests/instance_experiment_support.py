"""Local task adapters used by experiment integration checks."""

from types import SimpleNamespace

from lfm.all_models.inst_seg import training


def route_tasks(monkeypatch, builder):
    def resolve(model, task):
        assert task == "instance"
        return SimpleNamespace(
            create_recorded_task=lambda config, dm: builder(config, dm, model)
        )

    monkeypatch.setattr(training, "resolve_model_adapter", resolve)
