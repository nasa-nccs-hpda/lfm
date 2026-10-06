"""Notebook entry points keep the tested checkout and label-output conventions."""

import ast
import json
from pathlib import Path
import sys

import pytest

from lfm.data_processing._paths import REPO_ROOT


def notebook_cells():
    notebook = json.loads((REPO_ROOT / 'notebooks/crater_labeling.ipynb').read_text())
    return [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']


@pytest.mark.parametrize('working_directory', ['', 'notebooks'])
def test_setup_finds_restructured_checkout(tmp_path, monkeypatch, working_directory):
    marker = tmp_path / 'lfm/data_processing/labeling/craters.py'
    marker.parent.mkdir(parents=True)
    marker.touch()
    (tmp_path / 'notebooks').mkdir()
    monkeypatch.chdir(tmp_path / working_directory)
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    tree = ast.parse(notebook_cells()[0])
    # Exercise actual bootstrap without constructing a live widget application.
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module != 'pathlib')]
    namespace = {}
    exec(compile(tree, 'notebook_setup', 'exec'), namespace)
    assert namespace['repo_root'] == tmp_path
    assert namespace['NOTEBOOK_DIR'] == tmp_path / 'notebooks'
    assert sys.path[0] == str(tmp_path)


def test_configuration_preserves_labels_and_explicit_catalog_import(tmp_path):
    calls = []
    class Labeler:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.widget = object()
        def _guard(self, function):
            function()
        def load(self):
            pass
    namespace = dict(Path=Path, repo_root=tmp_path, NOTEBOOK_DIR=tmp_path / 'notebooks',
                     CraterLabeler=Labeler, display=lambda widget: None)
    exec(compile(notebook_cells()[1], 'notebook_configuration', 'exec'), namespace)
    assert calls[0]['output_dir'] == tmp_path / 'notebooks/outputs/labels'
    assert calls[0]['default_catalog'] == tmp_path / 'lfm/data_processing/tests/labeling/fixtures/robbins_scene.csv'
    # Choosing a catalog only supplies a default; the user explicitly imports it.
    assert not hasattr(namespace['dashboard'], 'import_catalog')
    full = tmp_path / 'data/catalogs/lunar_crater_database_robbins_2018.csv'
    full.parent.mkdir(parents=True)
    full.touch()
    namespace.pop('dashboard')
    exec(compile(notebook_cells()[1], 'notebook_configuration', 'exec'), namespace)
    assert calls[-1]['default_catalog'] == full
