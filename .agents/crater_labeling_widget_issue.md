# Crater labeling notebook: widget display issue

## Observed behavior

The crater labeling notebook intermittently displays:

```text
Error displaying widget: model not found
Exception opening new comm
Could not send widget sync message Error: Cannot send
```

The user reports this particularly after pulling notebook changes and restarting the kernel. Reloading the browser page and rerunning the notebook resolves it. Clearing console logs was also part of the workaround, but we have not isolated which steps are necessary.

No root cause has been confirmed, and no widget-lifecycle fix has been implemented.

## Likely causes, ranked

1. **Stale browser widget state after kernel restart — strongest hypothesis.** Widgets have corresponding Python objects and browser models. A restart removes the Python objects, while existing outputs can still reference old model IDs. Recovery after page reload fits this explanation. Jupyter's [widget-state documentation](https://github.com/jupyter-widgets/ipywidgets/blob/main/docs/source/embedding.md) recommends restarting the kernel and then refreshing the page to clear frontend widget state.

2. **Widget replacement/cleanup timing — worth investigating.** In `lfm/data_processing/labeling/craters.py`, `_draw()` replaces every accepted-crater polygon widget and immediately closes the old ones. `_remove_handles()` similarly removes and closes markers. This creates substantial model creation/destruction traffic; a frontend resolving references during replacement is a plausible failure point, not a confirmed bug.

3. **Incomplete cleanup when rerunning the dashboard cell.** `notebooks/crater_labeling.ipynb` already calls `dashboard.close()` before creating another dashboard. However, `CraterLabeler.close()` does not explicitly dispose of every owned child widget and observer. Audit this for same-kernel reruns; it does not itself explain everything following a kernel restart.

4. **Frontend/backend package mismatch or stale frontend assets.** Still possible, but lower priority because reloading resolves the issue without reinstalling anything.

The symptoms point to ipywidgets/ipyleaflet lifecycle or communication problems, not a label/GeoPackage problem. There is no indication that `tqdm` is involved.

## Suggested diagnostic checks

- Determine whether the first error occurs before running a cell, while creating the dashboard, or during editing.
- Compare a fresh browser reload followed by Run All against rerunning only the dashboard cell.
- Capture the first console exception, including any model ID or module name. Later “Cannot send” errors may be consequences rather than the initial failure.
- Record kernel-side `ipywidgets`, `ipyleaflet`, and `ipykernel` versions, plus server-side JupyterLab/widget-extension versions. The server and kernel may use different environments.
- Audit polygon replacement and complete dashboard teardown before attempting broad dependency changes.

## Current workaround

Save labels and confirm the save succeeded. Clear notebook outputs, restart the kernel, reload the browser page, and rerun the notebook. This is a workaround, not a verified fix; do not discard unsaved edits while testing it.

## Relevant code

- `notebooks/crater_labeling.ipynb`: closes the previous dashboard, creates a new `CraterLabeler`, and displays its widget.
- `lfm/data_processing/labeling/craters.py`: `CraterLabeler._draw()`, `_remove_handles()`, and `close()` manage the widget lifecycle under investigation.

This handoff reflects the code inspected on 2026-10-06 and the user's reported browser behavior; the browser failure has not been reproduced locally.
