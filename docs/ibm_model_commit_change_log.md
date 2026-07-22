# ibm_model Branch Commit Change Log

This document summarizes commits after `7d06a4d6e3ae4dec571f67d6b24530dde39175fc` in the requested range ending at `7b1b964a2d19a1484f9a6dc5221db556b00cdbe2`. The first 10 entries preserve the original review-draft style; the continuation starting at commit 10 adds more file-level detail so the history is easier to use as branch notes.

## Size Classification Rubric

- `[tiny]`: Very small change, usually one file and only a few lines, or a simple add/delete with limited behavioral impact.
- `[small]`: Small focused change touching one or two files, usually documentation, notebook configuration, or a narrow bug fix.
- `[moderate]`: Meaningful behavior or workflow change, or notebook-heavy changes where line counts are noisy but the conceptual change is still focused.
- `[large]`: Broad refactor or many files changed, often thousands of lines moved/removed/added, with repo-structure or workflow impact.
- `[very large]`: Repository-scale restructure, large merge, or thousands of changed lines across many files with broad implications.

## Commit 1: c07f01f6bc090ecb9d9f48b34592ce880d589b5f

- Date: 2026-06-30 09:41:04 -0400
- Subject: fixing chip example label globbing
- Size: `[small]`
- Files changed: `notebooks/chip_example.ipynb`
- Shortstat: 1 file changed, 29 insertions, 149 deletions
- Summary: This commit updates the chip creation/example notebook, specifically around how label files are discovered or matched by globbing. The line count is somewhat misleading because notebook JSON often changes many serialized lines for a small conceptual notebook edit.
- Technical details: The important behavior appears to be a correction to label globbing in `chip_example.ipynb`, likely reducing incorrect or overly broad label matching while running the example chip workflow. This matters because the project relies heavily on matching image chips and labels by filename/product/tile identifiers, and small globbing mistakes can silently produce wrong image-label pairings.
- Impact: Improves correctness of the chip example workflow. This is an early data-preparation hygiene fix rather than a model architecture change.
- Risk or follow-up: Notebook-only changes are harder to review in raw Git diff. If this logic became important outside an example, it would be better to move matching/globbing behavior into a Python helper with tests.

## Commit 2: 9326fcf3cf03ce4d9728bd4831cf209d59d75f5a

- Date: 2026-07-07 15:13:46 -0400
- Subject: adding Ethan's suggestions
- Size: `[small]`
- Files changed: `README.md`, `notebooks/semantic_seg_train.ipynb`
- Shortstat: 2 files changed, 25 insertions, 3 deletions
- Summary: This commit updates user-facing documentation and the semantic segmentation training notebook with feedback from Ethan. The visible README changes clarify JupyterHub setup and add a new FAQ section for common notebook interaction problems.
- Technical details: The README quickstart wording was adjusted to refer to the user's `nobackup` space more plainly. A FAQ section was added describing how to collapse/expand notebook outputs and how to switch cells between Code and Markdown. The model/data specification section also received a small syntax/link correction around the Mask2Former reference.
- Impact: Improves onboarding for users running the toy workflows in JupyterHub. The semantic notebook was touched, but the available summary suggests this was likely usability or presentation-oriented rather than a major model change.
- Risk or follow-up: The README still mixes operational quickstart material with model/data specifications. As workflows grow, separating user quickstart, developer notes, and experiment documentation would make the repo easier to navigate.

## Commit 3: c307b00673f14e1961c35ab310f8d7d3b8931aa9

- Date: 2026-07-09 12:13:27 -0400
- Subject: restructure initial commit
- Size: `[tiny]`
- Files changed: `planning.txt`
- Shortstat: 1 file changed, 24 insertions
- Summary: This commit adds a short planning document for an upcoming repo restructure.
- Technical details: The commit does not change runtime code. It appears to capture a temporary planning artifact used to organize the later move from the older `lfm/tasks` layout toward a clearer full-model/toy-model split.
- Impact: No direct runtime impact. It is process/context for the restructure that follows.
- Risk or follow-up: Temporary planning files are useful during refactors but can create repo noise if committed and then removed immediately, which happened in the next commit.

## Commit 4: bfe955c3c7028081b73866764c581a843e4b3ada

- Date: 2026-07-09 12:15:05 -0400
- Subject: Delete planning.txt
- Size: `[tiny]`
- Files changed: `planning.txt`
- Shortstat: 1 file changed, 24 deletions
- Summary: This commit removes the temporary `planning.txt` file added in the previous commit.
- Technical details: No runtime code changed. This effectively cleans up the planning artifact before the restructure is implemented in code.
- Impact: No direct runtime impact.
- Risk or follow-up: None technically. From a history perspective, this pair of commits shows planning churn before the restructure.

## Commit 5: 7b7dbe49496c078d6fcffc9058d9a984068d1885

- Date: 2026-07-09 12:16:57 -0400
- Subject: adding .gitignore changes
- Size: `[tiny]`
- Files changed: `.gitignore`
- Shortstat: 1 file changed, 2 insertions, 1 deletion
- Summary: This commit updates ignored files, adding `planning.txt` and preserving existing ignores for notebook checkpoints, outputs, Python cache directories, and `notebooks/lfm`.
- Technical details: The practical change is adding `planning.txt` to `.gitignore`, likely to avoid re-committing temporary planning notes after the prior add/delete pair.
- Impact: Reduces accidental commits of local planning artifacts. No runtime effect.
- Risk or follow-up: It may be better long-term to keep durable planning docs under `docs/` and ignore only scratch notes or local working files.

## Commit 6: 4d75cb0aff4248e32bddf9656d2b9b7772f506c1

- Date: 2026-07-09 12:21:09 -0400
- Subject: moving folders to support full/toy model
- Size: `[very large]`
- Files changed: 20 files
- Shortstat: 20 files changed, 6396 insertions, 14 deletions
- Summary: This is the first major repo-structure commit in the range. It introduces a new `lfm/full_model` namespace and a new `lfm/toy_model` namespace, then populates the toy model package with semantic segmentation, instance segmentation, and shared helper code.
- Technical details: New package files include `lfm/full_model/__init__.py`, `lfm/toy_model/__init__.py`, `lfm/toy_model/all_tasks/all_utils.py`, semantic segmentation modules under `lfm/toy_model/sem_seg`, and instance segmentation modules under `lfm/toy_model/inst_seg`. The semantic package includes `sseg_dataset.py`, `sseg_driver.py`, `sseg_model.py`, `sseg_utils.py`, and `data_cube_inference.py`. The instance package includes `iseg_dataset.py`, `iseg_driver.py`, `iseg_model.py`, and `iseg_utils.py`.
- Technical details: Notebook references were also updated in `chip_example.ipynb`, `inference_sseg.ipynb`, `instance_seg_train.ipynb`, `semantic_seg_train.ipynb`, and `tiling_example.ipynb`, likely to point imports at the new toy-model package locations.
- Impact: Establishes the conceptual split that later work depends on: "toy model" code lives separately from "full model" code. This matters because later Graha/Lunar-FM work needs a home that is not confused with the simpler DINO toy workflows.
- Impact: This commit makes the repo more extensible for comparisons between the original toy workflows and the imported Graha/full-model workflows.
- Risk or follow-up: Because this commit appears to add the toy-model package before deleting the old `lfm/tasks` tree, there is a temporary duplication window. Imports may be ambiguous until the old package is removed. Large moves like this are also easy to review incorrectly if Git treats them as add/delete instead of rename.

## Commit 7: ae328e374651fa719547ceea82bc3146aa36da45

- Date: 2026-07-09 12:27:52 -0400
- Subject: fixing torch.hub load error
- Size: `[tiny]`
- Files changed: `lfm/toy_model/inst_seg/iseg_model.py`, `lfm/toy_model/sem_seg/sseg_model.py`
- Shortstat: 2 files changed, 1 insertion, 3 deletions
- Summary: This commit removes `force_reload=True` from DINOv3 `torch.hub.load()` calls in both toy instance and toy semantic model loaders.
- Technical details: In `lfm/toy_model/inst_seg/iseg_model.py`, `force_reload=True` was removed from the local-checkpoint encoder load path. In `lfm/toy_model/sem_seg/sseg_model.py`, the same argument was removed from the DINOv3 encoder load path.
- Impact: This likely avoids repeated GitHub/hub reload behavior and fixes a torch.hub loading failure in the working environment. Removing forced reload also makes repeated notebook/script runs faster and less dependent on network/cache behavior.
- Risk or follow-up: If the cached DINOv3 repo becomes stale or corrupt, users may need a manual cache-clearing or explicit reload path. For normal HPC/Jupyter use, avoiding forced reload is the safer default.

## Commit 8: 040c9dd487af3a37a834bfef9405a392365b2754

- Date: 2026-07-09 12:35:01 -0400
- Subject: removing old tasks/ subdir
- Size: `[large]`
- Files changed: 14 files
- Shortstat: 14 files changed, 6378 deletions
- Summary: This commit removes the old `lfm/tasks` package after the toy-model code has been moved into `lfm/toy_model`.
- Technical details: Deleted paths include `lfm/tasks/all_tasks`, `lfm/tasks/inst_segmentation`, and `lfm/tasks/sem_segmentation`. The removed files correspond closely to the files added under `lfm/toy_model` in commit `4d75cb0`, including dataset, driver, model, utility, and data-cube inference modules.
- Impact: Completes the repo restructure by eliminating the old task namespace. This reduces ambiguity about where toy semantic and instance segmentation code should live.
- Impact: The branch now has a cleaner path for later adding full-model/Graha code without overloading the old `tasks` directory.
- Risk or follow-up: Any notebook, script, or downstream user code still importing from `lfm.tasks.*` would break after this commit. The prior notebook import updates reduce that risk, but external users may need migration guidance.

## Commit 9: 7f5de136e40134ffbcd390840071dfd4e86c6d65

- Date: 2026-07-09 12:40:56 -0400
- Subject: Merge pull request #15 from nasa-nccs-hpda/feature/repo_restructure
- Size: `[very large]`
- Files changed: 35 files
- Shortstat: 35 files changed, 6396 insertions, 6393 deletions
- Summary: This is a merge commit for the repo restructure pull request. The net effect is repository-wide restructuring around the new toy-model layout, with large add/delete counts reflecting files moving out of the old task namespace and into the new package structure.
- Technical details: Because this is a merge commit, it should be interpreted as integration of the restructure branch rather than a standalone logical code change. The meaningful changes are mostly represented by the preceding commits: adding `lfm/toy_model`, adding `lfm/full_model`, updating notebook imports, fixing torch.hub loading, and removing the old `lfm/tasks` package.
- Impact: Marks the point where the restructure work is merged into the branch history. After this point, the toy model package layout is the baseline structure for later work.
- Risk or follow-up: Merge commits can obscure file-level authorship and make change review noisy. For future technical summaries, the underlying non-merge commits are more useful than the merge commit itself.

## Commit 10: 7c2c11b96503a8d3276b213aabcdaecbf0442a3c

- Date: 2026-07-15 15:46:57 -0400
- Subject: Merge remote-tracking branch 'origin/develop' into ibm_model merging develop to incorporate new repo structure
- Size: `[very large]`
- Files changed: 36 files
- Shortstat: 36 files changed, 6450 insertions, 6545 deletions
- Summary: This merge brings `origin/develop` into `ibm_model` to incorporate the new repo structure and resolve branch divergence before continuing the IBM/Graha work.
- Technical details: The merge reports conflicts or combined changes in notebooks including `notebooks/instance_seg_train.ipynb`, `notebooks/semantic_seg_train.ipynb`, and `notebooks/tiling_example.ipynb`. The large insertion/deletion counts indicate that the branch was being aligned with the restructured repo state from develop.
- Impact: Establishes `ibm_model` on top of the current develop/restructure baseline. This is important because most later work assumes the reorganized package structure and no longer targets the older `lfm/tasks` layout.
- Risk or follow-up: Notebook merge conflicts are difficult to inspect manually because raw `.ipynb` diffs are noisy. After a merge like this, the important validation is that the notebooks still run and import from the intended package paths.
- Risk or follow-up: This commit is likely a boundary between the initial repo restructure work and the later full-model/Graha comparison work. It is a useful landmark in the branch history.

## File-Level Continuation From Commit 10 Onward

The entries below continue from the 10th commit and use more explicit per-file or per-file-group notes. For very large vendored-tree moves, related files are grouped by directory because the technical change is the move itself rather than distinct edits inside every TerraMind source file.

## Commit 10 Addendum: 7c2c11b96503a8d3276b213aabcdaecbf0442a3c

- Date: 2026-07-15
- Subject: Merge remote-tracking branch `origin/develop` into `ibm_model` merging develop to incorporate new repo structure
- Size: `[very large]`
- Per-file changes:
  - `notebooks/instance_seg_train.ipynb`: Merge-touched the toy instance segmentation notebook while reconciling the develop restructure with branch-local notebook changes.
  - `notebooks/semantic_seg_train.ipynb`: Merge-touched the toy semantic segmentation notebook, preserving it as the original toy workflow baseline for later comparison work.
  - `notebooks/tiling_example.ipynb`: Merge-touched the tiling example notebook during the same repo-structure alignment.
- Technical note: This is mostly an integration checkpoint, not a clean feature commit; the useful review lens is whether the notebooks still import from the restructured package locations.

## Commit 11: 4ca54782ca1cb7931a21ed41911c2c94c67a7883

- Date: 2026-07-15
- Subject: reorganizing repo
- Size: `[very large]`
- Per-file changes:
  - `README.md`: Updated repo-level documentation to match the new project organization.
  - `graha-lunar-fm/*` to `lfm/full_model/graha-lunar-fm/*`: Moved the copied Graha/TerraMind code under the `lfm/full_model` package tree so it is no longer a top-level sibling of the repo code.
  - `graha-lunar-fm/terramind/**` to `lfm/full_model/graha-lunar-fm/terramind/**`: Moved TerraMind model, tokenizer, VQ, data, and utility modules without changing file contents.
  - `graha-lunar-fm/terratorch_integration/**` to `lfm/full_model/graha-lunar-fm/terratorch_integration/**`: Moved the Lunar-FM/TerraTorch integration code, including configs, data adapters, Lunar backbone/task registration, necks, and prediction writer code.
  - `graha-lunar-fm/.git.backup/**` to `lfm/full_model/graha-lunar-fm/.git.backup/**`: Moved a copied Git metadata backup along with the folder, which was later removed as cleanup.
  - `notebooks/graha-flm-finetuning/*` to `notebooks/full_model/*`: Renamed the Graha fine-tuning notebook area into the more general `full_model` notebook area.
  - `notebooks/*.ipynb` to `notebooks/toy_model/*.ipynb`: Moved original toy notebooks into a toy-model notebook folder.
- Technical note: This commit establishes the high-level structure that later work relies on: full/Graha code under `lfm/full_model`, toy workflows under `lfm/toy_model` and `notebooks/toy_model`, and active full-model notebooks under `notebooks/full_model`.
- Risk or follow-up: The `.git.backup` move was repo noise and was correctly removed soon after.

## Commit 12: d977473a75c397302399b62ba197d9f65f4d81e1

- Date: 2026-07-15
- Subject: moving code for full model
- Size: `[moderate]`
- Per-file changes:
  - `notebooks/full_model/datamodule.py` to `lfm/full_model/datamodule.py`: Moved the notebook-local full-model datamodule into package code so scripts and notebooks can import the same implementation.
  - `notebooks/full_model/datamodule_utils.py` to `lfm/full_model/datamodule_utils.py`: Moved shared datamodule helpers out of notebooks.
  - `notebooks/full_model/lfm_seg_finetuning_direct.py` to `lfm/full_model/lfm_seg_finetuning_direct.py`: Moved direct fine-tuning task/script support into package code, with some edits during the move.
  - `notebooks/full_model/plot_utils.py` to `lfm/full_model/plot_utils.py`: Moved plotting helpers into package code.
  - `notebooks/full_model/utils.py` to `lfm/full_model/utils.py`: Moved general utility functions into package code.
  - `notebooks/full_model/cuda_error.txt`: Removed an old captured error log from the notebook directory.
  - `notebooks/full_model/lfm_seg_finetuning_direct.ipynb`: Updated notebook imports/usages after moving Python helpers into `lfm/full_model`.
  - `notebooks/full_model/sbatch_sem_seg_finetune.sh`: Updated the shell wrapper to use the moved code paths.
- Technical note: This is the first step toward keeping notebooks as user-facing interfaces while reusable logic lives in importable Python modules.

## Commit 13: 17bf851e09e5174a532f88877c28ee26f7792052

- Date: 2026-07-15
- Subject: adding lightning wrappers to old code + reorganize
- Size: `[large]`
- Per-file changes:
  - `lfm/full_model/datamodules/__init__.py`: Added a package initializer for full-model datamodule imports.
  - `lfm/full_model/datamodule.py` to `lfm/full_model/datamodules/datamodule.py`: Moved the full-model datamodule into a dedicated datamodules package.
  - `lfm/full_model/datamodule_utils.py` to `lfm/full_model/datamodules/datamodule_utils.py`: Moved datamodule utilities alongside datamodule code.
  - `lfm/full_model/lfm_seg_finetuning_direct.py`: Updated imports after moving datamodule and utility modules.
  - `lfm/full_model/utils/__init__.py`: Added exports for full-model utility imports.
  - `lfm/full_model/plot_utils.py` to `lfm/full_model/utils/plot_utils.py`: Moved plotting helpers into the new utilities package.
  - `lfm/full_model/utils.py` to `lfm/full_model/utils/utils.py`: Moved general helpers into the utilities package.
  - `lfm/toy_model/sem_seg/lightning_wrappers/__init__.py`: Added a package for Lightning wrappers around the existing toy semantic segmentation code.
  - `lfm/toy_model/sem_seg/lightning_wrappers/toy_sem_seg_comparison.py`: Added comparison-oriented wrapper code for using the toy model in the same experiment style as the full model.
  - `lfm/toy_model/sem_seg/lightning_wrappers/toy_sem_seg_datamodule.py`: Added a Lightning datamodule wrapper around the toy semantic segmentation dataset.
  - `lfm/toy_model/sem_seg/lightning_wrappers/toy_sem_seg_lightning.py`: Added the Lightning module wrapper for training/evaluating the toy semantic segmentation model.
  - `notebooks/full_model/lfm_seg_finetuning_direct.ipynb`: Updated notebook imports and flow after the package reorganization.
  - `notebooks/full_model/model_comparison_plan.txt`: Updated the comparison plan to include the new wrapper direction.
  - `notebooks/full_model/sbatch_toy_sem_seg_comparison.sh`: Added a batch wrapper for the semantic comparison workflow.
- Technical note: This is the bridge from the older toy code to the Lightning/TerraTorch-style comparison workflow.

## Commit 14: 69c5eb0cd5bc0bb51c1d969c41d820d32443a060

- Date: 2026-07-15
- Subject: removing git backup files
- Size: `[large]`
- Per-file changes:
  - `lfm/full_model/graha-lunar-fm/.git.backup/**`: Deleted copied Git metadata backup files including `HEAD`, `config`, hooks, object pack files, refs, and index data.
- Technical note: This removes accidental repository metadata that should not be part of the source tree.

## Commit 15: 743c63b7faba781d480b80185bb653e0b7457950

- Date: 2026-07-15
- Subject: adding comparison notebook/script
- Size: `[moderate]`
- Per-file changes:
  - `lfm/toy_model/sem_seg/lightning_wrappers/toy_sem_seg_comparison.py`: Expanded the toy semantic comparison wrapper so it can support notebook/script comparison behavior.
  - `notebooks/full_model/toy_sem_seg_comparison.ipynb`: Added the first notebook for comparing the toy semantic model against the full/Graha model.
- Technical note: This commit creates the first concrete side-by-side comparison artifact.

## Commit 16: 7e99d11cb99c52a341695de3f172e639b4a55de0

- Date: 2026-07-15
- Subject: reorganizing wrappers
- Size: `[moderate]`
- Per-file changes:
  - `notebooks/full_model/sbatch_toy_sem_seg_comparison.sh`: Updated the shell entrypoint after moving the comparison script.
  - `notebooks/full_model/toy_sem_seg_comparison.ipynb`: Updated notebook imports to use the moved script location.
  - `lfm/toy_model/sem_seg/lightning_wrappers/toy_sem_seg_comparison.py` to `notebooks/full_model/toy_sem_seg_comparison.py`: Moved the comparison script back into the notebook area as a user-specific experiment script rather than reusable package code.
- Technical note: This reflects a boundary decision: wrappers remain package code, but the active comparison script is treated as an experiment artifact.

## Commit 17: 0e01687254400df675edc4e2af540585b1fb4080

- Date: 2026-07-15
- Subject: updating comparison plan
- Size: `[small]`
- Per-file changes:
  - `notebooks/full_model/model_comparison_plan.txt`: Updated the model comparison plan with the current refactor/comparison direction.

## Commit 18: 219f384173b831270605f1caba5a229779f5606e

- Date: 2026-07-15
- Subject: improving shell script deps
- Size: `[moderate]`
- Per-file changes:
  - `notebooks/full_model/sbatch_sem_seg_finetune.sh`: Improved dependency/setup behavior for the full-model semantic fine-tuning batch script.
  - `notebooks/full_model/sbatch_toy_sem_seg_comparison.sh`: Improved dependency/setup behavior for the toy/full semantic comparison batch script.
  - `notebooks/full_model/toy_sem_seg_comparison.py`: Updated comparison script behavior to align with the shell wrapper dependency handling.
- Technical note: This was part of making the scripts runnable on HPC rather than only inside an already-configured notebook kernel.

## Commit 19: 53bf05ad0779f48702523f97a5dd2f670860c615

- Date: 2026-07-15
- Subject: updating plan
- Size: `[small]`
- Per-file changes:
  - `notebooks/full_model/model_comparison_plan.txt`: Updated the comparison plan again as the implementation path changed.

## Commit 20: cb314be1e07c5f72682300324dc9b9fb1815b049

- Date: 2026-07-16
- Subject: changing toy model to crop to size
- Size: `[moderate]`
- Per-file changes:
  - `lfm/toy_model/sem_seg/lightning_wrappers/toy_sem_seg_datamodule.py`: Updated the Lightning datamodule wrapper to request cropping behavior and match the comparison target size.
  - `lfm/toy_model/sem_seg/sseg_dataset.py`: Changed the toy semantic dataset to crop images/labels to the requested size rather than resize them.
  - `notebooks/full_model/model_comparison_plan.txt`: Marked the crop alignment step in the comparison plan.
  - `notebooks/full_model/toy_sem_seg_comparison.ipynb`: Updated notebook configuration/flow for crop-based inputs.
  - `notebooks/full_model/toy_sem_seg_comparison.py`: Updated script configuration and datamodule setup for crop-based inputs.
- Technical note: This matters experimentally because resizing changes crater geometry, while center/consistent cropping preserves native pixel scale.

## Commit 21: 2f911623d1b0f78c066545b513d8eabd42ca0f9a

- Date: 2026-07-16
- Subject: limiting max_epochs in comparison notebook
- Size: `[small]`
- Per-file changes:
  - `notebooks/full_model/toy_sem_seg_comparison.ipynb`: Lowered/defaulted `max_epochs` for notebook smoke testing so the workflow can be validated quickly.

## Commit 22: 563276b2f1844b22dede3616cfffdbbc6065e75e

- Date: 2026-07-16
- Subject: making notebooks closer to original toy model
- Size: `[moderate]`
- Per-file changes:
  - `notebooks/full_model/lfm_seg_finetuning_direct.ipynb`: Adjusted the full-model fine-tuning notebook structure/content to better align with the original toy semantic notebook style.
  - `notebooks/full_model/toy_sem_seg_comparison.ipynb`: Reworked the comparison notebook so its config and flow feel closer to the original semantic segmentation training notebook.

## Commit 23: 706090afc2d4e1695e1bc88a3120319eb44c7971

- Date: 2026-07-16
- Subject: more notebook structural changes
- Size: `[moderate]`
- Per-file changes:
  - `notebooks/full_model/model_comparison_plan.txt`: Updated the plan to track notebook-structure work.
  - `notebooks/full_model/toy_sem_seg_comparison.ipynb`: Split/renamed cells and clarified notebook flow.
  - `notebooks/full_model/toy_sem_seg_comparison.py`: Kept the Python script aligned with the notebook changes.

## Commit 24: d6a717098d1b0b27051cfb724541607793a40257

- Date: 2026-07-16
- Subject: moving symlink behavior to full model utils
- Size: `[small]`
- Per-file changes:
  - `lfm/full_model/utils/__init__.py`: Exported the new symlink helper.
  - `lfm/full_model/utils/utils.py`: Added shared `./data` symlink resolution/validation behavior.
  - `notebooks/full_model/toy_sem_seg_comparison.ipynb`: Updated the notebook to import/use the shared symlink helper.
  - `notebooks/full_model/toy_sem_seg_comparison.py`: Updated the script to use the same symlink behavior as the notebook.
- Technical note: This avoids duplicated notebook/script symlink logic and makes the `SIMLINK_DEST` behavior consistent.

## Commit 25: d8619e801e80f999499fbd31d5d20ad64b6136f2

- Date: 2026-07-16
- Subject: fixing notebook import error
- Size: `[small]`
- Per-file changes:
  - `notebooks/full_model/toy_sem_seg_comparison.ipynb`: Fixed import usage after exposing `ensure_data_symlink`.
  - `notebooks/full_model/toy_sem_seg_comparison.py`: Kept script imports consistent with the fixed notebook path.

## Commit 26: 1899e986146cc37b4063fefd6262d71a6eff2689

- Date: 2026-07-16
- Subject: adding plotting to comparison notebook
- Size: `[moderate]`
- Per-file changes:
  - `lfm/full_model/utils/plot_utils.py`: Added semantic segmentation plotting helpers for validation/comparison outputs.
  - `notebooks/full_model/toy_sem_seg_comparison.ipynb`: Added notebook cells to generate validation plots.
  - `notebooks/full_model/toy_sem_seg_comparison.py`: Added matching script-side plotting behavior.

## Commit 27: 5f2a046fe5763af33372b631bb35ba34ba3452ce

- Date: 2026-07-16
- Subject: making notebook side-by-side
- Size: `[moderate]`
- Per-file changes:
  - `lfm/full_model/lfm_seg_finetuning_direct.py`: Adjusted full-model direct fine-tuning support for comparison usage.
  - `lfm/full_model/utils/__init__.py`: Exported additional plotting/comparison helpers.
  - `lfm/full_model/utils/plot_utils.py`: Added side-by-side semantic comparison plotting helpers.
  - `notebooks/full_model/model_comparison_plan.txt`: Updated the plan to include side-by-side comparison output.
  - `notebooks/full_model/toy_sem_seg_comparison.ipynb`: Added side-by-side visualization cells.
  - `notebooks/full_model/toy_sem_seg_comparison.py`: Added script support for producing comparable prediction artifacts.

## Commit 28: 8f44f3b26bd06d3c3f3deac1fcd776c4b975c703

- Date: 2026-07-16
- Subject: auto compare in notebook
- Size: `[small]`
- Per-file changes:
  - `notebooks/full_model/toy_sem_seg_comparison.ipynb`: Changed notebook flow so both models can run linearly and then compare cached predictions without manual reruns.

## Commit 29: 746779de9d59436422b2ef9b7497dc60cce2649f

- Date: 2026-07-16
- Subject: making notebook markdown better
- Size: `[small]`
- Per-file changes:
  - `notebooks/full_model/toy_sem_seg_comparison.ipynb`: Improved markdown headings and narrative flow, including clearer Graha vs DINO section labels.

## Commit 30: 145df4f20d83fc02bd4c891be74a61e07dbcc72c

- Date: 2026-07-16
- Subject: deleting dino-only viz cell
- Size: `[tiny]`
- Per-file changes:
  - `notebooks/full_model/toy_sem_seg_comparison.ipynb`: Removed redundant DINO-only visualization cells after side-by-side plotting was added.

## Commit 31: b5d25481e8cd11a5633f5aceeff91e3c61866765

- Date: 2026-07-16
- Subject: improving side-by-side plots
- Size: `[small]`
- Per-file changes:
  - `lfm/full_model/utils/plot_utils.py`: Adjusted semantic side-by-side plot layout/title spacing and filename display behavior.

## Commit 32: 80d7eb50108e226a9b1cd49d6ae02596c8f44603

- Date: 2026-07-16
- Subject: improving side-by-side plots
- Size: `[small]`
- Per-file changes:
  - `lfm/full_model/utils/plot_utils.py`: Changed Graha prediction overlay styling and cleaned model title capitalization in semantic comparison plots.

## Commit 33: 26f6574179fcc2fce66ebaae76701fe35a8a7fe4

- Date: 2026-07-16
- Subject: adding comparison metrics
- Size: `[moderate]`
- Per-file changes:
  - `lfm/full_model/utils/__init__.py`: Exported new metric/comparison utilities.
  - `lfm/full_model/utils/plot_utils.py`: Added metric calculation/display support to semantic comparison visualization.
  - `notebooks/full_model/model_comparison_plan.txt`: Updated the plan to reflect metrics work.
  - `notebooks/full_model/toy_sem_seg_comparison.ipynb`: Added metric reporting to the notebook workflow.

## Commit 34: 91482fda7299c02cabdcebd70584fdaaf2094edf

- Date: 2026-07-16
- Subject: updating sbatch script to match notebook
- Size: `[moderate]`
- Per-file changes:
  - `notebooks/full_model/sbatch_toy_sem_seg_comparison.sh`: Updated CLI wrapper defaults/arguments to match notebook behavior.
  - `notebooks/full_model/toy_sem_seg_comparison.py`: Added or aligned arguments used by the sbatch wrapper.

## Commit 35: 42b90b5036729852b3d09b6c274df2bf36a14687

- Date: 2026-07-16
- Subject: improving dataset logging
- Size: `[small]`
- Per-file changes:
  - `lfm/full_model/datamodules/datamodule.py`: Added clearer split-specific dataset logging for full-model datamodules.
  - `lfm/toy_model/sem_seg/lightning_wrappers/toy_sem_seg_datamodule.py`: Added or propagated split labels to toy datamodule logging.
  - `lfm/toy_model/sem_seg/sseg_dataset.py`: Improved dataset initialization prints so messages identify the split they belong to.

## Commit 36: 1deca69ecc6f08a1fe919a7b3070cfadaf667cac

- Date: 2026-07-16
- Subject: adding capability to compare loss/lr/etc
- Size: `[large]`
- Per-file changes:
  - `lfm/full_model/lfm_seg_finetuning_direct.py`: Added configurable full-model training options needed for tighter semantic comparisons.
  - `lfm/full_model/utils/plot_utils.py`: Updated plots to support the new comparison outputs.
  - `lfm/toy_model/sem_seg/lightning_wrappers/toy_sem_seg_datamodule.py`: Added datamodule-side configuration support for tighter toy/full experiment matching.
  - `lfm/toy_model/sem_seg/lightning_wrappers/toy_sem_seg_lightning.py`: Added configurable loss, optimizer, learning-rate, and gradient-clipping behavior for the toy Lightning module.
  - `lfm/toy_model/sem_seg/sseg_dataset.py`: Updated dataset options to support the comparison setup.
  - `notebooks/full_model/toy_sem_seg_comparison.ipynb`: Added notebook config options for tighter comparisons.
  - `notebooks/full_model/toy_sem_seg_comparison.py`: Added matching CLI/config support for loss, learning rate, optimizer-related settings, and normalization options.
- Technical note: This commit is important experimentally because it makes non-architecture variables more controllable.

## Commit 37: 21495512aa8d3767befc713fdd0fcf0f12499f00

- Date: 2026-07-17
- Subject: updating dataset with 100 test samples
- Size: `[small]`
- Per-file changes:
  - `notebooks/full_model/scratch.ipynb`: Updated scratch dataset-generation logic to produce the intended test split size.

## Commit 38: 7a6755a674281e8eee2905296b643ef517af447d

- Date: 2026-07-17
- Subject: updating scratch nb and model checkpointing
- Size: `[moderate]`
- Per-file changes:
  - `notebooks/full_model/scratch.ipynb`: Updated scratch data-prep/checking content.
  - `notebooks/full_model/toy_sem_seg_comparison.py`: Modified checkpoint callback behavior for semantic comparison training.

## Commit 39: 460379a269bbdd48704b08a91bffa1cdf3a335d9

- Date: 2026-07-17
- Subject: updating model checkpointing
- Size: `[moderate]`
- Per-file changes:
  - `lfm/full_model/lfm_seg_finetuning_direct.py`: Updated full-model checkpoint naming/saving behavior.
  - `lfm/toy_model/sem_seg/lightning_wrappers/toy_sem_seg_lightning.py`: Adjusted logging/checkpoint-related behavior in the toy Lightning module.
  - `notebooks/full_model/toy_sem_seg_comparison.py`: Updated semantic comparison checkpoint directory/naming behavior.

## Commit 40: 8b4a878248d1bce86a5a892f74a08580145c1565

- Date: 2026-07-17
- Subject: fixing batch size warning
- Size: `[small]`
- Per-file changes:
  - `lfm/full_model/graha-lunar-fm/terratorch_integration/lunar_segmentation_task.py`: Added explicit batch-size logging behavior to avoid Lightning's ambiguous batch-size warning.
  - `lfm/full_model/lfm_seg_finetuning_direct.py`: Propagated the warning fix through the direct full-model task wrapper.
  - `lfm/toy_model/sem_seg/lightning_wrappers/toy_sem_seg_lightning.py`: Added explicit `batch_size` values to toy model logging calls.

## Commit 41: 057a55f3b40cecbce8380a7d8658e77cb9ba9033

- Date: 2026-07-17
- Subject: adding checkpoint support to comparison nb/py
- Size: `[moderate]`
- Per-file changes:
  - `lfm/full_model/lfm_seg_finetuning_direct.py`: Added full-model checkpoint load/resume support.
  - `notebooks/full_model/toy_sem_seg_comparison.ipynb`: Added config cells for loading existing toy or Graha checkpoints.
  - `notebooks/full_model/toy_sem_seg_comparison.py`: Added matching CLI/config support for checkpoint paths.
- Technical note: This made it possible to train one model, reuse the other, and still produce side-by-side comparison outputs.

## Commit 42: 7fe125050d280f42369031bc9aabb57a0581dc24

- Date: 2026-07-17
- Subject: moving docs
- Size: `[small]`
- Per-file changes:
  - `docs/instance_seg_plan.txt`: Added the instance segmentation planning document under repo docs.
  - `notebooks/full_model/model_comparison_plan.txt` to `docs/model_comparison_plan.txt`: Moved the semantic/full model comparison plan into the docs folder.

## Commit 43: 2194dcbd6922a5e04972677a7f1f8c8e4f4ae82d

- Date: 2026-07-17
- Subject: updating scratch nb
- Size: `[small]`
- Per-file changes:
  - `notebooks/full_model/scratch.ipynb`: Updated scratch notebook cells for data preparation or sanity-check work.

## Commit 44: cc00c9f6f119d70d19685c905a2b16de573da0d1

- Date: 2026-07-17
- Subject: adding test suite
- Size: `[large]`
- Per-file changes:
  - `docs/instance_seg_plan.txt`: Updated the instance segmentation plan while semantic checkpoint testing work was being added.
  - `notebooks/full_model/sbatch_semantic_checkpoint_sweep.sh`: Added an sbatch wrapper for running semantic checkpoint sweeps on HPC.
  - `notebooks/full_model/semantic_checkpoint_sweep.ipynb`: Added a notebook workflow for sweeping toy/Graha semantic checkpoints over the test set.
  - `notebooks/full_model/semantic_checkpoint_sweep.py`: Added the script equivalent of the semantic checkpoint sweep notebook.
- Technical note: This begins the test-suite workflow where each checkpoint can be evaluated and its sample inputs, targets, predictions, and metrics saved.

## Commit 45: 5b27c65ac9ec33bd066cc29863268fb83c28075b

- Date: 2026-07-17
- Subject: improving checkpointing
- Size: `[moderate]`
- Per-file changes:
  - `lfm/full_model/lfm_seg_finetuning_direct.py`: Refined full-model checkpoint callbacks/naming.
  - `notebooks/full_model/toy_sem_seg_comparison.ipynb`: Updated notebook checkpoint configuration.
  - `notebooks/full_model/toy_sem_seg_comparison.py`: Updated script checkpoint configuration so outputs are stored in the intended structure.

## Commit 46: f8da611e5e5ae9a087971cef0208227729e2865d

- Date: 2026-07-17
- Subject: improving test suite
- Size: `[moderate]`
- Per-file changes:
  - `notebooks/full_model/semantic_checkpoint_sweep.ipynb`: Refined checkpoint sweep notebook behavior.
  - `notebooks/full_model/semantic_checkpoint_sweep.py`: Refined checkpoint sweep script behavior.

## Commit 47: a0c855a818d283a4e1c571ca0fc15bd7f773c2fb

- Date: 2026-07-17
- Subject: further testing improvements
- Size: `[moderate]`
- Per-file changes:
  - `notebooks/full_model/sbatch_semantic_checkpoint_sweep.sh`: Updated sweep wrapper arguments/defaults.
  - `notebooks/full_model/semantic_checkpoint_sweep.ipynb`: Updated notebook-side sweep options.
  - `notebooks/full_model/semantic_checkpoint_sweep.py`: Updated script-side sweep options.

## Commit 48: c52634fc64dbb413de7914891a41b85badee80eb

- Date: 2026-07-17
- Subject: more testing improvements
- Size: `[small]`
- Per-file changes:
  - `notebooks/full_model/semantic_checkpoint_sweep.py`: Further refined semantic sweep logic, likely around progress, saved output structure, or metric handling.

## Commit 49: 4e3de2be4097c19cb3963cf7e96396972261cea1

- Date: 2026-07-17
- Subject: more testing improvements
- Size: `[moderate]`
- Per-file changes:
  - `notebooks/full_model/semantic_checkpoint_sweep.ipynb`: Updated notebook to match the improved sweep script.
  - `notebooks/full_model/semantic_checkpoint_sweep.py`: Continued semantic checkpoint sweep refinements.

## Commit 50: 504cf7d66b0b996139a533ca6a1e549e4086236f

- Date: 2026-07-20
- Subject: adding iseg baseline
- Size: `[moderate]`
- Per-file changes:
  - `docs/instance_seg_plan.txt`: Updated the instance segmentation plan to reflect the next baseline implementation step.
  - `lfm/full_model/datamodules/datamodule_utils.py`: Added or expanded helpers needed for instance labels, boxes, masks, and crop handling.
  - `lfm/full_model/utils/__init__.py`: Exported new plotting or instance segmentation helper utilities.
  - `lfm/full_model/utils/plot_utils.py`: Added early instance segmentation visualization support.

## Commit 51: 79face56ce53be1aa3ec4c92ccd6a9fb1ee117fe

- Date: 2026-07-20
- Subject: adding chmod changes
- Size: `[very large]`
- Per-file changes:
  - `.gitignore`, `README.md`, `environment.yaml`, `requirements.txt`, package `__init__.py` files, model files, notebooks, docs, TMS assets, toy-model modules, full-model modules, and Graha/TerraMind copied files: Git recorded many files as modified, apparently due file mode/permission normalization rather than functional content changes.
  - `TMS/RG/*.json` and `TMS/RG/tile_database.gpkg`: Permission metadata changed on TMS reference data files.
  - `lfm/full_model/graha-lunar-fm/**`: Permission metadata changed across the copied TerraMind/Graha subtree.
  - `lfm/toy_model/**`, `model/**`, `view/**`, and `notebooks/**`: Permission metadata changed across existing project code and notebooks.
- Technical note: This commit is operational noise from chmod/file-mode changes. It is large in Git but should not be interpreted as broad source behavior change unless line diffs show otherwise.

## Commit 52: 7766873cf4dbe46c5b17be63f88c6f7e3fc2e494

- Date: 2026-07-20
- Subject: permission changes
- Size: `[very large]`
- Per-file changes:
  - Same broad file families as commit 51: Git recorded another round of permission/file-mode changes across TMS assets, full-model code, Graha/TerraMind copied files, toy-model code, notebooks, docs, model utilities, and environment files.
- Technical note: Treat this as a second permission-normalization commit rather than a model or experiment behavior change.

## Commit 53: 5947dc93f0ea3bcec0f319529e8a4c79ef7af958

- Date: 2026-07-20
- Subject: updating scratch nb to create iseg dataset
- Size: `[small]`
- Per-file changes:
  - `notebooks/full_model/scratch.ipynb`: Added/refactored scratch cells to create instance segmentation train/val/test splits from the source data, slicing images to the first 7 bands and preserving `.npz` labels.

## Commit 54: 4ce7783a7b46e8e0654ba4ed0867afad86d3cf14

- Date: 2026-07-20
- Subject: updating scratch nb
- Size: `[small]`
- Per-file changes:
  - `notebooks/full_model/scratch.ipynb`: Refined scratch notebook data-prep/sanity-check cells after the initial instance split creation work.

## Commit 55: 55545964988572ab9f80eb3e846818f1c54a1f6e

- Date: 2026-07-20
- Subject: updating sem seg to create new dataset
- Size: `[small]`
- Per-file changes:
  - `notebooks/full_model/scratch.ipynb`: Refactored semantic segmentation split creation to use the same base input directory pattern as the instance segmentation data preparation while preserving label files.

## Commit 56: e1b3735151c17b63c96aa7ceba3ed47f676173cd

- Date: 2026-07-20
- Subject: clearing scratch outputs
- Size: `[small]`
- Per-file changes:
  - `notebooks/full_model/scratch.ipynb`: Cleared notebook outputs so scratch state is not committed with generated display/output noise.

## Commit 57: e0f44103c04f62b259834526bb381fcd57eec255

- Date: 2026-07-20
- Subject: adding iseg scripts
- Size: `[large]`
- Per-file changes:
  - `lfm/toy_model/sem_seg/lightning_wrappers/toy_sem_seg_datamodule.py`: Adjusted shared/toy datamodule behavior needed by comparison workflows.
  - `lfm/toy_model/sem_seg/sseg_dataset.py`: Updated toy semantic dataset behavior, likely shared with instance data handling conventions.
  - `notebooks/full_model/sbatch_toy_inst_seg_comparison.sh`: Added a shell wrapper for early toy instance segmentation comparison.
  - `notebooks/full_model/toy_inst_seg_comparison.ipynb`: Added an early toy instance segmentation comparison notebook.
  - `notebooks/full_model/toy_inst_seg_comparison.py`: Added the Python script equivalent for early toy instance comparison.

## Commit 58: 166106ad63dc8f6a038d46a61cfc7aee16991ac5

- Date: 2026-07-20
- Subject: updating iseg plan + scratch
- Size: `[small]`
- Per-file changes:
  - `docs/instance_seg_plan.txt`: Updated the instance segmentation plan after initial sanity tests.
  - `notebooks/full_model/scratch.ipynb`: Updated scratch notebook cells for instance segmentation inspection/testing.

## Commit 59: c0203e04385a1cf4e5cdf3083c159e025bdb461e

- Date: 2026-07-20
- Subject: adding terratorch iseg
- Size: `[moderate]`
- Per-file changes:
  - `docs/instance_seg_plan.txt`: Updated the plan to include TerraTorch object-detection/instance-segmentation direction.
  - `lfm/full_model/datamodules/__init__.py`: Exported new instance/object-detection datamodule classes.
  - `lfm/full_model/datamodules/datamodule.py`: Added object-detection-style instance segmentation datamodule behavior.
  - `lfm/full_model/datamodules/datamodule_utils.py`: Added helpers for converting `.npz` instance labels into boxes, masks, labels, and metadata.
  - `notebooks/full_model/scratch.ipynb`: Added object-detection sanity-check cells.
- Technical note: This is the start of true instance segmentation support through detection-style model targets, not just semantic masks with shape loss.

## Commit 60: a126048f7842318c324b24ddd3178fed1dd0b1cf

- Date: 2026-07-20
- Subject: cleaning up iseg/sseg datamodules
- Size: `[large]`
- Per-file changes:
  - `docs/instance_seg_plan.txt`: Updated plan status after the datamodule refactor.
  - `lfm/full_model/datamodules/__init__.py`: Updated package exports for split datamodule files.
  - `lfm/full_model/datamodules/datamodule.py`: Reduced old monolithic datamodule responsibilities.
  - `lfm/full_model/datamodules/instance_segmentation.py`: Added dedicated instance segmentation datamodule classes.
  - `lfm/full_model/datamodules/lunar_segmentation_dataset.py`: Added shared dataset implementation used by semantic and instance datamodules.
  - `lfm/full_model/datamodules/semantic_segmentation.py`: Added dedicated semantic segmentation datamodule classes.
- Technical note: This commit separates shared dataset mechanics from task-specific datamodule behavior.

## Commit 61: 73502bd8fcaae3da91aefebad38c628c2a6244dd

- Date: 2026-07-20
- Subject: more datamodule moving, adding iseg smoke test
- Size: `[moderate]`
- Per-file changes:
  - `docs/instance_seg_plan.txt`: Updated plan status for datamodule and smoke-test progress.
  - `lfm/full_model/datamodules/__init__.py`: Updated exports after renaming the parent datamodule file.
  - `lfm/full_model/datamodules/instance_segmentation.py`: Refined instance datamodule imports/inheritance.
  - `lfm/full_model/datamodules/datamodule.py` to `lfm/full_model/datamodules/lunar_segmentation_datamodule.py`: Renamed the parent datamodule file to make its role explicit.
  - `lfm/full_model/datamodules/semantic_segmentation.py`: Updated imports/inheritance after the rename.
  - `notebooks/full_model/scratch.ipynb`: Added or updated instance segmentation smoke-test cells.

## Commit 62: 66e49ccf1a90d29e3fba43527c3ae240084d465a

- Date: 2026-07-20
- Subject: removing scratch cells
- Size: `[small]`
- Per-file changes:
  - `notebooks/full_model/scratch.ipynb`: Removed exploratory scratch cells that were no longer needed.

## Commit 63: 39bf44ca1f34532326a8ad46a245f9c211a8e809

- Date: 2026-07-20
- Subject: removing scratch cells
- Size: `[small]`
- Per-file changes:
  - `notebooks/full_model/scratch.ipynb`: Continued cleanup of obsolete scratch cells.

## Commit 64: f1550fd9a76653c29566412cf6d4216b035a0149

- Date: 2026-07-20
- Subject: fixing iseg model input key error
- Size: `[small]`
- Per-file changes:
  - `notebooks/full_model/scratch.ipynb`: Updated the instance segmentation smoke-test model config to include the expected input key/band metadata for TerraTorch object detection model creation.

## Commit 65: 87bd046c1eca83ba15818d5b038c4c598a421945

- Date: 2026-07-20
- Subject: fixing model feature map shape mismatch
- Size: `[small]`
- Per-file changes:
  - `notebooks/full_model/scratch.ipynb`: Adjusted instance segmentation smoke-test settings, including anchor/feature-map compatibility, so Mask R-CNN could run against the model feature outputs.

## Commit 66: 8c2845dfc7f111cf540ed5c4ec231c870f7b6e89

- Date: 2026-07-20
- Subject: adding wrapper scripts for iseg
- Size: `[large]`
- Per-file changes:
  - `docs/instance_seg_plan.txt`: Updated the plan after adding runnable full-model instance training artifacts.
  - `notebooks/full_model/instance_seg_finetuning.ipynb`: Added notebook workflow for Graha/full-model instance segmentation fine-tuning.
  - `notebooks/full_model/instance_seg_finetuning.py`: Added script equivalent for instance segmentation fine-tuning.
  - `notebooks/full_model/sbatch_instance_seg_finetuning.sh`: Added sbatch wrapper for running instance segmentation fine-tuning on HPC.

## Commit 67: 67297cefa314437ca061556a8f9f644d5774afd1

- Date: 2026-07-20
- Subject: fixing torch error in iseg nb
- Size: `[moderate]`
- Per-file changes:
  - `notebooks/full_model/instance_seg_finetuning.ipynb`: Updated notebook code to avoid the observed Torch/runtime error.
  - `notebooks/full_model/instance_seg_finetuning.py`: Mirrored the notebook fix in the script.
  - `notebooks/full_model/scratch.ipynb`: Updated smoke/sanity test code to match the fixed instance segmentation behavior.

## Commit 68: 02f7dc003f0ddae3623e93134fa23005c15b3023

- Date: 2026-07-20
- Subject: adding plotting behavior to iseg
- Size: `[moderate]`
- Per-file changes:
  - `docs/instance_seg_plan.txt`: Updated plan status for validation plotting.
  - `lfm/full_model/utils/__init__.py`: Exported instance plotting helpers.
  - `lfm/full_model/utils/plot_utils.py`: Added instance segmentation validation plotting, including input/ground-truth/prediction visualization.
  - `notebooks/full_model/instance_seg_finetuning.ipynb`: Added plotting configuration/usage in the notebook.
  - `notebooks/full_model/instance_seg_finetuning.py`: Added plotting callback/script support.

## Commit 69: 4c2b2d8f9d2e13d19c488641c1010d923252abd0

- Date: 2026-07-20
- Subject: improving iseg plots
- Size: `[small]`
- Per-file changes:
  - `lfm/full_model/utils/plot_utils.py`: Adjusted instance segmentation plot styling, including ground-truth color behavior.

## Commit 70: ae0ac74ff15516b9ebad0fb5a027417f7c5deddf

- Date: 2026-07-20
- Subject: trying to fix iseg label offset
- Size: `[moderate]`
- Per-file changes:
  - `lfm/full_model/datamodules/datamodule_utils.py`: Added or adjusted crop/shift handling for instance masks and bounding boxes.
  - `lfm/full_model/datamodules/lunar_segmentation_datamodule.py`: Propagated mask/box shift configuration through the parent datamodule.
  - `lfm/full_model/datamodules/lunar_segmentation_dataset.py`: Applied shift/crop logic at dataset sample construction time.
  - `lfm/full_model/utils/__init__.py`: Exported new comparison/plotting helper symbols.
  - `lfm/full_model/utils/plot_utils.py`: Added label-comparison plotting utilities to visualize old vs new instance labels.
  - `notebooks/full_model/instance_seg_finetuning.ipynb`: Updated instance fine-tuning notebook to expose the label-shift behavior.
  - `notebooks/full_model/instance_seg_finetuning.py`: Updated script configuration for label-shift behavior.
  - `notebooks/full_model/scratch.ipynb`: Added an instance label comparison section to inspect whether shifts were introduced by the new dataset.
- Technical note: This work investigated a small apparent offset between crater labels and imagery; later discussion concluded the shift preexisted the new split and was not urgent.

## Commit 71: 4008a7ba7d6dd1f7055f970b6279427ede526e1c

- Date: 2026-07-20
- Subject: prepping for repo cleanup + iseg comparison
- Size: `[large]`
- Per-file changes:
  - `docs/repo_cleanup_plan.txt`: Added a plan for moving scripts out of notebooks and organizing reusable workflow code.
  - `lfm/toy_model/inst_seg/lightning_wrappers/__init__.py`: Added package exports for toy instance Lightning wrappers.
  - `lfm/toy_model/inst_seg/lightning_wrappers/toy_instance_seg_datamodule.py`: Added a Lightning datamodule wrapper for the toy instance segmentation workflow.
  - `lfm/toy_model/inst_seg/lightning_wrappers/toy_instance_seg_lightning.py`: Added a Lightning module wrapper around the toy instance segmentation model.
  - `notebooks/full_model/instance_seg_comparison.ipynb`: Added notebook workflow for comparing toy and Graha instance segmentation.
  - `notebooks/full_model/instance_seg_comparison.py`: Added script workflow for comparing toy and Graha instance segmentation.
  - `notebooks/full_model/sbatch_instance_seg_comparison.sh`: Added sbatch wrapper for the instance comparison script.

## Commit 72: a1e33b682a5fd5176433ec63a42dd8d0427b418f

- Date: 2026-07-20
- Subject: reorganizing repo
- Size: `[large]`
- Per-file changes:
  - `notebooks/full_model/crater_detection_nac_dtm_meta.yaml` to `configs/full_model/crater_detection_nac_dtm_meta.yaml`: Moved full-model config out of notebooks.
  - `notebooks/full_model/debug_crater.yaml` to `configs/full_model/debug_crater.yaml`: Moved debug config out of notebooks.
  - `docs/instance_seg_plan.txt`, `docs/model_comparison_plan.txt`, `docs/repo_cleanup_plan.txt`: Updated docs to reflect the new script/config organization.
  - `notebooks/full_model/*.ipynb`: Updated notebooks after moving Python and shell scripts out of `notebooks`.
  - `scripts/logs/.gitkeep`: Added an empty tracked log directory for sbatch outputs.
  - `notebooks/full_model/instance_seg_comparison.py` to `scripts/python/instance_seg_comparison.py`: Moved instance comparison Python workflow into `scripts/python`.
  - `notebooks/full_model/instance_seg_finetuning.py` to `scripts/python/instance_seg_finetuning.py`: Moved instance fine-tuning Python workflow into `scripts/python`.
  - `notebooks/full_model/semantic_checkpoint_sweep.py` to `scripts/python/semantic_checkpoint_sweep.py`: Moved semantic checkpoint sweep into `scripts/python`.
  - `notebooks/full_model/toy_inst_seg_comparison.py` to `scripts/python/toy_inst_seg_comparison.py`: Moved the older toy instance comparison script into `scripts/python`.
  - `notebooks/full_model/toy_sem_seg_comparison.py` to `scripts/python/toy_sem_seg_comparison.py`: Moved semantic comparison Python workflow into `scripts/python`.
  - `notebooks/full_model/sbatch_*.sh` to `scripts/shell/sbatch_*.sh`: Moved sbatch wrappers into `scripts/shell`.
- Technical note: This commit implements the repo hygiene goal: notebooks become notebooks only, reusable or runnable scripts move to `scripts/python` and `scripts/shell`.

## Commit 73: f475948e116d68540791a6ad0f4314fef539b1f7

- Date: 2026-07-20
- Subject: deleting old scripts
- Size: `[moderate]`
- Per-file changes:
  - `docs/instance_seg_plan.txt`: Updated plan to reflect removal of obsolete toy-only instance comparison artifacts.
  - `notebooks/full_model/toy_inst_seg_comparison.ipynb`: Deleted the older toy-only instance comparison notebook.
  - `scripts/python/toy_inst_seg_comparison.py`: Deleted the older toy-only instance comparison script.
  - `scripts/shell/sbatch_toy_inst_seg_comparison.sh`: Deleted the older toy-only instance comparison sbatch wrapper.
- Technical note: This makes the true toy-vs-Graha instance comparison workflow the surviving comparison path.

## Commit 74: 2178ab9a8bd06dd901f3377efb9670b6ecb5203f

- Date: 2026-07-20
- Subject: improving default output dirs
- Size: `[moderate]`
- Per-file changes:
  - `.gitignore`: Updated ignore rules for script-generated outputs.
  - `docs/repo_cleanup_plan.txt`: Updated cleanup plan status/context.
  - `lfm/full_model/lfm_seg_finetuning_direct.py`: Updated default output behavior away from notebook-local output paths.
  - `scripts/outputs/.gitkeep`: Added a tracked default output root under `scripts`.
  - `scripts/python/instance_seg_comparison.py`: Changed default outputs to `scripts/outputs`.
  - `scripts/python/instance_seg_finetuning.py`: Changed default outputs to `scripts/outputs`.
  - `scripts/python/semantic_checkpoint_sweep.py`: Changed default outputs to `scripts/outputs`.
  - `scripts/python/toy_sem_seg_comparison.py`: Changed default outputs to `scripts/outputs`.

## Commit 75: 4d7ff6e70e1ad339cfbe6d5f2727d03b61948e4a

- Date: 2026-07-20
- Subject: improving iseg comparison script
- Size: `[moderate]`
- Per-file changes:
  - `scripts/python/instance_seg_comparison.py`: Refined instance comparison workflow behavior, likely around CLI arguments, output paths, progress reporting, and model-specific setup.

## Commit 76: 4c33c996d170a656568e9f512869b7aa96f3d2e1

- Date: 2026-07-20
- Subject: removing eval mode modules in m2f
- Size: `[small]`
- Per-file changes:
  - `lfm/toy_model/inst_seg/lightning_wrappers/toy_instance_seg_lightning.py`: Adjusted Mask2Former toy instance Lightning module startup/training behavior so modules are not left in eval mode during training.

## Commit 77: 8b0d978330419242992229f42880fbd4a48058a0

- Date: 2026-07-20
- Subject: improving iseg prog reporting + plots
- Size: `[moderate]`
- Per-file changes:
  - `lfm/full_model/utils/__init__.py`: Exported updated instance plotting/progress helper functions.
  - `lfm/full_model/utils/plot_utils.py`: Improved instance plot output and status messages.
  - `scripts/python/instance_seg_comparison.py`: Added clearer training/progress logging to the instance comparison workflow.
  - `scripts/python/instance_seg_finetuning.py`: Added matching progress/plot logging to the single-model Graha instance fine-tuning workflow.

## Commit 78: 1b6d247b91d2eaf651ea12f06a917ffff2b34ac0

- Date: 2026-07-20
- Subject: changing iseg plots again
- Size: `[small]`
- Per-file changes:
  - `lfm/full_model/utils/plot_utils.py`: Changed instance segmentation validation plot layout/style.
  - `scripts/python/instance_seg_comparison.py`: Updated comparison script to call the revised plotting behavior.
  - `scripts/python/instance_seg_finetuning.py`: Updated fine-tuning script to call the revised plotting behavior.

## Commit 79: 1d7311b4ad4f83a8100a17f014745f1c2194f497

- Date: 2026-07-20
- Subject: making iseg plots more like toy model plots
- Size: `[small]`
- Per-file changes:
  - `lfm/full_model/utils/plot_utils.py`: Adjusted instance validation plots to resemble the existing toy instance segmentation driver style, including rows, colors, labels, and displayed stats.

## Commit 80: cde70a0661ab193331e4d92f8a3e59ad703bd39b

- Date: 2026-07-20
- Subject: adding plan + iseg bugfix
- Size: `[moderate]`
- Per-file changes:
  - `docs/full_model_complexity_refactor_plan.txt`: Added a plan for breaking up complex full-model scripts and utilities.
  - `scripts/python/instance_seg_comparison.py`: Fixed an instance comparison bug discovered during smoke testing or workflow cleanup.

## Commit 81: 8e363464743d144d91fc4d3377f74572f56b717b

- Date: 2026-07-20
- Subject: removing 'double logging' from iseg
- Size: `[small]`
- Per-file changes:
  - `scripts/python/instance_seg_comparison.py`: Removed duplicate callback/progress logging so each epoch/batch message appears once.

## Commit 82: 44032b91d33c957ff163ab3d639c2a2461ceb78a

- Date: 2026-07-20
- Subject: updating output dirs
- Size: `[moderate]`
- Per-file changes:
  - `docs/repo_cleanup_plan.txt`: Updated output-directory cleanup plan notes.
  - `scripts/python/instance_seg_comparison.py`: Changed instance comparison plot outputs to use `plots/single_model/{toy_model,full_model}` and `plots/comparison`.
  - `scripts/python/toy_sem_seg_comparison.py`: Changed semantic comparison plot outputs to use the same `single_model` and `comparison` convention.
- Technical note: This made semantic and instance comparison output structures consistent.

## Commit 83: c655cd9cfe230332aeb348ec9608f0505207e644

- Date: 2026-07-21
- Subject: adding iseg ckpt sweep
- Size: `[large]`
- Per-file changes:
  - `notebooks/full_model/instance_checkpoint_sweep.ipynb`: Added notebook workflow to evaluate instance segmentation checkpoints across a split and save per-sample artifacts/metrics.
  - `scripts/python/instance_checkpoint_sweep.py`: Added script equivalent for instance checkpoint sweeping.
  - `scripts/shell/sbatch_instance_checkpoint_sweep.sh`: Added sbatch wrapper for running the instance checkpoint sweep on HPC.

## Commit 84: 9f2ef8e525ad08533410d0fd48660573fa4e05d4

- Date: 2026-07-21
- Subject: adding r-cnn decoder for toy model
- Size: `[large]`
- Per-file changes:
  - `lfm/toy_model/inst_seg/dino_mask_rcnn_model.py`: Added a DINO-backbone Mask R-CNN model path for toy instance segmentation so the decoder family is closer to Graha Mask R-CNN.
  - `lfm/toy_model/inst_seg/lightning_wrappers/__init__.py`: Exported the new DINO Mask R-CNN wrappers.
  - `lfm/toy_model/inst_seg/lightning_wrappers/toy_dino_mask_rcnn_datamodule.py`: Added datamodule support for the DINO Mask R-CNN toy instance path.
  - `lfm/toy_model/inst_seg/lightning_wrappers/toy_dino_mask_rcnn_lightning.py`: Added Lightning module support for training/evaluating DINO Mask R-CNN.
  - `scripts/python/checkpoint_pipeline.py`: Added a train-then-sweep orchestration script that can run training and then checkpoint evaluation.
  - `scripts/python/instance_checkpoint_sweep.py`: Updated sweep logic to support the new toy architecture.
  - `scripts/python/instance_seg_comparison.py`: Added CLI/config support for selecting the toy instance architecture, including DINO Mask R-CNN.
  - `scripts/shell/sbatch_instance_train_then_checkpoint_sweep.sh`: Added full-pipeline sbatch wrapper for instance segmentation.
  - `scripts/shell/sbatch_semantic_train_then_checkpoint_sweep.sh`: Added full-pipeline sbatch wrapper for semantic segmentation.
- Technical note: This is a key architecture-alignment commit: toy instance segmentation no longer has to be Mask2Former-only.

## Commit 85: 4736e1deb30296f437b03b12d955740ba2b60ef3

- Date: 2026-07-21
- Subject: improving logs for iseg/pipeline
- Size: `[small]`
- Per-file changes:
  - `lfm/full_model/utils/plot_utils.py`: Fixed or clarified instance prediction cache/plot logging, including model-name labels in saved-output messages.

## Commit 86: ce0f7dba8e831eb570c8fe0eb5fa189395995437

- Date: 2026-07-21
- Subject: removed duplicate logs from iseg
- Size: `[small]`
- Per-file changes:
  - `scripts/python/instance_seg_comparison.py`: Removed a duplicate checkpoint callback log path so callback state messages are not printed twice.

## Commit 87: 5b200766c80042f169c58a16fc256e1abc84ff86

- Date: 2026-07-21
- Subject: adding metrics nb for future
- Size: `[moderate]`
- Per-file changes:
  - `notebooks/full_model/sem_seg_test_metrics.ipynb`: Added a semantic segmentation metrics notebook for post-training metric exploration and plotting.

## Commit 88: 36b96c6706df72a0d2aab31264a7976d301343f4

- Date: 2026-07-21
- Subject: updating metrics nb + training to allow test
- Size: `[moderate]`
- Per-file changes:
  - `notebooks/full_model/sem_seg_test_metrics.ipynb`: Cleaned up metrics notebook plotting sections and added output-saving behavior.
  - `scripts/python/checkpoint_pipeline.py`: Added options to run testing during or after training pipeline execution.
  - `scripts/python/instance_seg_comparison.py`: Added optional per-epoch testing/test-suite controls to the instance comparison workflow.
  - `scripts/python/toy_sem_seg_comparison.py`: Added optional per-epoch testing/test-suite controls to the semantic comparison workflow.

## Commit 89: 956af0f66d74e7f3423b43842b8b0fc9277e3f33

- Date: 2026-07-21
- Subject: adding better modality options to graha
- Size: `[large]`
- Per-file changes:
  - `lfm/full_model/lfm_seg_finetuning_direct.py`: Added Graha WAC modality configuration options, including a VIS/UV split mode and merge method.
  - `notebooks/full_model/instance_checkpoint_sweep.ipynb`: Updated instance sweep notebook to expose/use the new Graha WAC modality options.
  - `notebooks/full_model/instance_seg_comparison.ipynb`: Updated instance comparison notebook for the new Graha WAC modality configuration.
  - `notebooks/full_model/instance_seg_finetuning.ipynb`: Updated instance fine-tuning notebook for the new modality setup.
  - `notebooks/full_model/semantic_checkpoint_sweep.ipynb`: Updated semantic sweep notebook for the new modality setup.
  - `notebooks/full_model/toy_sem_seg_comparison.ipynb`: Updated semantic comparison notebook for the new modality setup.
  - `scripts/python/checkpoint_pipeline.py`: Passed Graha modality arguments through full train-then-sweep pipeline commands.
  - `scripts/python/instance_checkpoint_sweep.py`: Added Graha modality CLI/config support for instance sweeps.
  - `scripts/python/instance_seg_comparison.py`: Added Graha modality CLI/config support for instance comparison.
  - `scripts/python/instance_seg_finetuning.py`: Added Graha modality CLI/config support for instance fine-tuning.
  - `scripts/python/semantic_checkpoint_sweep.py`: Added Graha modality CLI/config support for semantic sweeps.
  - `scripts/python/toy_sem_seg_comparison.py`: Added Graha modality CLI/config support for semantic comparison.
- Technical note: This is important for avoiding ad hoc `0.95*red`-style channel mapping and instead using a more explicit WAC VIS/UV modality treatment.

## Commit 90: 08ae9e258f057b710b598f07c8c41af5b0e64e59

- Date: 2026-07-21
- Subject: adding MAP and AP metrics to ckpt sweeps
- Size: `[moderate]`
- Per-file changes:
  - `scripts/python/instance_checkpoint_sweep.py`: Added AP/mAP metric calculation/output for instance checkpoint sweeps.
  - `scripts/python/semantic_checkpoint_sweep.py`: Added additional metric outputs for semantic checkpoint sweeps, including logits/class-pred saving support.

## Commit 91: 261d4060e5a58bcc8308f299814f89d899af2410

- Date: 2026-07-21
- Subject: fixing test suite namespace bug
- Size: `[small]`
- Per-file changes:
  - `scripts/python/instance_checkpoint_sweep.py`: Added missing namespace/default attributes needed when sweep code constructs comparison configs.
  - `scripts/python/semantic_checkpoint_sweep.py`: Added missing namespace/default attributes needed when sweep code constructs comparison configs.

## Commit 92: ab9a1e0036d76c4b4926842c4a6ae77f9cd3cb5e

- Date: 2026-07-21
- Subject: making sem seg logging better
- Size: `[moderate]`
- Per-file changes:
  - `lfm/full_model/lfm_seg_finetuning_direct.py`: Improved full-model semantic training logging/progress output.
  - `scripts/python/checkpoint_pipeline.py`: Improved train-then-sweep pipeline logging so semantic and instance logs are more consistent.
  - `scripts/python/toy_sem_seg_comparison.py`: Improved semantic comparison training/progress logs, aligning them more closely with the clearer instance segmentation logs.

## Commit 93: 7b1b964a2d19a1484f9a6dc5221db556b00cdbe2

- Date: 2026-07-21
- Subject: another test bugfix
- Size: `[small]`
- Per-file changes:
  - `scripts/python/semantic_checkpoint_sweep.py`: Fixed another semantic checkpoint sweep bug, likely a missing default or namespace field needed by the test-suite path.
- Technical note: This is the end commit in the requested range and closes the immediate sweep/pipeline bug-fix sequence.

## Commit 94: 78d2e7022b13dcf31e80f1d26e5af7a54fae368b

- Date: 2026-07-21
- Subject: large renaming/restructuring change + changelog doc
- Size: `[very large]`
- Per-file changes:
  - `docs/full_model_complexity_refactor_plan.txt`: Updated the complexity/refactor plan to reflect the newer full-model comparison structure and remaining cleanup targets.
  - `docs/ibm_model_branch_technical_notes.md`: Added a technical notes document for the IBM model branch, summarizing major implementation decisions and repo evolution.
  - `docs/ibm_model_commit_change_log.md`: Added this commit-level changelog document so the branch history can be reviewed without reading raw diffs.
  - `docs/model_comparison_plan.txt`: Updated the semantic model comparison plan to reflect current comparison workflows and script/notebook deliverables.
  - `docs/repo_cleanup_plan.txt`: Updated cleanup planning for the reorganized notebook/script/full-model layout.
  - `lfm/full_model/lfm_seg_finetuning_direct.py`: Kept the older Graha semantic fine-tuning entry point while the replacement semantic-specific module was introduced.
  - `lfm/full_model/semantic_seg_finetuning.py`: Added a clearer semantic segmentation fine-tuning module for the full/Graha model path.
  - `notebooks/full_model/semantic_checkpoint_sweep.ipynb`: Updated the semantic checkpoint sweep notebook for the new script names and comparison structure.
  - `notebooks/full_model/semantic_seg_comparison.ipynb`: Renamed from the toy-specific comparison notebook to a semantic comparison notebook, reflecting that it now compares toy DINO and Graha models.
  - `notebooks/full_model/semantic_seg_finetuning.ipynb`: Renamed from the direct Graha fine-tuning notebook to a semantic fine-tuning notebook.
  - `scripts/python/checkpoint_pipeline.py`: Updated pipeline orchestration to reflect semantic comparison/fine-tuning naming and path changes.
  - `scripts/python/semantic_checkpoint_sweep.py`: Updated sweep script references for the renamed semantic workflow.
  - `scripts/python/semantic_seg_comparison.py`: Added a clearer semantic comparison script as the main script equivalent of the notebook.
  - `scripts/python/semantic_seg_finetuning.py`: Added a script entry point for semantic Graha/full-model fine-tuning.
  - `scripts/python/toy_sem_seg_comparison.py`: Kept/updated a compatibility wrapper for the older toy semantic comparison script name.
  - `scripts/shell/sbatch_sem_seg_finetune.sh`: Updated shell wrapper behavior to point toward the renamed semantic fine-tuning script.
  - `scripts/shell/sbatch_semantic_seg_comparison.sh`: Added a semantic comparison sbatch wrapper.
  - `scripts/shell/sbatch_semantic_seg_finetuning.sh`: Added a semantic fine-tuning sbatch wrapper.
  - `scripts/shell/sbatch_toy_sem_seg_comparison.sh`: Kept/updated compatibility wrapper behavior for the older toy semantic comparison name.
- Technical note: This commit was the first major move away from notebook-only/full-model ad hoc names toward task-specific semantic naming.

## Commit 95: ae48850d3973a44c6e1398654b0a400cc9fb195d

- Date: 2026-07-21
- Subject: another repo restructure
- Size: `[very large]`
- Per-file changes:
  - `docs/full_model_complexity_refactor_plan.txt`: Updated complexity notes after the package/script split.
  - `docs/instance_seg_plan.txt`: Updated the instance segmentation plan to reflect the true-instance path and current implementation status.
  - `docs/model_comparison_plan.txt`: Updated model-comparison sequencing after scripts and modules were moved into task-specific locations.
  - `docs/repo_cleanup_plan.txt`: Updated repo cleanup notes after reorganizing full-model utilities and scripts.
  - `lfm/full_model/__init__.py`: Updated package exports/imports for the new full-model package structure.
  - `lfm/full_model/all_tasks/__init__.py`: Added shared all-task namespace for code used across semantic and instance segmentation.
  - `lfm/full_model/all_tasks/datamodules/__init__.py`: Added shared datamodule exports for full-model task code.
  - `lfm/full_model/all_tasks/datamodules/datamodule_utils.py`: Moved shared datamodule helpers out of the old flat full-model datamodule folder.
  - `lfm/full_model/all_tasks/datamodules/lunar_segmentation_datamodule.py`: Moved the parent lunar segmentation datamodule into the all-task datamodule namespace.
  - `lfm/full_model/all_tasks/datamodules/lunar_segmentation_dataset.py`: Moved the shared lunar segmentation dataset into the all-task datamodule namespace.
  - `lfm/full_model/all_tasks/utils/__init__.py`: Moved shared utility exports into the all-task utility namespace.
  - `lfm/full_model/all_tasks/utils/plot_utils.py`: Moved shared plotting utilities into the all-task utility namespace.
  - `lfm/full_model/all_tasks/utils/utils.py`: Moved shared utility functions into the all-task utility namespace.
  - `lfm/full_model/datamodules/__init__.py`: Removed the obsolete flat datamodule namespace.
  - `lfm/full_model/inst_seg/__init__.py`: Added instance segmentation package exports.
  - `lfm/full_model/inst_seg/instance_mask_datamodule.py`: Renamed/moved the instance segmentation datamodule to a clearer task-specific module name.
  - `lfm/full_model/inst_seg/instance_seg_finetuning.py`: Moved full-model instance fine-tuning code from scripts into the package implementation location.
  - `lfm/full_model/lfm_seg_finetuning_direct.py`: Removed the obsolete direct semantic fine-tuning module.
  - `lfm/full_model/sem_seg/__init__.py`: Added semantic segmentation package exports.
  - `lfm/full_model/sem_seg/semantic_mask_datamodule.py`: Renamed/moved the semantic segmentation datamodule to a clearer task-specific module name.
  - `lfm/full_model/sem_seg/semantic_seg_finetuning.py`: Moved semantic full-model fine-tuning into the task-specific package folder.
  - `lfm/toy_model/inst_seg/lightning_wrappers/toy_dino_mask_rcnn_datamodule.py`: Updated imports for moved full-model datamodule utilities.
  - `lfm/toy_model/inst_seg/lightning_wrappers/toy_instance_seg_datamodule.py`: Updated imports for moved full-model datamodule utilities.
  - `notebooks/full_model/instance_checkpoint_sweep.ipynb`: Updated notebook imports/paths for the restructured package and scripts.
  - `notebooks/full_model/instance_seg_comparison.ipynb`: Updated notebook imports/paths for the restructured package and scripts.
  - `notebooks/full_model/instance_seg_finetuning.ipynb`: Updated notebook imports/paths for the restructured package and scripts.
  - `notebooks/full_model/scratch.ipynb`: Updated scratch notebook imports/paths for the restructured package.
  - `notebooks/full_model/semantic_checkpoint_sweep.ipynb`: Updated notebook imports/paths for the restructured semantic scripts.
  - `notebooks/full_model/semantic_seg_comparison.ipynb`: Updated notebook imports/paths for the restructured semantic scripts.
  - `notebooks/full_model/semantic_seg_finetuning.ipynb`: Updated notebook imports/paths for the restructured semantic package.
  - `scripts/python/all_tasks/checkpoint_pipeline.py`: Moved the train-then-sweep pipeline script into an all-tasks script namespace.
  - `scripts/python/instance_seg/instance_checkpoint_sweep.py`: Moved instance checkpoint sweep script into the instance-specific script folder.
  - `scripts/python/instance_seg/instance_seg_comparison.py`: Moved instance comparison script into the instance-specific script folder.
  - `scripts/python/instance_seg/instance_seg_finetuning.py`: Added a script wrapper for packaged instance fine-tuning code.
  - `scripts/python/semantic_seg/semantic_checkpoint_sweep.py`: Moved semantic checkpoint sweep script into the semantic-specific script folder.
  - `scripts/python/semantic_seg/semantic_seg_comparison.py`: Moved semantic comparison script into the semantic-specific script folder.
  - `scripts/python/semantic_seg/semantic_seg_finetuning.py`: Added/updated a script wrapper for packaged semantic fine-tuning code.
  - `scripts/python/semantic_seg/toy_sem_seg_comparison.py`: Moved compatibility script wrapper into the semantic script folder.
  - `scripts/shell/instance_seg/*.sh`: Moved instance sbatch wrappers into the instance-specific shell folder.
  - `scripts/shell/semantic_seg/*.sh`: Moved semantic sbatch wrappers into the semantic-specific shell folder.
- Technical note: This commit established the aligned task layout used now: `full_model/all_tasks`, `full_model/sem_seg`, `full_model/inst_seg`, `scripts/python/{all_tasks,semantic_seg,instance_seg}`, and matching `scripts/shell/...` folders.

## Commit 96: 8beec4cbb2da92c8a46daad4ceb80e6e9facb738

- Date: 2026-07-21
- Subject: adding terramind-style norm
- Size: `[large]`
- Per-file changes:
  - `lfm/full_model/all_tasks/datamodules/datamodule_utils.py`: Updated shared datamodule normalization helpers to support explicit normalization behavior needed by TerraMind-style stats.
  - `lfm/full_model/all_tasks/utils/__init__.py`: Exported the new shared normalization/stat-loading utility functions.
  - `lfm/full_model/all_tasks/utils/utils.py`: Added utility support for loading TerraMind WAC pretraining stats from modality metadata and wiring them into training/sweep configs.
  - `lfm/full_model/inst_seg/instance_seg_finetuning.py`: Added normalization-source handling to full-model instance fine-tuning.
  - `lfm/full_model/sem_seg/semantic_seg_finetuning.py`: Added normalization-source handling to full-model semantic fine-tuning.
  - `lfm/toy_model/inst_seg/lightning_wrappers/toy_dino_mask_rcnn_datamodule.py`: Added support for supplied mean/std stats and pretraining-style normalization for DINO Mask R-CNN instance data.
  - `lfm/toy_model/inst_seg/lightning_wrappers/toy_instance_seg_datamodule.py`: Added support for supplied mean/std stats and pretraining-style normalization for toy instance data.
  - `lfm/toy_model/sem_seg/lightning_wrappers/toy_sem_seg_datamodule.py`: Added support for supplied mean/std stats and pretraining-style normalization for toy semantic data.
  - `lfm/toy_model/sem_seg/sseg_dataset.py`: Added lower-level dataset support for controlled input scaling and externally supplied normalization stats.
  - `scripts/python/all_tasks/checkpoint_pipeline.py`: Passed normalization-source flags through the full train-then-sweep pipeline.
  - `scripts/python/instance_seg/instance_checkpoint_sweep.py`: Added normalization-source support for instance checkpoint sweeps.
  - `scripts/python/instance_seg/instance_seg_comparison.py`: Added normalization-source support for instance comparisons.
  - `scripts/python/semantic_seg/semantic_checkpoint_sweep.py`: Added normalization-source support for semantic checkpoint sweeps.
  - `scripts/python/semantic_seg/semantic_seg_comparison.py`: Added normalization-source support for semantic comparisons.
- Technical note: This enabled the key experiment switch between finetuning/train-split DINO z-score stats and TerraMind/Graha pretraining stats.

## Commit 97: 8cf1fd1a5281abd264bcfde5f07317f38ebbd775

- Date: 2026-07-21
- Subject: adding spatial loss to toy model + terramind style norm
- Size: `[large]`
- Per-file changes:
  - `lfm/full_model/sem_seg/__init__.py`: Exported the instance-derived semantic datamodule classes.
  - `lfm/full_model/sem_seg/semantic_from_instance_datamodule.py`: Added a semantic datamodule/dataset that reads `.npz` instance labels, converts instance masks to binary semantic masks, and keeps crater boxes available for shape/spatial loss.
  - `lfm/full_model/sem_seg/semantic_seg_finetuning.py`: Added semantic label-source selection and configurable Graha shape-loss weight/padding.
  - `lfm/toy_model/sem_seg/lightning_wrappers/toy_sem_seg_from_instance_datamodule.py`: Added toy semantic datamodule support for deriving semantic masks and crater boxes from instance `.npz` labels.
  - `lfm/toy_model/sem_seg/lightning_wrappers/toy_sem_seg_shape_lightning.py`: Added a toy semantic Lightning module that combines the base segmentation loss with a box-localized spatial/shape loss.
  - `scripts/python/all_tasks/checkpoint_pipeline.py`: Added pipeline arguments for semantic label source, toy shape loss, and Graha shape-loss controls.
  - `scripts/python/semantic_seg/semantic_checkpoint_sweep.py`: Updated semantic sweep setup to support instance-derived semantic labels and required namespace defaults.
  - `scripts/python/semantic_seg/semantic_seg_comparison.py`: Added CLI/config support for instance-derived semantic labels, toy spatial loss, and Graha shape-loss controls.
  - `scripts/shell/semantic_seg/sbatch_4_sem_seg_exp.sh`: Added a four-experiment launcher for normalization/loss comparison runs.
- Technical note: This laid the groundwork for the four semantic experiments: DINO norm versus TerraMind norm, each with Dice-only versus Dice-plus-spatial loss.

## Commit 98: 3cbe7a4b8d049b818276dd1cfecea42c3f2714c3

- Date: 2026-07-21
- Subject: changing .sh files to not use CRLF
- Size: `[moderate]`
- Per-file changes:
  - `.gitattributes`: Added `*.sh text eol=lf` so shell scripts stay compatible with Linux/sbatch even when edited on Windows.
  - `scripts/shell/instance_seg/sbatch_instance_checkpoint_sweep.sh`: Normalized line endings to LF.
  - `scripts/shell/instance_seg/sbatch_instance_seg_comparison.sh`: Normalized line endings to LF.
  - `scripts/shell/instance_seg/sbatch_instance_seg_finetuning.sh`: Normalized line endings to LF.
  - `scripts/shell/instance_seg/sbatch_instance_train_then_checkpoint_sweep.sh`: Normalized line endings to LF.
  - `scripts/shell/semantic_seg/sbatch_sem_seg_finetune.sh`: Normalized line endings to LF.
  - `scripts/shell/semantic_seg/sbatch_semantic_checkpoint_sweep.sh`: Normalized line endings to LF.
  - `scripts/shell/semantic_seg/sbatch_semantic_seg_comparison.sh`: Normalized line endings to LF.
  - `scripts/shell/semantic_seg/sbatch_semantic_seg_finetuning.sh`: Normalized line endings to LF.
  - `scripts/shell/semantic_seg/sbatch_semantic_train_then_checkpoint_sweep.sh`: Normalized line endings to LF.
  - `scripts/shell/semantic_seg/sbatch_toy_sem_seg_comparison.sh`: Normalized line endings to LF.
- Technical note: This fixes the `sbatch: Batch script contains DOS line breaks` failure mode and makes future shell edits safer on Windows.

## Commit 99: d61ed3bdfcde1e95eeb65ed232cd04ddfb1be34c

- Date: 2026-07-21
- Subject: editing 4 exp script to have more descriptive output names
- Size: `[tiny]`
- Per-file changes:
  - `scripts/shell/semantic_seg/sbatch_4_sem_seg_exp.sh`: Updated four-experiment output directory names to include experiment number, semantic-label source, 7-band WAC input, crop size, normalization source, loss configuration, train-sweep behavior, epoch count, and sweep sample count.
- Technical note: This makes long-running experiment output directories self-describing without needing to open `config.json` first.

## Commit 100: d53a247fc97382dc177b0710e68ee3505554b105

- Date: 2026-07-21
- Subject: adding copying sbatch helper
- Size: `[small]`
- Per-file changes:
  - `scripts/shell/semantic_seg/sbatch_publish_sem_seg_experiment.sh`: Added a CPU sbatch publishing helper that copies a selected semantic experiment output directory to the shared project experiment space, then applies recursive `chmod -R 755` and `chgrp -R j1123`.
- Technical note: This supports leaving a background publish/copy job running after selecting the best experiment output.

## Commit 101: e0a7d11d6e36fe8b5c1111527d2448c375132207

- Date: 2026-07-21
- Subject: fixing toy model embeddings
- Size: `[small]`
- Per-file changes:
  - `lfm/toy_model/inst_seg/dino_mask_rcnn_model.py`: Changed the old `"0.95*red"` compatibility branch so it uses plain red patch weights instead of scaling by 0.95.
  - `lfm/toy_model/inst_seg/iseg_dataset.py`: Changed WAC/NIR metadata assignment to emit `"red"` instead of `"0.95*red"` for toy instance segmentation.
  - `lfm/toy_model/inst_seg/iseg_model.py`: Changed the old `"0.95*red"` compatibility branch so it uses plain red patch weights instead of scaling by 0.95.
  - `lfm/toy_model/sem_seg/sseg_dataset.py`: Changed WAC/NIR metadata assignment to emit `"red"` instead of `"0.95*red"` for toy semantic segmentation.
  - `lfm/toy_model/sem_seg/sseg_model.py`: Changed the old `"0.95*red"` compatibility branch so it uses plain red patch weights instead of scaling by 0.95.
- Technical note: This removes the ad hoc 0.95 red scaling from toy DINO flexible input embeddings while keeping older `"0.95*red"` assignment strings loadable as plain red.

## Commit 102: d5d6cf2551f364ca78c931f03e5ccbddc3a712a7

- Date: 2026-07-21
- Subject: updating changelog
- Size: `[small]`
- Per-file changes:
  - `docs/ibm_model_commit_change_log.md`: Extended this changelog with entries for the late-July semantic experiment, normalization, task-layout, shell-script, publishing, and toy-embedding commits.
- Technical note: This is documentation-only history maintenance. It records the branch evolution through commit 101 without changing runtime behavior.

## Commit 103: 455449f345850026035e973cc0ce2ba04bae5dd6

- Date: 2026-07-21
- Subject: fixing checkpointing hopefully
- Size: `[small]`
- Per-file changes:
  - `lfm/full_model/inst_seg/instance_seg_finetuning.py`: Changed full-model instance checkpointing to avoid saving `last.ckpt` and save weights only for each epoch checkpoint.
  - `lfm/full_model/sem_seg/semantic_seg_finetuning.py`: Applied the same weights-only/no-last checkpoint behavior to full-model semantic fine-tuning.
  - `scripts/python/instance_seg/instance_seg_comparison.py`: Applied the same checkpoint-size reduction to the toy/comparison instance trainer.
  - `scripts/python/semantic_seg/semantic_seg_comparison.py`: Applied the same checkpoint-size reduction to the toy/comparison semantic trainer.
  - `scripts/shell/semantic_seg/sbatch_4_sem_seg_exp.sh`: Added a free-space preflight check for the four-experiment launcher, defaulting to a 500 GB minimum under `BASE_OUTPUT_PARENT`.
- Technical note: The checkpoint changes reduce storage pressure from Lightning checkpoint files by saving model weights only and avoiding duplicate `last.ckpt` artifacts.

## Commit 104: 85fb2f57bf38fb8efaccb1ee4c5c7c20364f36d9

- Date: 2026-07-21
- Subject: adding smaller orchestrator
- Size: `[tiny]`
- Per-file changes:
  - `scripts/shell/semantic_seg/sbatch_4_sem_seg_exp.sh`: Commented out the first three semantic experiment submissions and left only the fourth experiment active.
- Technical note: This turns the four-experiment launcher into a smaller one-experiment orchestrator, useful when storage or queue pressure makes launching all four variants impractical.

## Commit 105: 1b3c48c881d3a70338aec9e6b2496a51d0125224

- Date: 2026-07-21
- Subject: updating test plots
- Size: `[moderate]`
- Per-file changes:
  - `notebooks/full_model/Metrics_LFM.ipynb`: Added a new metrics-oriented notebook for reviewing Lunar-FM/Graha semantic test results.
  - `notebooks/full_model/sem_seg_test_metrics.ipynb`: Updated the semantic test metrics notebook and plot workflow.
- Technical note: The large line count is notebook JSON churn. Conceptually, this commit improves result inspection and plotting for semantic segmentation test metrics.

## Commit 106: 9e01eb01d8751c7e5c83e469173cd87fe6dee45a

- Date: 2026-07-22
- Subject: updating stats nb to have faster stats calc
- Size: `[moderate]`
- Per-file changes:
  - `notebooks/full_model/sem_seg_test_metrics.ipynb`: Updated the semantic metrics notebook to calculate statistics faster.
- Technical note: This is notebook-only performance work for analysis/metrics iteration, not a training-code or model-architecture change.

## Commit 107: c3961074b81dadf4dafb0ff6c3d072aa52843169

- Date: 2026-07-22
- Subject: adding AGENTS.md
- Size: `[small]`
- Per-file changes:
  - `AGENTS.md`: Added repo-level agent instructions describing the `lfm` repo purpose, local micromamba environment, current package/script/notebook layout, important semantic and instance workflows, HPC notes, experiment defaults, docs to read, and working rules.
- Technical note: This creates the working context document for future Codex/changelog agents. It also clarifies that the active repository root is the `lfm` directory and documents the current full-model/toy-model workflow expectations.

## Commit 108: 0a0a316a9bb0545c3a55aaf3d5bebc6f57f5e4b9

- Date: 2026-07-22
- Subject: Update agent working rules
- Size: `[tiny]`
- Per-file changes:
  - `AGENTS.md`: Added a working rule that agents should not generate code unless explicitly prompted with wording such as "start the refactor" or "generate code for x".
- Technical note: This tightens the agent operating contract so future agents can still inspect, review, document, and plan by default while avoiding unrequested code generation.

## Commit 109: cbdb54aad4da5d4cfa661e8647799768daafe274

- Date: 2026-07-22
- Subject: Update IBM model commit changelog
- Size: `[small]`
- Per-file changes:
  - `docs/ibm_model_commit_change_log.md`: Added changelog entries for commits 102 through 107, covering the prior changelog update, checkpointing/storage fixes, the smaller semantic experiment launcher, metrics notebook updates, faster stats calculation, and the initial `AGENTS.md` document.
- Technical note: This is documentation-only history maintenance and does not affect runtime code.

## Commit 110: b6e75cf58e888e5631f70d85fe9a23bdf7cd21f0

- Date: 2026-07-22
- Subject: plotting reorganization init commit
- Size: `[large]`
- Per-file changes:
  - `docs/sem_inst_refactor_opportunities.md`: Added a refactor/opportunity note for semantic and instance segmentation organization.
  - `lfm/full_model/all_tasks/utils/_plot_utils_impl.py`: Added the extracted plotting implementation while preserving existing plotting behavior during the initial split.
  - `lfm/full_model/all_tasks/utils/plot_utils.py`: Reduced the older monolithic plotting module substantially, shifting implementation into the new helper module.
  - `lfm/full_model/all_tasks/utils/callbacks.py`: Added shared callback-related utility exports/code for plotting workflows.
  - `lfm/full_model/all_tasks/utils/display.py`: Added shared display helpers for full-model result visualization.
  - `lfm/full_model/all_tasks/utils/metrics.py`: Added shared metric helper structure for plot/analysis code.
  - `lfm/full_model/all_tasks/utils/prediction_cache.py`: Added shared prediction-cache helper structure.
  - `lfm/full_model/all_tasks/utils/__init__.py`: Updated shared utility package exports for the reorganized plotting modules.
  - `lfm/full_model/inst_seg/__init__.py`: Updated instance package exports.
  - `lfm/full_model/inst_seg/plotting.py`: Added an instance-specific plotting wrapper/module.
  - `lfm/full_model/sem_seg/__init__.py`: Updated semantic package exports.
  - `lfm/full_model/sem_seg/plotting.py`: Added a semantic-specific plotting wrapper/module.
- Technical note: This is the first major step in breaking up plotting utilities into shared all-task helpers plus task-specific semantic and instance plotting entry points.

## Commit 111: eb6aa14cc6a49cd29f088a97368649bb14161e9e

- Date: 2026-07-22
- Subject: plotting restructure cleanup
- Size: `[large]`
- Per-file changes:
  - `lfm/full_model/all_tasks/utils/_plot_utils_impl.py`: Removed the temporary extracted plotting implementation from the initial split.
  - `lfm/full_model/all_tasks/utils/plot_utils.py`: Removed the old shared plotting module after moving plotting responsibilities into clearer shared and task-specific modules.
  - `lfm/full_model/all_tasks/utils/common.py`: Added shared common helpers used by the reorganized plotting/cache/metrics utilities.
  - `lfm/full_model/all_tasks/utils/callbacks.py`: Expanded and cleaned up shared callback utilities.
  - `lfm/full_model/all_tasks/utils/display.py`: Expanded display helpers after separating them from the old plotting module.
  - `lfm/full_model/all_tasks/utils/metrics.py`: Expanded metrics helpers for the reorganized result-analysis workflow.
  - `lfm/full_model/all_tasks/utils/prediction_cache.py`: Expanded shared prediction-cache support.
  - `lfm/full_model/all_tasks/utils/utils.py`: Updated imports or shared utility references for the new helper layout.
  - `lfm/full_model/all_tasks/utils/__init__.py`: Updated utility exports after removing old plotting modules and adding the new common/display/metrics/cache structure.
  - `lfm/full_model/inst_seg/plotting.py`: Removed the temporary instance plotting wrapper.
  - `lfm/full_model/inst_seg/instance_plotting.py`: Added the main instance-specific plotting implementation.
  - `lfm/full_model/inst_seg/instance_prediction_cache.py`: Added instance-specific prediction-cache support.
  - `lfm/full_model/inst_seg/instance_mask_datamodule.py`: Updated instance datamodule references for the reorganized helpers.
  - `lfm/full_model/inst_seg/instance_seg_finetuning.py`: Updated instance fine-tuning code to use the new plotting/cache organization.
  - `lfm/full_model/inst_seg/__init__.py`: Updated instance exports for the renamed task-specific modules.
  - `lfm/full_model/sem_seg/plotting.py`: Removed the temporary semantic plotting wrapper.
  - `lfm/full_model/sem_seg/semantic_plotting.py`: Added the main semantic-specific plotting implementation.
  - `lfm/full_model/sem_seg/semantic_prediction_cache.py`: Added semantic-specific prediction-cache support.
  - `lfm/full_model/sem_seg/semantic_seg_finetuning.py`: Updated semantic fine-tuning code to use the new plotting/cache organization.
  - `lfm/full_model/sem_seg/__init__.py`: Updated semantic exports for the renamed task-specific modules.
  - `scripts/python/instance_seg/instance_checkpoint_sweep.py`: Updated instance checkpoint sweep plotting/cache behavior for the reorganized modules.
- Technical note: This completes the plotting refactor direction started in commit 110 by replacing the temporary generic plotting split with explicit shared helpers plus task-specific `semantic_*` and `instance_*` modules.
