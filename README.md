# Lunar Foundation Model Working Repository

Working repo for LFM project. Current workflows are found in the notebooks, listed in the quickstart section below.

## Quickstart

To run one of the notebooks:

1. Login to Explore JupyterHub: `https://jh-ml.nccs.nasa.gov` using your NCCS LDAP credentials.

2. Select the JupyterHub GPU profile: "([aarch64] 1 H100, 70 CPU Cores, 550GB Memory, 6 Hour Session)" from the dropdown menu. Click on "Start" after the selection.

3. As your session starts, your session should take you directly to a "Launcher" section. If, on the other hand, you are prompted to select a kernel, you can go ahead and click on "Select" within that window. If your session does not start, this could be related to waiting on available resources, and you will need to try again.

4. Use the file explorer interface on the left to navigate to a directory where you would like to run the example workflow. We suggest you use a directory in your $NOBACKUP space (e.g. /explore/nobackup/people/my_username/lfm).

**Note: in the file explorer interface, paths will be shown as `nobackup/...`, instead of something like `/explore/nobackup/people/my_username/...`. This is intended behavior in JupyterHub, and paths should still be referred to using the full `/explore/nobackup/people/my_username/...` format for notebooks and command-line operations.**

Feel free to create a new directory to run these workflows as well. To create a new directory, click on the "New Folder" icon in the upper left corner, and set the name of the new directory. Then double click on the directory to open it.

5. Open a Terminal from JupyterHub using the "Launcher" screen (it is open by default in a new Jupyter session). The Terminal option is at the very bottom of this screen with a "$_" symbol under the "Other" section.

6. From the newly opened Terminal:

    a. Make sure you are in the directory you intend to locate the code on. You can verify with the pwd command:

    ```bash
    pwd
    ```

    Assuming you wanted to be in the directory `/explore/nobackup/people/my_username/lunar_fm`, after running the `pwd` command, that directory should be the one shown in the terminal. If that is not the case, you will need to go to the intended directory using the `cd` command as shown below:

    ```bash
    cd /explore/nobackup/people/my_username/lunar_fm
    ```

   b. Now, you can retrieve the LFM code with this command (**note: you do not need a GitHub account to run this**):

      ```bash
      git clone https://github.com/nasa-nccs-hpda/lfm.git
      ```

   c. With the terminal still open, run the following command to set up your environment:

      ```bash
      cd lfm && bash scripts/shell/copy_kernel_graha_h100.sh
      ```

7. Reload the web page (by clicking the "⟳" button in your browser, or by pressing the F5 key) to finalize environment setup.

8. Close the terminal tab by clicking "x" on the top tab.

9. Using the file explorer interface again, navigate to the folder at: `<your_folder>/lfm/notebooks/`, where `<your_folder>` is the same one you created in step 6. Following the example from earlier, the full path would look like `/explore/nobackup/people/my_username/lunar_fm/lfm/notebooks`. The `notebooks/` folder contains Jupyter Notebooks for the IBM/"graha" model finetuning and inference workflows across two machine learning tasks (instance/semantic segmentation).

**Note: the structure of the folders is such that we have 2 lfm/ folders; the outermost lfm/ folder contains the notebooks/ directory.**

- The two finetuning notebooks available for the IBM/"graha" model are called instance_ibm_train.ipynb and semantic_ibm_train.ipynb. Each runs training for that machine learning task.
- The inference notebook for the IBM model is called inference_sseg.ipynb. It performs inference on the "data cubes" created from the LTM tiling scheme after the semantic finetuning notebook has been run. **This notebook requires you to manually set the checkpoint path to a previously created finetuning checkpoint. You need to both run the finetuning notebook, and change the GRAHA_LIGHTNING_CHECKPOINT variable in the inference notebook to run inference.**

**Note 2: toy model notebooks are still found under <your_folder>/lfm/notebooks/toy_model. These are no longer supported in this release.**

10. After navigating to the `<your_folder>/lfm/notebooks/` folder, open your notebook of choice by double-clicking it. If this is your first time opening the notebook, you will get a box asking to select a kernel profile. **Select "lfm_kernel"**. If this box does not appear automatically, click the kernel name in the top-right corner (it might display "Python 3" or similar), and select "lfm_kernel" from the dropdown menu.

**Verify that "lfm_kernel" now appears in the top-right corner.**

11. Run the notebook, by clicking on the restart button (looks like the fast-forward icon [>>]). You may see another dialog box pop up; if you do, click the red "restart" button to run the notebook. You should now see all cells of the notebook running in order, shown by the symbol [*] to the left of each notebook cell. **Note: only a singular notebook should be run at once, since the models take significant compute to run.**

## Dataset Specifications
There are 3 datasets currently supported by the LFM: WAC crater detection, NAC crater detection, and NAC IMP detection. Each has its own input data modality, so users need to configure what input bands will enter the LFM during training. Each dataset has a DATA_DICT variable; examples for each dataset are found under each dataset subsection below, as well as the supported tasks for each dataset (IMP only works for semantic segmentation).

**Note: if you would like to create and use your own dataset, reference the `docs/dataset_contribution.md` file.**

### More on the data dictionary

The notebook data dictionary tells the model how to match files and which stored chip bands to use. `selected_modalities` controls which frontend modalities are loaded from the chip, and backend modalities are inferred by default (`vis` -> `vis`, `uv` -> `uv`, `static` -> `static`, `pho` -> `nac`, `dtm` -> `dtm`). `band_filters` are modality-local indices. For example, WAC VIS has 5 stored VIS bands, so `"vis": [0, 1, 2, 3, 4]` selects all 5 VIS bands. NAC PHO and DTM are single-band modalities, so use `[0]` for each selected modality. File suffixes are inferred automatically from common terminal names such as `_input_nac_chip`, `_input_wac_chip`, `_input_wac_static_chip`, `_label`, `_mask`, `_mask_orig`, and `_img`; add `image_suffix` or `label_suffix` only for unusual datasets. Normalization source defaults to `"pretrain"`. Normalization modality is inferred from `dataset_modality`: `"wac"` uses internal `"vis_uv"`, while `"nac"` and `"nac_dtm"` use `"nac"`. Multi-modal features use `"concat"` merging by default; override `graha_vis_uv_merge_method` only when comparing merge strategies. For semantic segmentation, label source is inferred automatically: `.npz` labels are treated as instance archives and `.npy` labels are treated as semantic masks.

### WAC crater dataset
WAC crater data has 2 versions: the first consists of 5 VIS bands followed by 2 UV bands, saved as 7-band chips. The second contains the same VIS and UV bands, but also contains all 63 static data bands, for a total of 70 bands per chip. STATIC data has multiple nodata values in the latest code version, so you need to explicitly state which values are nodata in the data dictionary (see below).

**Both versions supports semantic segmentation and instance segmentation workflows.**

**For more info on the WAC + Static dataset, see docs/wac_static_bands.md**.

```python
DATA_DICT = {
    "dataset_name": "wac_craters",  # Human-readable dataset name (has no functional effect)
    "data_dir": "/explore/nobackup/projects/lfm/model_inputs/300_300_inputs/full_model_inst_seg_v2",  # Data directory on /explore
    "dataset_modality": "wac",  # Dataset modality
    "selected_modalities": ["vis", "uv"],  # Frontend modalities; backend modalities are inferred
    "band_filters": {  # Band filters for each modality
        "vis": [0, 1, 2, 3, 4],  # Vis channels to use (0-indexed)
        "uv": [0, 1],  # UV channels to use (0-indexed)
    },
}
```

```python
DATA_DICT = {
    "dataset_name": "wac_static_craters",  # Human-readable dataset label; does not change model behavior.
    "data_dir": "/explore/nobackup/projects/lfm/model_inputs/300_300_inputs/fm_all_static_all_wac_iseg_v3",  # Dataset root containing train/val/test split folders.
    "dataset_modality": "wac_static",  # Stored chip layout: 5 VIS + 2 UV + 63 static bands.
    "band_filters": {  # Modality-local band indices to keep from each selected modality.
        "vis": [0, 1, 2, 3, 4],  # Keep all 5 VIS bands.
        "uv": [0, 1],  # Keep both UV bands.
        "static": [  # Keep all 63 static bands; remove individual indices here for ablation tests.
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
            50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
            60, 61, 62,
        ],
    },
    "excluded_nodata_values": [  # Known NoData sentinels to ignore when building model masks.
        -32768.0,  # Shared Lunar FM datacube NoData value.
        -3.4028226550889045e38,  # Static-source Float32 sentinel variant.
        -3.4028230607370965e38,  # Static-source Float32 sentinel variant.
        -3.4028234663852886e38,  # Static-source Float32 sentinel variant.
    ],
}
```

### NAC crater dataset
NAC crater data includes a PHO/NAC image band and a paired DTM band. The PHO+DTM chips are saved as 2-band chips with band 1 = PHO and band 2 = DTM. **It supports instance segmentation and semantic segmentation from instance labels**.

```python
DATA_DICT = {
    "dataset_name": "nac_craters_pho_dtm",  # Human-readable dataset name (has no functional effect)
    "data_dir": "/explore/nobackup/projects/lfm/model_inputs/256_256_inputs/nac/nac_coco_inst_seg_pho_dtm",  # Data directory on /explore
    "dataset_modality": "nac_dtm",  # Dataset modality
    "selected_modalities": ["pho", "dtm"],  # Frontend modalities; backend maps PHO to NAC
    "band_filters": {  # Band filters for each modality
        "pho": [0],  # PHO channels to use (0-indexed)
        "dtm": [0],  # DTM channels to use (0-indexed)
    },
}
```

For PHO-only NAC crater experiments, use `selected_modalities=["pho"]` and remove the `dtm` entry from `band_filters`; the backend will infer `["nac"]`.

### NAC IMP dataset
NAC IMP data includes a single PHO/NAC-like image band and semantic labels. **It only supports semantic segmentation workflows.**

**Data dirs (there are 2 different IMP datasets supported):**
- Full IMP dataset (copied from /explore/nobackup/projects/lfm/processed_data/Lunar/data_release/IMP_dataset): `/explore/nobackup/projects/lfm/model_inputs/256_256_inputs/imp/old_imp/imp_sem_seg_clean_targets_only`
- Newer, smaller IMP dataset (copied from /explore/nobackup/projects/lfm/processed_data/Lunar/data_release/IMP_dataset_mike/): `/explore/nobackup/projects/lfm/model_inputs/256_256_inputs/imp/new_imp/imp_sem_seg_clean_targets_only`

```python
DATA_DICT = {
    "dataset_name": "nac_imp_old",  # Human-readable dataset name (has no functional effect)
    "data_dir": "/explore/nobackup/projects/lfm/model_inputs/256_256_inputs/imp/old_imp/imp_sem_seg_clean_targets_only",   # Data directory on /explore
    "dataset_modality": "nac",  # Dataset modality
    "selected_modalities": ["pho"],  # Frontend modality; backend maps PHO to NAC
    "band_filters": {  # Band filters for each modality
        "pho": [0],  # PHO channels to use (0-indexed)
    },
}
```

```python
DATA_DICT = {
    "dataset_name": "nac_imp_new",  # Human-readable dataset name (has no functional effect)
    "data_dir": "/explore/nobackup/projects/lfm/model_inputs/256_256_inputs/imp/new_imp/imp_sem_seg_clean_targets_only",  # Data directory on /explore
    "dataset_modality": "nac",   # Dataset modality
    "selected_modalities": ["pho"],  # Frontend modality; backend maps PHO to NAC
    "band_filters": {  # Band filters for each modality
        "pho": [0],  # PHO channels to use (0-indexed)
    },
}
```

## FAQ

### The outputs are too long/too short in the notebooks. How can I change this?

Jupyter notebooks have a feature where you can collapse or expand the outputs, which helps manage information in long notebooks, or helps to see figures or text outputs. To collapse/expand a notebook output:

1. Navigate to the cell whose output you're interested in.
2. Ensure that you've run the cell and that it's generated an output. You can verify this by looking to the left of the cell; it should say `[1]` (for the first cell, for instance), or the corresponding cell number.
3. Click to the left of the notebook cell, right where you see the cell number.
4. You will see two blue bars; one will be right next to the cell itself, and one will be next to the output (image, text, etc)
5. Hover to the right of the blue bar that's to the left of the output (this bar will be below the first blue bar).
6. You should see a gray rectangle next to the blue bar; click this to either collapse the output (creating a scrollable view) or expand the output (showing the full contents).
**Note**: clicking the blue bar next to the output will hide the output. You can reverse this by clicking on the blue bar again.

### I did something and the code in the cell no longer works/looks different. How can I reverse this?

You have likely changed the cell from code to markdown, or vice versa! Markdown is just another form of text that's easy to read and format, and can easily be converted back to code. To change the cell type, follow these steps:

1. Navigate to the cell that you wish to change.
2. Click to the left of the cell, where you see either `[ ]`, or the corresponding cell number in parentheses. For example, the first cell would have `[ ]` or `[1]`.
3. After clicking this area, you should see the cell highlighted with a blue outline.
4. At the top of the notebook, look for the dropdown that says either "Code" or "Markdown".
5. Click this dropdown, and select the appropriate cell type. The appropriate cell type for plain english is markdown, while everything else should be code.

## Collaborators
- **Mike Barker**: [michael.k.barker@nasa.gov](mailto:michael.k.barker@nasa.gov)
- **Vishnu Viswanathan**: [vishnu.viswanathan@nasa.gov](mailto:vishnu.viswanathan@nasa.gov)
- **Andrew Annex**: [annex@seti.org](mailto:annex@seti.org)
- **Ethan Schaefer**: [eschaefer@seti.org](mailto:eschaefer@seti.org)
- **Alexander Kerr**: [alexander.j.kerr@nasa.gov](mailto:alexander.j.kerr@nasa.gov)
- **Roger Gill**: [roger.l.gill@nasa.gov](mailto:roger.l.gill@nasa.gov)
- **Jordan Caraballo-Vega**: [jordan.a.caraballo-vega@nasa.gov](mailto:jordan.a.caraballo-vega@nasa.gov)
- **Mark Carroll**: [mark.carroll@nasa.gov](mailto:mark.carroll@nasa.gov)
