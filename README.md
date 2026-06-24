# Lunar Foundation Model Working Repository

Working repo for LFM project. Current workflows are found in the notebooks, listed in the quickstart section below.

## Collaborators
- **Mike Barker**: [michael.k.barker@nasa.gov](mailto:michael.k.barker@nasa.gov)
- **Vishnu Viswanathan**: [vishnu.viswanathan@nasa.gov](mailto:vishnu.viswanathan@nasa.gov)
- **Andrew Annex**: [andrew.m.annex@nasa.gov](mailto:andrew.m.annex@nasa.gov)
- **Alexander Kerr**: [alexander.j.kerr@nasa.gov](mailto:alexander.j.kerr@nasa.gov)
- **Roger Gill**: [roger.l.gill@nasa.gov](mailto:roger.l.gill@nasa.gov)
- **Jordan Caraballo-Vega**: [jordan.a.caraballo-vega@nasa.gov](mailto:jordan.a.caraballo-vega@nasa.gov)
- **Mark Carroll**: [mark.carroll@nasa.gov](mailto:mark.carroll@nasa.gov)

## Quickstart

To test one of the example crater segmentation workflows:

1. Login to Explore JupyterHub: `https://jh-ml.nccs.nasa.gov`.
2. Select the JupyterHub GPU session: "[aarch64] 1 H100..." for 6 hours.
3. Use the file explorer interface on the left to navigate to a folder where you would like to run the example workflow. Feel free to create a new folder to run these workflows as well!
4. Open a Terminal from JupyterHub using the "launcher" screen (is open by default in a new Jupyter session). The terminal option is at the very bottom of this screen.
5. Retrieve the LFM code with this command (**note: you do not need a GitHub account to run this, it will work for anyone on the JupyterHub**):
   ```bash
   git clone https://github.com/nasa-nccs-hpda/lfm.git
   ```
6. Using the file explorer interface again, navigate to the folder at: `lfm/notebooks`. This contains Jupyter Notebooks for different steps of the toy model workflows, for the two machine learning tasks (instance/semantic segmentation)
   - The two toy model notebooks are called `instance_seg_train.ipynb` and `semantic_seg_train.ipynb`, respectively. These are the notebooks to create/train the model for those tasks.
   - The inference notebook, `inference_sseg.ipynb`, only works after running the semantic segmentation notebook, which saves a model "checkpoint" file to disk. This checkpoint will be used to load the model, and perform inference on new data.
   - The other two notebooks, `tiling_example.ipynb` and `chip_example.ipynb`, are used as examples for how we created the training dataset used for the training notebooks. These will run limited examples of tiling/chip creation.
7. After navigating to the `lfm/notebooks` folder, open your notebook of choice by double-clicking it.
8. Before running the notebook, you must select the kernel environment:

   a. Look for a dropdown menu to select "lfm_container" as your kernel:
      - OPTION 1: This dropdown may appear automatically when you first open the notebook
      - OPTION 2: If it doesn't appear automatically, click the kernel name in the top-right
      corner (it might display "Python 3" or similar)

   b. From the dropdown list, select "lfm_container"

   c. Verify that "lfm_container" now appears in the top-right corner

9. Run the notebook, using the button that looks like the fast-forward icon (>>).


## Model and data specifications

### Model specifications
The SAT-493M ViT-L/16 distilled DinoV3 encoder was used (trained on Satellite data). All encoder parameters were unfrozen for fine-tuning. See the [DinoV3 repo](https://github.com/facebookresearch/dinov3) for more info. For instance segmentation, the Mask2Former architecture was used as part of the encoder, on top of the DinoV3 SAT-493M encoder ([M2F Example](https://github.com/Carti-97/DINOv3-Mask2former), [M2F Website](https://arxiv.org/abs/2112.01527](https://mask2former.com/)).

### Data Specifications

#### Input Data
Input data was comprised of 2 UV bands and 5 vis bands, as well as the 5 KAGUYA static bands. These were preprocessed by extracting all bands from the processed data geotiffs, matching them to the AOI of the (300, 300, 3) netCDF chips, and normalizing values to [0,1] range. Data was saved as georeferenced, variable-band geotiffs under the LFM project space. See "data locations" below for more info on bands.

#### Labels
Labels were processed from the annotations JSON file. Annotations were sorted by corresponding filename, then all labels for a given filename were saved single composite (300, 300) shape .npy images under the LFM project space.

#### Input/label matching
Labels and inputs were matched by product ID, as well as tile row/column ID. Since the AOI of the original images were used, labels could be reused for the 12-band geotiffs.

#### Data locations
Data is kept under the LFM project space, under the ```/explore/nobackup/projects/lfm/model_inputs/300_300_inputs``` subdirectory. 3 band and 5 band vis data is kept there, as well as the 7-band vis/uv data, and the 12-band vis/uv/static (kaguya) data. All images (chips) are kept in .tif format, while labels are kept in .npz format.

### Training specifications
The toy models were trained on 500 input/label pairs for 50 epochs, using a PRISM JupyterHub job on 1 H100 GPU, chosen over a V100 for its larger VRAM capacity. The parameters used were: "focal dice" loss function (Focal Loss + Dice loss), 5e-5 initial learning rate, AdamW optimizer, and Cosine Annealing LR scheduling with warmup. A train/val split of 80/20% was used as well, and training was run for 100 epochs with no early stopping.
