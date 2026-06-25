# Lunar Foundation Model Working Repository

Working repo for LFM project. Current workflows are found in the notebooks, listed in the quickstart section below.

## Quickstart

To run one of the notebooks:

1. Login to Explore JupyterHub: `https://jh-ml.nccs.nasa.gov` using your NCCS LDAP credentials.
2. Select the JupyterHub GPU profile: "[aarch64] 1 H100, 70 CPU Cores, 550GB Memory, 6 Hour Session" from the dropdown menu. Click on "Start" after the selection.
3. As your session starts, your session should take you directly to a "Launcher" section. If, on the other hand, you are prompted to select a kernel, you can go ahead and click on "Select" within that window. If your session does not start, this could be related to waiting on available resources, and you will need to try again.
4. Use the file explorer interface on the left to navigate to a directory where you would like to run the example workflow. We suggest you use a directory in your $NOBACKUP space (e.g. /explore/nobackup/people/your_username/lfm). Feel free to create a new directory to run these workflows as well. To create a new directory, click on the directory icon in the upper left corner, and set the name of the new directory.
5. Open a Terminal from JupyterHub using the "Launcher" screen (it is open by default in a new Jupyter session). The Terminal option is at the very bottom of this screen with a "$_" symbol under the "Other" section.

6. From the newly opened Terminal:
    a. Make sure you are in the directory you intend to locate the code on. You can verify with the pwd command. 
       ```bash
           pwd
       ```
      Assuming I wanted to be in the directory `/explore/nobackup/people/my_username/lfm`, after running the `pwd` command, that directory should be the one shown in the terminal. If that is not the case, you will need to go to the intended directory using the `cd` command as shown below:
       ```bash
           cd /explore/nobackup/people/my_username/lfm
       ```

   b. Now, you can retrieve the LFM code with this command (**note: you do not need a GitHub account to run this**):
      ```bash
      git clone https://github.com/nasa-nccs-hpda/lfm.git
      ```
   c. With the terminal still open, run the following command to set up your environment:
      ```bash
      cd lfm && bash copy_kernel.sh
      ```
7. Close the terminal tab by clicking "x".
8. Using the file explorer interface again, navigate to the folder at: `lfm/notebooks`. This contains Jupyter Notebooks for different steps of the toy model workflows, for the two machine learning tasks (instance/semantic segmentation)
   - The two toy model notebooks are called `instance_seg_train.ipynb` and `semantic_seg_train.ipynb`, respectively. These are the notebooks to create/train the model for those tasks.
   - The inference notebook, `inference_sseg.ipynb`, only works after running the **semantic segmentation training notebook**, which saves a model "checkpoint" file to disk. This checkpoint will be used to load the model, and perform inference on new data.
   - The other two notebooks, `tiling_example.ipynb` and `chip_example.ipynb`, are used as examples for how we created the **semantic segmentation training dataset** used for the training notebooks. These will run limited examples of tiling/chip creation.
9. After navigating to the `lfm/notebooks` folder, open your notebook of choice by double-clicking it. If this is your first time opening the notebook, you will get a box asking to select a kernel profile. **Select "lfm_container"**. If this box does not appear automatically, click the kernel name in the top-right corner (it might display "Python 3" or similar), and select "lfm_container" from the dropdown menu. **Verify that "lfm_container" now appears in the top-right corner**
10. Run the notebook, using the button that looks like the fast-forward icon (>>). Click the red "Restart" button. This will execute all the cells from the notebook in order. One notebook should be run at a time.

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

## Collaborators
- **Mike Barker**: [michael.k.barker@nasa.gov](mailto:michael.k.barker@nasa.gov)
- **Vishnu Viswanathan**: [vishnu.viswanathan@nasa.gov](mailto:vishnu.viswanathan@nasa.gov)
- **Andrew Annex**: [andrew.m.annex@nasa.gov](mailto:andrew.m.annex@nasa.gov)
- **Alexander Kerr**: [alexander.j.kerr@nasa.gov](mailto:alexander.j.kerr@nasa.gov)
- **Roger Gill**: [roger.l.gill@nasa.gov](mailto:roger.l.gill@nasa.gov)
- **Jordan Caraballo-Vega**: [jordan.a.caraballo-vega@nasa.gov](mailto:jordan.a.caraballo-vega@nasa.gov)
- **Mark Carroll**: [mark.carroll@nasa.gov](mailto:mark.carroll@nasa.gov)
