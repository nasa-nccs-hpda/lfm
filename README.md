# Lunar Foundation Model Working Repository

Working repo for LFM project. See finetuning notebook for an example workflow.

## Quickstart

To test one of the example crater segmentation workflows:

1. Open an ADAPT terminal, and run these commands:
```bash
module load mamba
```
```bash
micromamba activate /explore/nobackup/projects/lfm/env
```
```bash
python -m ipykernel install --user --name=lfm_env --display-name="LFM_ENV"
```
2. Login to Explore JupyterHub: `https://jh-ml.nccs.nasa.gov`.
   1. Select the ILAB session for 6 hours with 1 V100 GPU.
   2. From the top right corner, click the text that says Python [...]. Select "Python [conda env: ilab-pytorch]" from the dropdown.
3. Navigate to a folder where you would like to run the example workflow.
4. Open a Terminal from JupyterHub.
5. Download the semantic segmentation notebook with:

   ```bash
   wget https://github.com/nasa-nccs-hpda/lfm/blob/develop/notebooks/semantic_seg.ipynb
   ```

    Or download the instance segmentation notebook with:
   ```bash
   https://github.com/nasa-nccs-hpda/lfm/blob/develop/notebooks/instance_seg.ipynb
   ```
7. Navigate to the top right where it says "Python 3 (ipykernel)" or something similar. Click on this, and select  ```LFM_ENV``` from the dropdown list.
8. Run the Notebook.

## Full Repository usage

To fully utilize this Repo, you will need a fine-grained access token:

1. Create an account:
   1. Create an account on github: https://github.com/signup. Connecting to Google account is usually pretty easy/convenient.
   2. After creating the account, sign in.
2. Create an access token: this will be needed to do things like "git clone" in the command-line.
   1. Generating a token:
      1. Click on your profile picture in the top right, click settings (2/3 of the way down the dropdown).
      2. On the settings page, note the different options on the left (public profile, account, etc). Scroll down all the way until you see "Developer Settings" in this left-hand menu.
      3. Click on developer settings.
      4. Click on personal access tokens, then select "tokens (classic)".
      5. Click "generate new token".
   2. Configuring the token:
      1. Name your token using the "note" field, set an expiration date as long as you would like (shorter is considered more secure).
      2. Scroll down to "select scopes".
      3. Check the main "repo" checkbox -- this allows you to modify repos.
      4. Scroll down further until you see "admin:org"; don't check this box, but do check "read" org option under this box. This allows you to access the lfm repo, which is part of a nasa organization.
   3. Confirm token creation: click "generate token" in green at the bottom of the screen. A screen will pop up to confirm your choice.
   4. Copy and paste the token, keep it somewhere secure like a password manager. This will be used instead of your password to login to github when using the CLI.
3. Clone the lfm repo:
   1. Verify you have access to the repo at this URL: ```https://github.com/nasa-nccs-hpda/lfm```. If you don't have access, let Sandy know.
   2. Go to an ADAPT terminal window; navigate to the lfm project space: ```cd /explore/nobackup/projects/lfm```
   3. Create your own subdirectory in the lfm space: ```mkdir <dir_name>```
   4. Enter your new directory: ```cd <dir_name>```
   5. To retrieve the code from the git repo, run: ```git clone https://github.com/nasa-nccs-hpda/lfm.git```. This should fetch the LFM code into your new directory.
   6. Start a PRISM JH session with 1 V100 GPU.
   7. Once the session has started, navigate to <dir_name>.
   8. Select the "Python [conda env: ilab-pytorch]" kernel in the top right of the screen where you see "Python 3 (ipykernel)".
   9. Read the introduction section of the notebook for information on the model, etc. If you want to modify the behavior of the notebook, various values can be changed under "configuration". Some standouts:
      - **INPUT_DIR**: if you want to load your own data. By default this path points to the data Sandy created in the LFM space. Data must be preprocessed to be the proper dimensions if loaded into the model.
      - **OUTPUT_DIR**: change this if you want to have the output of the workflow be somewhere specific.
   10. Click on the button on the top of the screen that looks like the "fast forward button" (restart kernel and run all cells". This will run the workflow!

## Model and data specifications

### Model specifications
The SAT-493M ViT-L/16 distilled DinoV3 encoder was used (trained on Satellite data). All encoder parameters were unfrozen for fine-tuning. See the [DinoV3 repo](https://github.com/facebookresearch/dinov3) for more info. For instance segmentation, the Mask2Former architecture was used as part of the encoder, on top of the DinoV3 SAT-493M encoder ([M2F Example](https://github.com/Carti-97/DINOv3-Mask2former), [M2F Website](https://arxiv.org/abs/2112.01527](https://mask2former.com/)).

### Data Specifications

#### Input Data
Input data was comprised of 2 UV bands and 5 vis bands. These were preprocessed by extracting all bands from the processed data geotiffs, matching them to the AOI of the (300, 300, 3) netCDF chips, and normalizing values to [0,1] range. Data was saved as georeferenced, 7-band geotiffs under the LFM project space.

#### Labels
Labels were processed from the annotations JSON file. Annotations were sorted by corresponding filename, then all labels for a given filename were saved single composite (300, 300) shape .npy images under the LFM project space.

#### Input/label matching
Labels and inputs were matched by asset ID, as well as tile row/column ID. Since the AOI of the original images were used, labels could be reused for the 7-band geotiffs.

#### Data locations
Data is kept under the LFM project space, under the ```/explore/nobackup/projects/lfm/model_inputs/300_300_inputs``` subdirectory. 3 band, 5_band vis data is kept there, as well as the primary 7-band vis/uv data. 3 and 5 band data is kept in .npy format, while 7-band is kept in .tif format. Data is separated into semantic segmentation and instance segmentation, due to their differing labels.

### Training specifications
The toy models were trained on 500 input/label pairs for 50 epochs, using a PRISM JupyterHub job on 4 V100 GPUs (1 V100 will also work, but will be slower). When adding bands (5/7 bands), a H100 node was used for its larger VRAM capacity. The parameters used were: "focal dice" loss function (Focal Loss + Dice loss), 5e-5 initial learning rate, AdamW optimizer, and Cosine Annealing LR scheduling with warmup. A train/val split of 80/20% was used as well, with early stopping patience of 5 epochs.
