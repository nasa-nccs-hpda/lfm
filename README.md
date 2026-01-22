# Lunar Foundation Model Working Repository

Working repo for LFM project. See finetuning notebook for an example workflow. 

## Quickstart

To test one of the example crater segmentation workflows:

1. Login to Explore JupyterHub: `https://jh-ml.nccs.nasa.gov`.
   1. Select the ILAB session for 6 hours with 1 V100 GPU.
   2. From the top right corner, click the text that says Python [...]. Select "Python [conda env: ilab-pytorch]" from the dropdown.
2. Navigate to a folder where you would like to run the example workflow. 
3. Open a Terminal from JupyterHub.
4. Download the Notebook with

   ```wget https://raw.githubusercontent.com/nasa-nccs-hpda/lfm/refs/heads/main/notebooks/finetune_dinov3.ipynb```.
6. Run the Notebook.

## Full Repository usage

To fully utilize this Repo, you will need a fine-grained access token: 

1. Create an account: 
   1. Create an account on github: https://github.com/signup. Connecting to Google account is usually pretty easy/convenient. 
   2. After creating the account, sign in.
2. Create an access token: this will be needed to do things like "git clone" in the command-line. 
   1. Generating a token:
      1. Go to the profile picture in the top right, click settings (2/3 of the way down the dropdown). 
      2. On the settings page, note the different options on the left (public profile, account, etc). Scroll down all the way until you see "Developer Settings" in this left-hand menu. 
      3. Click on developer settings. 
      4. Click on personal access tokens, then select "fine-grained" tokens. 
      5. Click "generate new token".
   2. Configuring the token:
      1. Name your token, set an expiration date as long as you would like. Generally a shorter duration is considered more secure, I usually go with 30 days.
      2. Scroll down to "repository access", click "all repositories". 
      3. Scroll down to "permissions", click "add permissions". 
      4. Search for "contents" and click the checkbox: an option for contents will pop up, select "read and write" from the dropdown. Don't worry about the "metadata" option.
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
The SAT-493M ViT-L/16 distilled DinoV3 encoder was used (trained on Satellite data). All encoder parameters were unfrozen for fine-tuning. See the [DinoV3 repo](https://github.com/facebookresearch/dinov3) for more info. 

#### Input data specifications
The vis data, (hosted at /explore/nobackup/projects/lfm/rawdata/Lunar/LowRes_MLDataset_v1_bilinear), was preprocessed by extracting the following bands and normalizing values to [0,1] range: [643, 566, 415]. Data was saved in (3, 300, 300) shape .npy files under the LFM project space (explore/nobackup/projects/lfm/vis_chips). 

#### Label specifications
Labels were processed from the annotations JSON file. Annotations were sorted by corresponding filename, then all labels for a given filename were saved single composite (300, 300) shape .npy images under the LFM project space (explore/nobackup/projects/lfm/vis_chips). 

#### Input/label matching
Labels and inputs were matched by asset ID, as well as tile row/column ID. 

#### Training specifications
Model was trained on 500 input/label pairs for 50 epochs, using a PRISM JupyterHub job on 4 V100 GPUs (1 V100 will also work, but will be slower). The parameters used were: "combined" loss function (Dice loss + Binary CE), 1e-4 LR, AdamW optimizer, and Cosine Annealing LR scheduling. A train/val split of 80/20% was used as well.
