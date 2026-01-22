Working repo for LFM project. See finetuning notebook for an example workflow. 

To use this Repo, you will need a fine-grained access token: 

1) Create an account: 
  a. Create an account on github: https://github.com/signup. Connecting to Google account is usually pretty easy/convenient. 
  b. After creating the account, sign in.
2) Create an access token: this will be needed to do things like "git clone" in the command-line. 
  a. Generating a token:
    i. Go to the profile picture in the top right, click settings (2/3 of the way down the dropdown). 
    ii. On the settings page, note the different options on the left (public profile, account, etc). Scroll down all the way until you see "Developer Settings" in this left-hand menu. 
    iii. Click on developer settings. 
    iv. Click on personal access tokens, then select "fine-grained" tokens. 
    v. Click "generate new token".
  b. Configuring the token:
    i. Name your token, set an expiration date as long as you would like. Generally a shorter duration is considered more secure, I usually go with 30 days.
    ii. Scroll down to "repository access", click "all repositories". 
    iii. Scroll down to "permissions", click "add permissions". 
    iv. Search for "contents" and click the checkbox: an option for contents will pop up, select "read and write" from the dropdown. Don't worry about the "metadata" option.
  c. Confirm token creation: click "generate token" in green at the bottom of the screen. A screen will pop up to confirm your choice.
  d. Copy and paste the token, keep it somewhere secure like a password manager. This will be used instead of your password to login to github when using the CLI. 
3) Clone the lfm repo: 
  a. Verify you have access to the repo at this URL: https://github.com/nasa-nccs-hpda/lfm. If you don't have access, let Sandy know.  
  b. Go to an ADAPT terminal window; navigate to the lfm project space: cd /explore/nobackup/projects/lfm
  c. Create your own subdirectory in the lfm space: ```mkdir <dir_name>```
  d. Enter your new directory: ```cd <dir_name>```  
  e. To retrieve the code from the git repo, run: ```git clone https://github.com/nasa-nccs-hpda/lfm.git```. This should fetch the LFM code into your new directory. 
  f. Start a PRISM JH session with 1 V100 GPU.
  g. Once the session has started, navigate to <dir_name>.
  h. Select the "Python [conda env: ilab-pytorch]" kernel in the top right of the screen where you see "Python 3 (ipykernel)".
  i. Read the introduction section of the notebook for information on the model, etc. If you want to modify the behavior of the notebook, various values can be changed under "configuration". Some standouts:
    - **INPUT_DIR**: if you want to load your own data. By default this path points to the data Sandy created in the LFM space. Data must be preprocessed to be the proper dimensions if loaded into the model. 
    - **OUTPUT_DIR**: change this if you want to have the output of the workflow be somewhere specific.
  j. Click on the button on the top of the screen that looks like the "fast forward button" (restart kernel and run all cells". This will run the workflow!