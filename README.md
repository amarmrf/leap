# dsse-group-project

## GCP VM Instance Information

- Account: `galacticrelic@gmail.com`
- Project: `DSSE`
- Region: `us-west4` (Las Vegas)

- VM Instance Name: `dsse-vm-2`
- Disk (500 GB): `dsse-disk-2`

- Conda Environment: `py12`

- `su` password: `password`

### Getting Started

1. Make sure you are in `/mnt/data/docs`
- Run `cd /mnt/data/docs`

2. Make sure you have the right Conda environment
- Run `conda activate py12`

3. Create a folder for yourself under `/mnt/data/docs`
- Run `mkdir <folder-name>`

4. Make sure you follow the setup below on your initial login

5. Create a new COnda Env for yourself using:
- `conda create -n <env-name> python=<version>` 

You're all set!

### Setup

The VM has a dedicated hard disk attached to it (apart from boot disk). We will use the dedicated disk (not boot disk) for our project as much as we can.

IMPORTANT: As soon as you ssh into the system, you should be in the folder `/mnt/data/docs`. If you are not in the folder already, please `cd` there.

#### File System

Add new user to the group `team` by running:
`sudo usermod -a -G team pkotchav`

#### BashRC 

```sh
# Add mounted disk folders to path
export TMPDIR="/mnt/data/tmp"
export PATH="/sbin:/mnt/data/bin:/mnt/data/lib:$PATH"

# Change directory to mounted disk folder
cd /mnt/data/docs

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/mnt/data/bin/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/mnt/data/bin/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/mnt/data/bin/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/mnt/data/bin/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```

#### Enable SSH
1. On your local terminal, run: 
- `ssh-keygen -t rsa -f ~/.ssh/gcloud-galacticrelic -C <username>`
- `chmod 700 ~/.ssh/gcloud-galacticrelic`
- `chmod 600 ~/.ssh/gcloud-galacticrelic`

2. On browser SSH into the VM, run:
- `chmod 700 /home/<username>/.ssh`
- `chmod 600 /home/<username>/.ssh/authorized_keys`

3. Copy public key generated locally at `~/.ssh/gcloud-galacticrelic.pub` to `~/.ssh/authorized_keys` on the VM

4. Login to the VM from your local terminal using: `ssh -i <path-to-private-key> <username>@ExternalIP`

Note: `ExternalIP` is on the VM Instances page on GCP for the VM. **It will change each time the VM is rebooted.**

#### Conda

I like to use Anaconda3 to setup the virtual python environment and then use pip inside the conda environment. 

Run `conda activate py12` to activate a `python 3.12.0` environment that I already created. Please use this environment for training/testing.

#### HuggingFace Access

My HuggingFace token is currently saved to disk. It should work if you run any files that import libraries related to `llama`

### Llama test

I tested `llama-3.2-1b-instruct` int the folder `/mnt/data/docs/test-llm` using GPU acceleration. It ran successfully and displayed the following output:

```
model.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.47G/2.47G [00:58<00:00, 42.1MB/s]
generation_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [00:00<00:00, 1.27MB/s]
tokenizer_config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 54.5k/54.5k [00:00<00:00, 9.28MB/s]
tokenizer.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9.09M/9.09M [00:00<00:00, 19.7MB/s]
special_tokens_map.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 296/296 [00:00<00:00, 2.30MB/s]
Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
{'role': 'assistant', 'content': "Arrrr, I be Captain Cutlass, the scurvy dog o' the digital seas! Me and me crew o' chatbots be sailin' through the vast ocean o' knowledge, answerin' yer questions and swabbin' the decks o' ignorance. Me heart be made o' gold, me brain be full o' wit, and me love be for the sea o' information. So hoist the sails and set course fer a day o' learnin' with ol' Captain Cutlass!"}
```

You can run the same test by running `python test.py` once you have activated the conda environment.


## Run score

```
python run.py --task MATH --model_variant meta-llama/Llama-3.2-1B-Instruct --data_path ./data --output_dir ./outputs --mixed_precision --no_bleu --no_rouge
```
