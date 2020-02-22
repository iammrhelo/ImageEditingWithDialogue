# Updates on 2020-02-21

This repo contains code/data of our paper [Adjusting Image Attributes of Localized Regions with Low-level Dialogue](https://arxiv.org/abs/2002.04678).  If you find our paper useful, here's the bibtex
```
@article{lin2020adjusting,
  title={Adjusting Image Attributes of Localized Regions with Low-level Dialogue},
  author={Lin, Tzu-Hsiang and Rudnicky, Alexander and Bui, Trung and Kim, Doo Soon and Oh, Jean},
  journal={arXiv preprint arXiv:2002.04678},
  year={2020}
}
```

The 2 datasets published along with our paper are at the following locations
- 2.5k Image Edit Requests (tagged): `imadial-nlu/data`
- Visualizations of 83 Dialogues from our user study: `./vis`

If you aim to setup the system, I strongly recommend **NOT** to use MattNet, but look for other open-source implementations, as MattNet requires an older version of PyTorch (0.2.0) and Cuda (0.8.0), while the code base has multiple layers of submodules, which is extremely difficult to install (for me).  I was unable to prepare the 83 dialogues as a dataset because of this (sigh).

# ImageEditingWithDialogue

This repository contains code and data of ImaDial -- the system used in our [paper](https://arxiv.org/abs/2002.04678), and contains 3 separate repos ```./imadial-nlu```, ```./imadial-system```, ```./imadial-mattnet```. 

### Setup

To setup the system, you'll need to start the servers in each of the 3 directories.

#### NLU

cd into ```imadial-nlu```, and follow installation instructions in ```imadial-nlu/README.md```.

```bash
# Basic workflow here
cd imadial-nlu 
conda create -n imadial-nlu python=3.5 
conda activate imadial-nlu 
pip install -r requirements.txt 
# Trains a model and starts server
sh run.sh 
```

***Note*** model trained in ```imadial-nlu``` may not have same results as the model in our paper

#### MattNet

cd into ```imadial-mattnet```, and follow installation instructions in the ```imadial-mattnet/README.md```.  I forked the original MattNet repo and added a flask server script.  (To be honest, installing MattNet was ~~a pain in the ass~~ not straight forward for me.)

```bash
cd imadial-mattnet
# do a lot of MattNet installation
...
...
...
# After MattNet installation
cd imadial-mattnet/tools
# Starts server
python server.py --mode 2
```


#### System 

After setting up NLU and MattNet, cd into ```imadial-system``` and follow installation instructions in README.md

```bash
cd imadial-system 
conda create -n imadial python=3.5 
conda activate imadial 
pip install -r requirements.txt 
conda install -c menpo opencv 
unzip ./config/deploy/deploy_images.zip -d ./config/deploy
python serve.py 
```

### Deploy 

After starting all 3 servers, go to http://0.0.0.0:2000 and play!  

Sessions will be saved as pickled objects in ```imadial-system/pickled```.  To visualize dialogues, run the below command
```bash
cd imadial-system 
python visualize.py -s SESSION_PICKLE -o OUTPUT_PNG
```
