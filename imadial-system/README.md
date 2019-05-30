# ImaDial-System

This repo contains the code for the main system of ImaDial


### Installation

Python: 3.5 

```bash
# install requirements and opencv
pip install -r requirements.txt 
conda install -c menpo opencv 
```

### Serve 

```bash
# Unzip images
unzip ./config/deploy/deploy_images.zip -d ./config/deploy
# Run server
python serve.py 
```

Then go to http://0.0.0.0:2000/ to see if everything is working properly


#### Configuration

Specify the correct endpoints in ```config/deploy/rule.json```.  By default, NLU and MattNet's uri will be
- Vision Engine (MattNet): http://0.0.0.0:6000/nlu
- NLU: http://0.0.0.0:5000/nlu
- image_dir: ```./config/deploy/deploy_images```

