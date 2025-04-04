## conda setup

```
conda create --name carla0910 python=3.7.7 numpy

conda activate carla0910

pip install torch torchvision opencv-python gym stable-baselines3

pip install 'shimmy>=0.2.1'

pip install tqdm rich
```

## carla pythonAPI

make sure 'carla-0.9.10-py3.7-linux-x86_64.egg' is in the folder

## how to run
```
# Launch Carla
# low quality is not needed if RAM and GPU is capable
./CarlaUE4.sh -quality-level=Low

# if run into rendering problem, try
./CarlaUE4.sh -opengl -quality-level=Low

Under the created conda environment, the python client files (train.py, test.py) can be run
```
## others

'dqn_carla.zip' is the trained model

