# Offline-to-Online Reinforcement learning with Directed Exploration Models

## How to run it

Paper results were collected with [MuJoCo 2.1.0](http://www.mujoco.org/) (and [mujoco-py 2.1.2.14](https://github.com/openai/mujoco-py)) in [OpenAI gym 0.23.1](https://github.com/openai/gym) with the [D4RL datasets](https://github.com/rail-berkeley/d4rl). Networks are trained using [PyTorch 2.0.1](https://github.com/pytorch/pytorch) and Python 3.9.

Run it by:

```shell
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 libboost-all-dev

- git clone https://github.com/zherunliu/DEM.git
+ git clone git@github.com:zherunliu/dem_mirror.git

cd ./dem_mirror

conda create --prefix ./.venv python=3.9 -y
conda activate ./.venv

wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xzf ./mujoco210-linux-x86_64.tar.gz
mkdir -p ~/.mujoco
mv ./mujoco210 ~/.mujoco/mujoco210

- # Add next 3 lines to ~/.bashrc
+ # Add next 3 lines to ~/.zshrc

- # vim ~/.bashrc
+ # vim ~/.zshrc

export CC=gcc
export CXX=g++
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin

- source ~/.bashrc
+ source ~/.zshrc

git checkout tc28/chore

pip install -r ./requirements.txt
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python run_dem.py
```
