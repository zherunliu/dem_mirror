# Offline-to-Online Reinforcement learning with Directed Exploration Models

## How to run it

Paper results were collected with [MuJoCo 2.1.0](http://www.mujoco.org/) (and [mujoco-py 2.1.2.14](https://github.com/openai/mujoco-py)) in [OpenAI gym 0.23.1](https://github.com/openai/gym) with the [D4RL datasets](https://github.com/rail-berkeley/d4rl). Networks are trained using [PyTorch 2.0.1](https://github.com/pytorch/pytorch) and Python 3.9.

Run it by:

```shell
git clone https://github.com/zherunliu/DEM.git

conda create --prefix .venv python=3.9 -y
conda activate .venv

pip install -r ./requirements.txt
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl

wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xzf ./mujoco210-linux-x86_64.tar.gz
rm -rf ~/.mujoco && mkdir ~/.mujoco
mv ./mujoco210 ~/.mujoco/mujoco210

# Add next line to ~/.bashrc
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin

python run_dem.py
```
