#!/bin/bash

wget https://www.roboti.us/download/mujoco200_linux.zip

unzip mujoco200_linux.zip
mkdir .mujoco
cp -r mujoco200_linux .mujoco
cp -r mujoco200_linux .mujoco/mujoco200
touch .mujoco/mjkey.txt

git clone --branch v2 https://github.com/BrunoBSM/mjrl.git

cd mjrl
conda env create -f setup/env.yml
source activate mjrl-env
export MUJOCO_PY_FORCE_CPU=True
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200_linux/bin
pip install mujoco-py<2.1,>=2.0

pip install -e .

