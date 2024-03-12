#!/bin/bash

CONDA_ENV_NAME='grad_attacks'
ORIGINAL_DIR=$(pwd)

CONDA_DIR=${HOME}/anaconda3
if [ ! -d ${CONDA_DIR} ]; then
  echo "[INFO] Install Anaconda Package Manager..."
  cd ~
  wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
  bash Anaconda3-2020.11-Linux-x86_64.sh -b -p ${CONDA_DIR}
  export PATH=${CONDA_DIR}/bin:$PATH
  rm Anaconda3-2020.11-Linux-x86_64.sh
  conda init bash
  cd $ORIGINAL_DIR  # important! return to project dir
else
  echo "[INFO] Anaconda already installed."
fi

source ~/anaconda3/etc/profile.d/conda.sh
pip install -U huggingface_hub
ENVS=$( conda env list | awk '{print $1}' )

if [[ $ENVS = *"${CONDA_ENV_NAME}"* ]]; then
  echo "[INFO] Environment ${CONDA_ENV_NAME} already exists."
else
  echo "[INFO] Create environment "${CONDA_ENV_NAME}"..."
  conda create -n ${CONDA_ENV_NAME} python=3.9 -y
  conda activate ${CONDA_ENV_NAME}

  echo "[INFO] Installing dependencies..."
  pip install -r requirements.txt --upgrade
  conda deactivate
fi
