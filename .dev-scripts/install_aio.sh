#!/usr/bin/env bash

# this script attempts to install an environment for green-bit-llm without manual intervention
# (environment) variables can be set before running

set -o xtrace
set -o errexit

trap exit INT

: "${WORK_DIR:="${HOME}/.gbai-tmp"}"
: "${INSTALL_DIR:="${HOME}/gbai_workspace"}"


mkdir -p "${WORK_DIR}"
pushd "${WORK_DIR}"


########################
# 1. install miniconda #
########################
: "${MINICONDA_INSTALLER:="Miniconda3-latest-Linux-x86_64.sh"}"
: "${MINICONDA_DIR:="${HOME}/miniconda"}"

wget "https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}" -O miniconda.sh
bash miniconda.sh -f -b -u -p "${MINICONDA_DIR}"


###################################
# 2. setup base conda environment #
###################################
: "${CONDA_ENV_PREFIX:="${INSTALL_DIR}/conda-env"}"
: "${PYTHON_VERSION:="3.10"}"
: "${CUDA_VERSION:="12.1.0"}"
: "${FA2_MAX_JOBS:="8"}"
: "${CUSTOM_TORCH_PKG_PATH:="https://packages.greenbit.ai/whl/cu121/torch/torch-2.3.0-cp310-cp310-linux_x86_64.whl"}"
: "${BIE_PKG_PATH:="https://packages.greenbit.ai/whl/cu121/bitorch-engine/bitorch_engine-0.2.6-cp310-cp310-linux_x86_64.whl"}"

conda create -y --prefix "${CONDA_ENV_PREFIX}" python="${PYTHON_VERSION}"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_PREFIX}"
conda install -y -c "nvidia/label/cuda-${CUDA_VERSION}" cuda-toolkit
pip install "${CUSTOM_TORCH_PKG_PATH}" "${BIE_PKG_PATH}"
pip install green-bit-llm
MAX_JOBS="${FA2_MAX_JOBS}" pip install flash-attn --no-build-isolation

popd
