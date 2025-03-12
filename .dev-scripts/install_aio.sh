#!/usr/bin/env bash

# this script attempts to install a conda environment for green-bit-llm without manual intervention
# (environment) variables can be set before running

set -o xtrace
set -o errexit

trap exit INT

#=================================#
# 0. basic settings               #
#=================================#
: "INSTALL_DIR:" "${INSTALL_DIR:="${HOME}/gbai_workspace"}"


#=================================#
# 1. install miniconda            #
#=================================#
: "CONDA_DIR:"           "${CONDA_DIR:="/opt/conda"}"
: "CONDA_INSTALLER:"     "${CONDA_INSTALLER:="Miniconda3-latest-Linux-x86_64.sh"}"
: "CONDA_INSTALLER_URL:" "${CONDA_INSTALLER_URL:="https://repo.anaconda.com/miniconda/${CONDA_INSTALLER}"}"

mkdir -p "${CONDA_DIR}"
wget -q "${CONDA_INSTALLER_URL}" -O "${CONDA_DIR}/miniconda.sh"
#curl -sL "${CONDA_INSTALLER_URL}" -o "${CONDA_DIR}/miniconda.sh"
bash "${CONDA_DIR}/miniconda.sh" -f -b -u -p "${CONDA_DIR}"
rm "${CONDA_DIR}/miniconda.sh"

#=================================#
# 2. setup base conda environment #
#=================================#
: "CONDA_ENV_PREFIX:"      "${CONDA_ENV_PREFIX:="${INSTALL_DIR}/conda_env"}"
: "CONDA_CUSTOMIZATION:"   "${CONDA_CUSTOMIZATION:="true"}"
: "PYTHON_VERSION:"        "${PYTHON_VERSION:="3.10"}"
: "CUDA_VERSION:"          "${CUDA_VERSION:="12.1.0"}"
: "CUSTOM_TORCH_PKG_PATH:" "${CUSTOM_TORCH_PKG_PATH:="https://packages.greenbit.ai/whl/cu121/torch/torch-2.3.0-cp310-cp310-linux_x86_64.whl"}"
: "BIE_PKG_PATH:"          "${BIE_PKG_PATH:="https://packages.greenbit.ai/whl/cu121/bitorch-engine/bitorch_engine-0.2.6-cp310-cp310-linux_x86_64.whl"}"
# environment variable MAX_JOBS can be set to limit amount of RAM needed during flash-attn installation

set +o xtrace
eval "$("${CONDA_DIR}/bin/conda" shell.bash hook)"
conda init
conda create -y --prefix "${CONDA_ENV_PREFIX}" python="${PYTHON_VERSION}"
conda activate "${CONDA_ENV_PREFIX}"
if [ "${CONDA_CUSTOMIZATION}" = "true" ]; then
    echo "conda activate '${CONDA_ENV_PREFIX}'" >> "${HOME}/.bashrc"
fi

set -o xtrace
conda install -y -c "nvidia/label/cuda-${CUDA_VERSION}" cuda-toolkit
python --version
pip --version
pip install "${CUSTOM_TORCH_PKG_PATH}" "${BIE_PKG_PATH}"
pip install green-bit-llm
pip install flash-attn --no-build-isolation

#=================================#
# 3. clean up                     #
#=================================#
pip cache purge
