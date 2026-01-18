#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/root/Workspace/python_study/src/tmm"
MODEL_DIR="${REPO_DIR}/triton_model_repository/ttm/1"
VENV_DIR="${MODEL_DIR}/venv"
TAR_PATH="${REPO_DIR}/triton_model_repository.tar.gz"

if [[ "${1:-}" == "--clean" ]]; then
  rm -rf "${VENV_DIR}"
  echo "Removed ${VENV_DIR}"
  exit 0
fi

if ! command -v python3.10 >/dev/null 2>&1; then
  apt-get update
  apt-get install -y software-properties-common
  add-apt-repository -y ppa:deadsnakes/ppa
  apt-get update
  apt-get install -y python3.10-venv
fi

rm -rf "${VENV_DIR}"
python3.10 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install -r "${MODEL_DIR}/requirements.txt"

rm -rf "${MODEL_DIR}/python"

tar -czf "${TAR_PATH}" -C "${REPO_DIR}/triton_model_repository" ttm
echo "Built ${TAR_PATH}"
