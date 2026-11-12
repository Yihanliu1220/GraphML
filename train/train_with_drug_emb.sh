#!/bin/bash
#SBATCH --job-name=train_knowddi_emb
#SBATCH --partition=gpu_b200
#SBATCH --ntasks=1 --nodes=1
#SBATCH --mem-per-cpu=150G
#SBATCH --time=24:00:00
#SBATCH --gpus=b200:1
#SBATCH --account=pi_hx235
#SBATCH --mail-type=ALL

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate knowddi_env

export HF_DATASETS_CACHE=/home/zc347/project_pi_hx235/zc347/cache/huggingface/datasets
export TRANSFORMERS_CACHE=/home/zc347/project_pi_hx235/zc347/cache/huggingface/transformers

REPO_DIR=/home/zc347/project/KnowDDI
PYTORCH_DIR="${REPO_DIR}/pytorch"
DEFAULT_GPU=0

EMB_CHOICE="${1:-gemma}"
CUSTOM_EMB_PATH="${2:-}"
EXPERIMENT_NAME="${3:-}"
GPU_ID="${4:-$DEFAULT_GPU}"

case "${EMB_CHOICE}" in
  gemma)
    DRUG_EMB_PATH="${REPO_DIR}/data/drugbank/drug_embeddings_embedding_gemma.pt"
    DEFAULT_EXPERIMENT_NAME="Drugbank_GemmaEmb"
    ;;
  llama)
    DRUG_EMB_PATH="${REPO_DIR}/data/drugbank/drug_embeddings_embedding_llama.pt"
    DEFAULT_EXPERIMENT_NAME="Drugbank_LlamaEmb"
    ;;
  custom)
    if [[ -z "${CUSTOM_EMB_PATH}" ]]; then
      echo "Usage for custom embeddings:"
      echo "  sbatch train/train_with_drug_emb.sh custom /abs/path/to/embeddings.pt [experiment_name] [gpu_id]"
      exit 1
    fi
    DRUG_EMB_PATH="${CUSTOM_EMB_PATH}"
    DEFAULT_EXPERIMENT_NAME="Drugbank_CustomEmb"
    ;;
  *)
    echo "Unknown embedding choice: ${EMB_CHOICE}"
    echo "Usage:"
    echo "  sbatch train/train_with_drug_emb.sh gemma"
    echo "  sbatch train/train_with_drug_emb.sh llama"
    echo "  sbatch train/train_with_drug_emb.sh custom /abs/path/to/embeddings.pt [experiment_name] [gpu_id]"
    exit 1
    ;;
esac

if [[ -z "${EXPERIMENT_NAME}" ]]; then
  EXPERIMENT_NAME="${DEFAULT_EXPERIMENT_NAME}"
fi

if [[ ! -f "${DRUG_EMB_PATH}" ]]; then
  echo "Embedding file not found: ${DRUG_EMB_PATH}"
  exit 1
fi

echo "Embedding choice: ${EMB_CHOICE}"
echo "Embedding path: ${DRUG_EMB_PATH}"
echo "Experiment name: ${EXPERIMENT_NAME}"
echo "GPU: ${GPU_ID}"

nvidia-smi

cd "${PYTORCH_DIR}"
python train_with_drug_emb.py \
  -e "${EXPERIMENT_NAME}" \
  --dataset drugbank \
  --gpu 0 \
  --drug_embeddings_path "${DRUG_EMB_PATH}"
