# KnowDDI DrugBank Embedding Study (CPSC5830 Final Project)

This repository contains our course project built on top of `KnowDDI` for the DrugBank benchmark. The main goal of this extension is to test whether replacing the initial drug-node representations with external text-derived embeddings improves drug-drug interaction (DDI) prediction.

The project keeps the original KnowDDI pipeline and adds support for:

- external drug embeddings for the 1710 DrugBank drug nodes
- projection of those embeddings into the model hidden space
- text-to-embedding generation from drug metadata
- training scripts for baseline, Embedding-Gemma, and Llama-based experiments
- plotting scripts/notebooks for the report figures

## What We Changed

Relative to the original KnowDDI codebase, the main additions are:

- `pytorch/model/GraphSAGE_with_drug_emb.py`
  Graph encoder that injects external drug embeddings while keeping non-drug nodes trainable.
- `pytorch/model/Classifier_model_with_drug_emb.py`
  Wrapper classifier that uses the new graph encoder.
- `pytorch/train_with_drug_emb.py`
  Training entry point for experiments with external drug embeddings.
- `generate_emb.py`
  Builds external drug embeddings from drug text and writes a `.pt` file plus an audit CSV.
- `train/train_with_drug_emb.sh`
  Example SLURM launcher for embedding-based runs.
- `raw_data/Drugbank/generate_drug_node_csv.py`
  Utility script to export the mapping between DrugBank IDs and internal node IDs.

## Repository Layout

- `pytorch/`
  Core KnowDDI model and training code.
- `train/`
  Example SLURM scripts for baseline and embedding-based runs.
- `generate_emb.py`
  Embedding generation script.

This submission is code-only. Protected data files are not redistributed in this repository.

## Environment

The original KnowDDI repository targets older versions of PyTorch and DGL. Our runs were performed in a custom GPU environment on Linux.

Minimum packages used by the original codebase:

- Python 3.7+
- PyTorch
- DGL
- lmdb
- networkx
- scikit-learn
- tqdm

Extra packages needed for the embedding pipeline:

- transformers
- pandas
- sentencepiece

If you want to reproduce the embedding-generation step, install the Hugging Face dependencies in addition to the original `requirements.txt`.

## Data Availability

This project focuses only on the DrugBank benchmark. However, the `data/` and `raw_data/` directories are not included in this submission because the benchmark files are protected and cannot be redistributed.

To reproduce the experiments, users must obtain the original DrugBank / KnowDDI benchmark files from the official sources and place them under the expected directory structure before running training.

The external drug embedding file must align with the DrugBank node order defined by the original KnowDDI DrugBank mapping files.

## Running Experiments

### 1. Baseline KnowDDI

From the repository root:

```bash
cd pytorch
python train.py -e Drugbank --dataset drugbank --gpu 0
```

### 2. Generate External Drug Embeddings

Example for Embedding-Gemma:

```bash
python generate_emb.py \
  --model embedding-gemma \
  --output_pt data/drugbank/drug_embeddings_embedding_gemma.pt \
  --output_text_csv data/drugbank/drug_embeddings_embedding_gemma_text.csv
```

Example for Llama-3.1-8B-Instruct:

```bash
python generate_emb.py \
  --model llama-3.1-8b-instruct \
  --output_pt data/drugbank/drug_embeddings_llama3-1.pt \
  --output_text_csv data/drugbank/drug_embeddings_llama3-1_text.csv
```

### 3. Train with External Drug Embeddings

From `pytorch/`:

```bash
python train_with_drug_emb.py \
  -e Drugbank_GemmaEmb \
  --dataset drugbank \
  --gpu 0 \
  --drug_embeddings_path ../data/drugbank/drug_embeddings_embedding_gemma.pt
```

```bash
python train_with_drug_emb.py \
  -e Drugbank_LlamaEmb \
  --dataset drugbank \
  --gpu 0 \
  --drug_embeddings_path ../data/drugbank/drug_embeddings_llama3-1.pt
```

The `train/` directory also contains example SLURM scripts, but they may need path and cluster-specific edits before use.

## Reproducing Report Results

Our report compares three methods on DrugBank:

- baseline KnowDDI
- KnowDDI + Embedding-Gemma drug embeddings
- KnowDDI + Llama-3.1-8B drug embeddings

For fair comparison, the reported tables and figures use only the first 50 epochs of each run, and the selected checkpoint is the one with the best validation Macro-F1 within those 50 epochs.

To reproduce the report:

1. Prepare the protected DrugBank / KnowDDI data locally.
2. Generate the external embedding files if needed.
3. Run baseline and embedding-based training.
4. Parse the logs and regenerate the plots used in the report.

## Notes

- This repository is an adaptation of the original KnowDDI codebase for a class project.
- Some original scripts and paths were designed for a specific cluster environment, so small path changes may be needed on another machine.
- The original KnowDDI repository should be cited if this code is reused in future work.
