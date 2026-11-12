#!/usr/bin/env python3
"""
Build external drug embeddings for KnowDDI from drug text metadata.

Features:
- Reads drug metadata from `KnowDDI/data/drugbank/full_data.csv`
- Uses `description` if present, otherwise falls back to drug `name`
- Supports two model presets:
  - `embedding-gemma`
  - `llama-3.1-8b-instruct`
- Saves a `.pt` file compatible with `pytorch/train_with_drug_emb.py`
- Saves a companion CSV that records the exact text used for each drug
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm


DEFAULT_MODEL_MAP = {
    "embedding-gemma": "google/embeddinggemma-300m",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
}


def parse_args() -> argparse.Namespace:
    repo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate KnowDDI drug embeddings from text.")
    parser.add_argument("--model", required=True, choices=sorted(DEFAULT_MODEL_MAP.keys()))
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=repo_dir / "data" / "drugbank" / "full_data.csv",
    )
    parser.add_argument(
        "--mapping_csv",
        type=Path,
        default=repo_dir / "raw_data" / "Drugbank" / "drug_node_mapping.csv",
    )
    parser.add_argument("--output_pt", type=Path, required=True)
    parser.add_argument("--output_text_csv", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--pooling", choices=["mean", "last_token"], default="mean")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--id_column", default=None)
    parser.add_argument("--name_column", default=None)
    parser.add_argument("--description_column", default=None)
    parser.add_argument(
        "--missing_text_strategy",
        choices=["error", "drugbank_id"],
        default="drugbank_id",
        help=(
            "What to do when a drug is missing from full_data.csv or has no usable "
            "description/name. 'drugbank_id' uses the DrugBank ID string itself as text."
        ),
    )
    return parser.parse_args()


def normalize_col(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def pick_column(fieldnames: Sequence[str], explicit: Optional[str], candidates: Sequence[str], label: str) -> str:
    if explicit is not None:
        if explicit not in fieldnames:
            raise ValueError(f"{label} column '{explicit}' not found. Available columns: {list(fieldnames)}")
        return explicit

    normalized_to_original = {normalize_col(name): name for name in fieldnames}
    for candidate in candidates:
        if candidate in normalized_to_original:
            return normalized_to_original[candidate]

    raise ValueError(
        f"Could not infer {label} column from {list(fieldnames)}. "
        f"Please provide it explicitly."
    )


def load_mapping(mapping_csv: Path) -> List[Tuple[int, str]]:
    rows: List[Tuple[int, str]] = []
    with mapping_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "node_id" not in reader.fieldnames or "drugbank_id" not in reader.fieldnames:
            raise ValueError("Mapping CSV must contain columns: node_id, drugbank_id")
        for row in reader:
            rows.append((int(row["node_id"]), row["drugbank_id"].strip()))
    rows.sort(key=lambda x: x[0])
    return rows


def load_drug_texts(
    input_csv: Path,
    id_column: Optional[str],
    name_column: Optional[str],
    description_column: Optional[str],
) -> Dict[str, Dict[str, str]]:
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header row.")

        fieldnames = reader.fieldnames
        id_col = pick_column(
            fieldnames,
            id_column,
            ["drugbank_id", "drugbankid", "drug_id", "drugbank", "id"],
            "id",
        )
        name_col = pick_column(
            fieldnames,
            name_column,
            ["name", "drug_name", "generic_name", "drug"],
            "name",
        )
        desc_col = pick_column(
            fieldnames,
            description_column,
            ["description", "drug_description", "summary", "desc"],
            "description",
        )

        records: Dict[str, Dict[str, str]] = {}
        for row in reader:
            drugbank_id = (row.get(id_col) or "").strip()
            if not drugbank_id:
                continue

            name = (row.get(name_col) or "").strip()
            description = (row.get(desc_col) or "").strip()

            if drugbank_id not in records:
                records[drugbank_id] = {
                    "drugbank_id": drugbank_id,
                    "name": name,
                    "description": description,
                }
            else:
                if not records[drugbank_id]["name"] and name:
                    records[drugbank_id]["name"] = name
                if not records[drugbank_id]["description"] and description:
                    records[drugbank_id]["description"] = description

    return records


def choose_text(record: Dict[str, str]) -> Tuple[str, str]:
    description = record.get("description", "").strip()
    name = record.get("name", "").strip()
    if description:
        return description, "description"
    if name:
        return name, "name"
    raise ValueError(f"No usable text found for {record.get('drugbank_id', '<unknown>')}")


def resolve_device(device_arg: Optional[str]) -> str:
    if device_arg:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(dtype_arg: str, device: str):
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if dtype_arg == "float32":
        return torch.float32
    if device.startswith("cuda"):
        return torch.bfloat16
    return torch.float32


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1)
    return summed / counts


def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.sum(dim=1) - 1
    lengths = lengths.clamp_min(0)
    batch_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
    return last_hidden_state[batch_indices, lengths]


def load_model_and_tokenizer(model_name_or_path: str, trust_remote_code: bool, device: str, dtype):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        device_map="auto" if device.startswith("cuda") else None,
    )
    if not device.startswith("cuda"):
        model = model.to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def encode_texts(
    texts: Sequence[str],
    tokenizer,
    model,
    max_length: int,
    batch_size: int,
    device: str,
    pooling: str,
) -> torch.Tensor:
    all_embeddings: List[torch.Tensor] = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding", unit="batch"):
        batch_texts = list(texts[start:start + batch_size])
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        outputs = model(**encoded, return_dict=True)
        hidden = outputs.last_hidden_state

        if pooling == "mean":
            batch_emb = mean_pool(hidden, encoded["attention_mask"])
        else:
            batch_emb = last_token_pool(hidden, encoded["attention_mask"])

        batch_emb = F.normalize(batch_emb.float(), p=2, dim=1)
        all_embeddings.append(batch_emb.cpu())

    return torch.cat(all_embeddings, dim=0)


def write_text_audit_csv(output_csv: Path, rows: List[Dict[str, object]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "node_id",
                "drugbank_id",
                "text_source",
                "text_used",
                "drug_name",
                "description",
                "record_found",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    model_name_or_path = args.model_name_or_path or DEFAULT_MODEL_MAP[args.model]
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    output_text_csv = args.output_text_csv
    if output_text_csv is None:
        output_text_csv = args.output_pt.with_name(args.output_pt.stem + "_texts.csv")

    mapping_rows = load_mapping(args.mapping_csv)
    records = load_drug_texts(
        args.input_csv,
        args.id_column,
        args.name_column,
        args.description_column,
    )

    ordered_node_indices: List[int] = []
    ordered_drugbank_ids: List[str] = []
    texts: List[str] = []
    audit_rows: List[Dict[str, object]] = []
    missing_ids: List[str] = []

    for node_id, drugbank_id in mapping_rows:
        record = records.get(drugbank_id)
        record_found = record is not None

        if record is None:
            if args.missing_text_strategy == "error":
                missing_ids.append(drugbank_id)
                continue
            record = {
                "drugbank_id": drugbank_id,
                "name": "",
                "description": "",
            }
            text_used, text_source = drugbank_id, "drugbank_id"
        else:
            try:
                text_used, text_source = choose_text(record)
            except ValueError:
                if args.missing_text_strategy == "error":
                    missing_ids.append(drugbank_id)
                    continue
                text_used, text_source = drugbank_id, "drugbank_id"

        ordered_node_indices.append(node_id)
        ordered_drugbank_ids.append(drugbank_id)
        texts.append(text_used)
        audit_rows.append(
            {
                "node_id": node_id,
                "drugbank_id": drugbank_id,
                "text_source": text_source,
                "text_used": text_used,
                "drug_name": record.get("name", ""),
                "description": record.get("description", ""),
                "record_found": record_found,
            }
        )

    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        raise ValueError(
            f"Missing usable text for {len(missing_ids)} drug IDs. "
            f"First few: {preview}"
        )

    tokenizer, model = load_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        device=device,
        dtype=dtype,
    )

    embeddings = encode_texts(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
        pooling=args.pooling,
    )

    output = {
        "drug_embeddings": embeddings,
        "ordered_node_indices": ordered_node_indices,
        "ordered_drugbank_ids": ordered_drugbank_ids,
        "num_drug_nodes": len(ordered_node_indices),
        "embedding_dim": embeddings.shape[1],
        "source_model": model_name_or_path,
        "input_csv": str(args.input_csv.resolve()),
        "mapping_csv": str(args.mapping_csv.resolve()),
        "pooling": args.pooling,
        "max_length": args.max_length,
        "text_audit_csv": str(output_text_csv.resolve()),
    }

    args.output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output_pt)
    write_text_audit_csv(output_text_csv, audit_rows)

    print(f"Saved {len(ordered_node_indices)} embeddings to {args.output_pt}")
    print(f"Embedding shape: {tuple(embeddings.shape)}")
    print(f"Saved text audit CSV to {output_text_csv}")
    print(f"Model: {model_name_or_path}")


if __name__ == "__main__":
    main()