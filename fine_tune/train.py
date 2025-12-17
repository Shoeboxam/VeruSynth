import json
import random
from functools import partial
from typing import Dict, Any, List, Tuple
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from plotting import LossHistoryCallback
from verusynth.harness import DEFAULT_BASE_MODEL
from verusynth.llm import build_annotation_prompt
from extractors import collect_all_from_scratch


def make_hf_dataset(examples: List[Dict[str, Any]]) -> Dataset:
    rows = []
    for ex in examples:
        inp = ex["input"]
        prompt = build_annotation_prompt(
            func_src=inp["base_func_src"],  # from-scratch: base function
            sites=inp["sites"],
            current_annotations=inp.get("current_annotations") or {"annotations": []},
            latest_errors=inp.get("latest_errors", "") or "",
            error_summary=inp.get("error_summary", "") or "",
        )

        target = json.dumps(ex["output"], ensure_ascii=False, indent=2)
        full_text = prompt + "\n\n" + target
        rows.append({"text": full_text})
    return Dataset.from_list(rows)


def tokenize_with_labels(example, tokenizer, cutoff_len: int = 4096):
    """
    Turn example["text"] into input_ids + labels where:
      - labels corresponding to the prompt part are -100 (ignored)
      - labels for the target JSON are actual token IDs.
    We detect the boundary using the last occurrence of "\n\n{" (start of JSON).
    """
    text = example["text"]

    # crude but usually robust: last JSON block starts with '\n\n{'
    split_idx = text.rfind("\n\n{")
    if split_idx == -1:
        split_idx = len(text)

    prompt_str = text[:split_idx]
    target_str = text[split_idx:]

    full = prompt_str + target_str

    # Tokenize full text
    tokens = tokenizer(
        full,
        truncation=True,
        max_length=cutoff_len,
        padding="max_length",
        return_tensors=None,
    )
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    # Tokenize prompt alone to find boundary in token space
    prompt_tokens = tokenizer(
        prompt_str,
        truncation=True,
        max_length=cutoff_len,
        padding="max_length",
        return_tensors=None,
    )["input_ids"]

    prompt_len = sum(1 for t in prompt_tokens if t != tokenizer.pad_token_id)

    labels = [-100] * len(input_ids)
    for i in range(prompt_len, len(input_ids)):
        if attention_mask[i] == 1:
            labels[i] = input_ids[i]

    tokens["labels"] = labels
    return tokens

from typing import Optional

def run_fine_tune(
    train_examples: List[Dict[str, Any]],
    output_dir: str,
    base_model: str,
    val_examples: Optional[List[Dict[str, Any]]] = None,
):
    # 1) Build HF dataset with "text" field
    train_ds = make_hf_dataset(train_examples)
    val_ds = make_hf_dataset(val_examples) if val_examples else None

    # 2) Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    # keep tokenizer + model config aligned; don't override pad_token_id manually

    # 3) Tokenize with labels
    train_tokenized = train_ds.map(
        partial(tokenize_with_labels, tokenizer=tokenizer, cutoff_len=4096),
        remove_columns=["text"],
        batched=False,
    )

    val_tokenized = None
    if val_ds is not None:
        val_tokenized = val_ds.map(
            partial(tokenize_with_labels, tokenizer=tokenizer, cutoff_len=4096),
            remove_columns=["text"],
            batched=False,
        )

    # 4) Load base model with 4-bit quantization via BitsAndBytesConfig
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # 5) Training args
    has_val = val_tokenized is not None and len(val_tokenized) > 0
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to=[],
        save_strategy="epoch" if has_val else "steps",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    history_cb = LossHistoryCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        processing_class=tokenizer,
        callbacks=[history_cb],
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def split_autoverus_val(
    examples: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split dataset into (train, val), where:
      - val set is drawn ONLY from AutoVerus examples
      - fraction of AutoVerus examples sent to val is `val_frac`

    Non-AutoVerus examples are always in train.
    """
    rng = random.Random(42)
    validation_fraction = 0.15

    auto = [ex for ex in examples if ex.get("meta", {}).get("data_source") == "verus-proof-synthesis"]
    non_auto = [ex for ex in examples if ex.get("meta", {}).get("data_source") != "verus-proof-synthesis"]

    if not auto:
        # Degenerate case: no autoverus, all train, empty val
        return examples, []

    n_val = max(1, int(round(len(auto) * validation_fraction))) if len(auto) > 1 else 1
    auto_indices = list(range(len(auto)))
    rng.shuffle(auto_indices)

    val_indices = set(auto_indices[:n_val])
    auto_val = [auto[i] for i in val_indices]
    auto_train = [auto[i] for i in auto_indices[n_val:]]

    train = non_auto + auto_train
    val = auto_val

    return train, val

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-model",
        type=str,
        help='HuggingFace base model, e.g. "Qwen/Qwen3-8B-Base"',
    )

    args = parser.parse_args()
    base_model: str = args.base_model or DEFAULT_BASE_MODEL

    # ------------------------------------------------------------------
    # Derive <org>/<model> from "org/model"
    # ------------------------------------------------------------------
    parts = base_model.split("/")
    if len(parts) != 2:
        raise ValueError(
            f"Expected base model like 'Qwen/Qwen3-8B-Base', got {base_model}"
        )
    org, model = parts

    # Base directory where LoRA adapters for this model live:
    model_root = Path("src/verusynth/lora") / org / model
    model_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Auto-pick next training run number
    # ------------------------------------------------------------------
    existing = []
    for child in model_root.iterdir():
        if child.is_dir():
            try:
                n = int(child.name)
                existing.append(n)
            except ValueError:
                pass

    next_train_number = max(existing) + 1 if existing else 1

    output_dir = model_root / str(next_train_number)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[verusynth] Starting LoRA fine-tune run #{next_train_number}")
    print(f"[verusynth] Output directory: {output_dir}")

    # ------------------------------------------------------------------
    # Load dataset (always from ./repositories)
    # ------------------------------------------------------------------
    from extractors import collect_all_from_scratch
    repositories_root = Path("repositories")     # ← fixed; no CLI argument
    examples = collect_all_from_scratch(repositories_root)

    from train import split_autoverus_val, run_fine_tune

    # Split: val = 15% autoverus only
    train_examples, val_examples = split_autoverus_val(examples)

    # ------------------------------------------------------------------
    # Run the fine-tune
    # ------------------------------------------------------------------
    run_fine_tune(
        train_examples=train_examples,
        val_examples=val_examples,
        output_dir=str(output_dir),
        base_model=base_model,
    )

    print(f"[verusynth] Finished training run #{next_train_number}")
