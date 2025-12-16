# **VeruSynth: Automatic Annotation Synthesis for Verus**

VeruSynth is a **verifier-in-the-loop annotation assistant** for
**[Verus](https://verus-lang.github.io/)**, a verifier for Rust.

It automatically synthesizes:

* **loop invariants**
* **decreases clauses**
* **assertions**

using a **large language model (LLM)** and integrates tightly with Verus to refine them iteratively until verification succeeds.

VeruSynth can be **installed as a Python package**, works directly on Rust source files, and requires no modifications to the Verus toolchain.

## Installation

### Requirements

* Python 3.10+
* Verus installed and available on `$PATH`
* CUDA GPU recommended for running large models

Clone, then:
```bash
cd verusynth
pip install -e .
```

This installs the `verusynth` package and the CLI tool `verusynth`.

---

## Quick Start

Run on a function:

```bash
verusynth src/my_module.rs my_function
```

Be sure to watch as the model writes annotations!

VeruSynth will:

1. Extract the function.
2. Identify annotation sites.
3. Load a base LLM (and optionally a fine-tuned LoRA adapter).
4. Iteratively propose annotations and test them with Verus.
5. Update the Rust file if a verifying set of annotations is found.

If you have a function that already verifies, and has annotations, 
then you can ignore existing annotations with:

```bash
verusynth src/my_module.rs my_function --ignore-existing
```

It stores an adjacent backup file with extension `.bak` first, 
and restores when done if AutoVerus fails to verify.

### Choose a model

The package comes with a LoRa for `Qwen/Qwen3-4B-Thinking-23507`.

VeruSynth automatically searches:

```
src/verusynth/lora/<org>/<model>/<run_number>/
```

and selects the latest run if no adapter is specified.

You can specify the base model:
```
verusynth src/my.rs f --base-model <org>/<model>
```

VeruSynth automatically chooses the highest-numbered adapter, 
but you can specify the adapter explicitly:
```
verusynth src/my.rs f --base-model <org>/<model> --adapter-path <path>
```

## Training and Fine-Tuning

VeruSynth includes tools for dataset generation and LoRA fine-tuning.

To set up the tooling for fine-tuning, run the following 
(you may want to tailor the script first):
```bash
./setup.sh
```

The script will attempt to install Verus, clone the repositories examples are mined from,
and set up a virtual environment with the dependencies necessary for training.

You can then train with:
```
python -m fine_tune.train --base-model Qwen/Qwen3-7B
```

This:

* Scans repositories under `repositories/`
* Extracts all verified Verus functions

Note that the dataset is prepared dynamically from the repository sources,
so you can `git pull` or `checkout` in these repositories to change the underlying dataset.

LoRA weights are automatically saved under:

```
src/verusynth/lora/<org>/<model>/<run_number>/
```

Evaluate the fine-tuned model with:

```
python fine_tune/eval_verusynth.py \
  --base-model Qwen/Qwen3-7B \
  --finetuned-adapter 3
```

Produces a verification-success score across the held-out test set.
