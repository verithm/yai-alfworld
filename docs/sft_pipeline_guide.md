# Supervised Fine-Tuning (SFT) Pipeline Guide

Step-by-step instructions for fine-tuning Llama-3.1-8B on ALFWorld oracle trajectories
using QLoRA (4-bit quantized LoRA). Covers oracle data collection, environment setup,
training, and evaluation.

---

## Proven Stable Stack

The versions below are known-good as of early 2026 for Llama-3.1-8B QLoRA training.

| Component      | Version              | Notes                                          |
|----------------|----------------------|------------------------------------------------|
| Python         | **3.11**             | 3.12/3.13 have incomplete wheel coverage       |
| CUDA Toolkit   | **12.1**             | Widest pre-built wheel support                 |
| PyTorch        | **2.4.1**            | Official cu121 wheels; all LoRA ops work       |
| transformers   | **4.45.0**           | Llama-3.1 tokenizer fixes included             |
| peft           | **0.11.0**           | Required for Unsloth integration               |
| trl            | **0.24.0**           | `SFTConfig`/`SFTTrainer` API stable            |
| bitsandbytes   | **0.44.0**           | 4-bit QLoRA with CUDA 12.1                     |
| accelerate     | **1.0.0**            | Single-GPU or multi-GPU training               |
| datasets       | **2.20.0**           | JSONL loading and formatting                   |
| unsloth        | **git main (2025+)** | Llama-3.1 patching; requires peft ≥ 0.11       |
| flash-attn     | **2.6.3+**           | Optional; speeds up attention; build after torch |

> **If your VESSL image ships CUDA 13.x or Python 3.13**, downgrade the image.
> The recommended VESSL image tag is:
> `quay.io/vessl-ai/torch:2.4.1-cuda12.1-cudnn9-r3` (or equivalent PyTorch 2.4 + CUDA 12.1)

---

## Phase 2 Overview

```
Phase 2 consists of two steps:

  Step A — Oracle Data Collection
    Run collect_oracle_trajectories.py against the ALFWorld *training* split.
    The ALFWorld handcoded oracle generates expert action sequences at runtime.
    Output: results/oracle/trajectories.jsonl  (~2 500–3 000 games)

  Step B — SFT Training
    Fine-tune Llama-3.1-8B with QLoRA on the collected JSONL.
    Output: results/sft_adapter/  (saved LoRA adapter)
```

---

## Step A — Oracle Data Collection

### Why not use `traj_data.json` directly?

The on-disk `traj_data.json` files use ALFRED vision-format actions (e.g. `GotoLocation`,
`PickupObject`). These are **not** the TextWorld text commands the LLM sees at runtime
(e.g. `go to countertop 1`, `pick up mug 1`). The text oracle must be collected by
stepping through the environment with `expert_type: handcoded` via the Python API.

### Run the collection script

```bash
export ALFWORLD_DATA=~/data/alfworld

python scripts/collect_oracle_trajectories.py \
  --split train \
  --output results/oracle/trajectories.jsonl \
  --max-steps 50
```

**Expected output**: ~2 500–3 000 successful JSONL records (games where oracle score > 0).
Each record is one multi-turn chat conversation:

```json
{
  "messages": [
    {"role": "system",    "content": "You are a household robot..."},
    {"role": "user",      "content": "Observation: ...\n\nAdmissible actions:\n  - look\n  - ..."},
    {"role": "assistant", "content": "go to countertop 1"},
    {"role": "user",      "content": "Observation: You arrive at countertop 1...\n\nAdmissible actions:\n  - ..."},
    {"role": "assistant", "content": "pick up mug 1"},
    ...
  ],
  "gamefile": "/data/alfworld/json_2.1.1/train/.../game.tw-pddl",
  "steps": 12
}
```

### Verify the output

```bash
python3 -c "
import json
records = [json.loads(l) for l in open('results/oracle/trajectories.jsonl')]
print(f'Total records: {len(records)}')
turns = [(len(r['messages']) - 1) // 2 for r in records]
print(f'Avg turns per record: {sum(turns)/len(turns):.1f}')
print('Sample keys:', list(records[0].keys()))
"
```

---

## Step B — SFT Training

### B1 — Environment Setup

```bash
# Create a clean Python 3.11 virtual environment
python3.11 -m venv ~/sft_env
source ~/sft_env/bin/activate

# Install PyTorch + CUDA 12.1 wheels (must be first)
pip install \
  torch==2.4.1 \
  torchvision==0.19.1 \
  torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121

# Install core training stack (pinned versions)
pip install \
  transformers==4.45.0 \
  peft==0.11.0 \
  trl==0.24.0 \
  bitsandbytes==0.44.0 \
  accelerate==1.0.0 \
  datasets==2.20.0

# Install Unsloth from git (required for Llama-3.1 patching)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Optional: flash-attn for faster training (build after torch)
pip install flash-attn>=2.6.3
```

**Verify installation:**

```bash
python3 -c "
import torch, transformers, peft, trl, bitsandbytes, unsloth
print(f'torch:          {torch.__version__}')
print(f'CUDA:           {torch.version.cuda}')
print(f'transformers:   {transformers.__version__}')
print(f'peft:           {peft.__version__}')
print(f'trl:            {trl.__version__}')
print(f'bitsandbytes:   {bitsandbytes.__version__}')
print(f'GPU available:  {torch.cuda.is_available()}')
print(f'GPU name:       {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')
"
```

### B2 — Training Script

The training script is at `scripts/sft_train.py` in the repository.
Key implementation notes (for reference):

- Unsloth must be imported before trl/transformers/peft to apply its patches.
- `SFTConfig` uses `max_length` (not `max_seq_length`) in trl 0.24.0.
- `SFTTrainer` uses `processing_class=tokenizer` (not `tokenizer=`) in trl 0.24.0.
- Unsloth's `FastLanguageModel.from_pretrained` still uses `max_seq_length` (its own parameter).
- `batch=1, grad_accum=16` → effective batch size = 16 (A100 80 GB OOMs at batch > 1 with seq_len = 8192).

### B3 — Run Training

```bash
source ~/sft_env/bin/activate

python scripts/sft_train.py \
  --data       results/oracle/trajectories.jsonl \
  --output     results/sft_adapter \
  --base       unsloth/Meta-Llama-3.1-8B-Instruct \
  --epochs     3 \
  --batch      1 \
  --grad-accum 16
```

**Expected training time** on a single A100 (80 GB): ~2–4 hours for 3 epochs over ~2 500 records.
`batch=1 × grad_accum=16` → effective batch size = 16.

### B4 — Register the Adapter with Ollama

After training, export the merged model for Ollama serving:

```bash
# Merge adapter into base weights
python3 - <<'EOF'
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base  = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", torch_dtype="auto")
model = PeftModel.from_pretrained(base, "results/sft_adapter")
merged = model.merge_and_unload()

merged.save_pretrained("results/sft_merged")
AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct").save_pretrained("results/sft_merged")
print("Merged model saved to results/sft_merged/")
EOF

# Convert to GGUF and register with Ollama
# Requires llama.cpp — clone it if not already present
git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp
pip install -r /tmp/llama.cpp/requirements.txt

python3 /tmp/llama.cpp/convert_hf_to_gguf.py results/sft_merged/ \
  --outfile results/llama3.1-8b-sft.gguf --outtype q8_0
ollama create llama3.1:8b-sft -f - <<'MODELFILE'
FROM ./results/llama3.1-8b-sft.gguf
MODELFILE
```

> **Note — quantization**: `--outtype q4_k_m` is no longer supported in recent llama.cpp builds.
> We use `q8_0` (8-bit quantization) for both the SFT model and the control baseline.
>
> **Baseline model registration**: The unmodified `meta-llama/Llama-3.1-8B-Instruct` weights
> were converted and quantized with the same `q8_0` method and registered in Ollama as
> `llama3.1:8b-base`. Use this model name when running Phase 1 / control evaluations.

### B5 — Run Phase 3 Evaluation with SFT Model

```bash
python scripts/run_baseline.py \
  --agents zero_shot few_shot react reflexion hierarchical \
  --num-games 134 \
  --max-steps 50 \
  --model llama3.1:8b-sft \
  --split eval_out_of_distribution \
  --checkpoint results/e2_checkpoint.json \
  --output    results/e2_sft_results.json
```

---

## Troubleshooting

### `ImportError: bitsandbytes not compiled for CUDA`

```bash
# Force reinstall with explicit CUDA version
pip install bitsandbytes==0.44.0 --force-reinstall
# Verify: should print CUDA version
python3 -c "import bitsandbytes; print(bitsandbytes.__version__)"
```

### `flash-attn` build fails

```bash
pip install flash-attn --no-build-isolation
# If that also fails, skip flash-attn — it is optional
```

### `CUDA out of memory` during training

Reduce batch size and increase gradient accumulation to compensate:

```bash
python scripts/sft_train.py --batch 2   # halve batch
# Gradient accumulation is 4 by default → effective batch = 8 either way
```

### Unsloth `ValueError: ... peft version`

```bash
pip install --upgrade peft
# Must be ≥ 0.11.0 for Unsloth to patch correctly
```

### HuggingFace gated model access

Llama-3.1-8B requires accepting Meta's license:

1. Visit https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct and accept the license.
2. Authenticate: `huggingface-cli login`
3. Re-run the training script.

---

## Compatibility Matrix

| VESSL Image                              | Python | CUDA | Recommendation          |
|------------------------------------------|--------|------|-------------------------|
| `torch:2.4.1-cuda12.1-cudnn9-r3`        | 3.11   | 12.1 | ✅ Use this              |
| `torch:2.3.1-cuda12.1-r5`               | 3.10   | 12.1 | ✅ Acceptable (downgrade peft/trl to match) |
| `torch:2.5.x-cuda12.4`                  | 3.11   | 12.4 | ⚠️  Third-party wheels lagging |
| `torch:2.x-cuda13.x` or Python 3.13     | —      | 13.x | ❌ Avoid — incomplete wheel support |
