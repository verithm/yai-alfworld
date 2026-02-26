# ALFWorld LLM Agent Evaluation

Text-based household task environment using **ALFWorld (text-only mode)** with
**Llama-3.1-8B** served via **Ollama**.

---

## Hardware Requirements

Experiments are run on **VESSL** GPU workspaces.

| Component | Spec |
|-----------|------|
| GPU       | NVIDIA RTX 3090 (24 GB VRAM) |
| Model     | `llama3.1:8b` via Ollama (Q4_K_M, ~5 GB VRAM) |
| Execution | Native Python — no Docker on VESSL |

> GPU inference via Ollama on the RTX 3090 gives ~50–80 tokens/second.
> A 134-game full evaluation takes roughly 2–4 hours per agent.

---

## Project Structure

```
nlp_ws/
├── Dockerfile                    # ALFWorld + evaluation environment (local Docker)
├── docker-compose.yml            # Ollama + ALFWorld services (local Docker)
├── requirements.txt
├── configs/
│   └── alfworld_config.yaml      # ALFWorld env settings
├── scripts/
│   ├── run_baseline.py                     # Main evaluation script
│   ├── collect_oracle_trajectories.py      # Phase 2: extract text oracle data for SFT
│   └── agents/
│       ├── base_agent.py                   # Shared Ollama client + command matching
│       ├── zero_shot_agent.py              # Zero-shot prompting
│       ├── few_shot_agent.py               # Few-shot with 3 curated demonstrations
│       ├── react_agent.py                  # ReAct (Thought + Action loop)
│       ├── reflexion_agent.py              # Reflexion (ReAct + verbal self-correction)
│       └── hierarchical_agent.py # Hierarchical planning + Reflexion (SOTA)
├── data/                         # ALFWorld game files (auto-downloaded, ~800 MB)
└── results/                      # JSON results written here
```

---

## Quick Start (VESSL)

All experiments run natively on VESSL GPU workspaces — no Docker required.

### Step 1 — Open a VESSL workspace and install dependencies

```bash
pip install alfworld textworld rich
```

### Step 2 — Start Ollama and pull the model

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
sleep 10
ollama pull llama3.1:8b
```

### Step 3 — Download ALFWorld game data (first time only)

```bash
export ALFWORLD_DATA=~/data/alfworld
mkdir -p "$ALFWORLD_DATA"
alfworld-download
```

### Step 4 — Run the full OOD evaluation

```bash
cd ~/nlp_ws
python scripts/run_baseline.py \
  --agents zero_shot few_shot react reflexion hierarchical \
  --num-games 134 \
  --max-steps 50 \
  --split eval_out_of_distribution \
  --checkpoint results/checkpoint.json \
  --output    results/baseline_results.json
```

### Step 5 — Check results

```bash
python3 -c "
import json
d = json.load(open('results/baseline_results.json'))
for agent, info in d.get('agents', {}).items():
    print(f\"{agent:<28} SR={info['success_rate']:.1%}  avg_steps={info.get('avg_steps', 0):.1f}\")
"
```

---

## Configuration

All parameters are passed as CLI flags to `run_baseline.py`.

### Run only specific agents

```bash
python scripts/run_baseline.py \
  --agents react reflexion hierarchical \
  --num-games 134 --max-steps 50
```

### Run a quick smoke test (2 games per agent, ~10–15 min)

```bash
python scripts/run_baseline.py \
  --agents zero_shot few_shot react reflexion hierarchical \
  --num-games 2 --max-steps 20 \
  --output results/smoke_results.json
```

### Resume an interrupted run from a checkpoint

```bash
python scripts/run_baseline.py \
  --agents zero_shot few_shot react reflexion hierarchical \
  --num-games 134 \
  --checkpoint results/checkpoint.json \
  --output    results/baseline_results.json
```

### Run against the in-distribution split

```bash
python scripts/run_baseline.py \
  --agents zero_shot few_shot react reflexion hierarchical \
  --num-games 134 --split eval_in_distribution
```

---

## Prompting Strategies

| Agent | Description |
|-------|-------------|
| **Zero-shot** | System instruction only; model sees current observation + recent history |
| **Few-shot**  | 3 curated ALFWorld task demonstrations prepended to every prompt |
| **ReAct**     | Explicit `Thought: …` + `Action: …` scratchpad maintained across steps |
| **Reflexion** | ReAct actor + verbal self-critique after each failed trial; up to 3 retries per game |
| **Hierarchical** | Two-pass architecture: LLM planner generates subgoal sequence; executor follows each subgoal; Reflexion self-critique targets the planner on retry |

---

## ALFWorld Task Types

| ID | Type |
|----|------|
| 1  | Pick and place simple |
| 2  | Look at object in light |
| 3  | Pick, clean, then place |
| 4  | Pick, heat, then place |
| 5  | Pick, cool, then place |
| 6  | Pick two objects and place |

---

## Troubleshooting

**Ollama not responding**
```bash
# Check if the server is running
curl http://localhost:11434/api/tags
# If not, restart it
ollama serve &
```

**Model pull stalls or fails**
```bash
# Retry the pull directly
ollama pull llama3.1:8b
# Verify it's available
ollama list
```

**ALFWorld data not found**
```bash
# Ensure ALFWORLD_DATA points to the right directory
export ALFWORLD_DATA=~/data/alfworld
alfworld-download
```

**Evaluation interrupted mid-run**
Pass `--checkpoint results/checkpoint.json` — the run will resume from the last completed game.

---

## Project Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Baseline agent evaluation — all 5 agents on ALFWorld OOD split | 🔄 In progress |
| **Phase 2** | Supervised Fine-Tuning (SFT) — oracle collection + LoRA fine-tuning | ⏳ Planned |
| **Phase 3** | Evaluation with the SFT model — Baseline vs SFT comparison | ⏳ Planned |

---

## Experiment Plan

All experiments run on VESSL (`cluster: yonsei-ai-gpu`, `preset: gpu-1`,
image `quay.io/vessl-ai/torch:2.3.1-cuda12.1-r5`, storage org `YS-YAI`).

### VESSL Setup (one-time)

Upload scripts and ALFWorld data to VESSL volumes before running any experiment:

```bash
# Upload scripts (zip the scripts/ and configs/ directories)
vessl storage upload --volume alfworld-scripts scripts/ configs/

# Upload ALFWorld game data (~800 MB)
vessl storage upload --volume alfworld-data data/
```

Results are automatically written back to the `alfworld-results` volume at the end of each job.

---

### Phase 1 — Baseline Evaluation

#### E0 · Smoke Test
- **Purpose:** Verify GPU access, Ollama serving, and all agent imports before committing to a long run.
- **Scope:** 2 games × 5 agents (~10–15 min)
- **Command:**
  ```bash
  python scripts/run_baseline.py \
    --agents zero_shot few_shot react reflexion hierarchical \
    --num-games 2 --max-steps 20 \
    --split eval_out_of_distribution \
    --output results/e0_smoke_results.json
  ```

#### E1 · Full OOD Baseline
- **Purpose:** Establish the definitive pre-SFT baseline for all 5 agents across the complete OOD dataset.
- **Scope:** 134 games × 5 agents (full OOD split, ~4–6 h on RTX 3090). No specific target success rate — results are recorded as-is for comparison with Phase 3.
- **Command:**
  ```bash
  python scripts/run_baseline.py \
    --agents zero_shot few_shot react reflexion hierarchical \
    --num-games 134 --max-steps 50 \
    --split eval_out_of_distribution \
    --checkpoint results/e1_checkpoint.json \
    --output    results/e1_results.json
  ```

---

### Phase 2 — Supervised Fine-Tuning

ALFWorld provides a built-in `handcoded` oracle for every training game. However, the
on-disk `traj_data.json` files use ALFRED vision-format actions (e.g. `GotoLocation`,
`PickupObject`) — **not** the TextWorld text strings the LLM sees at runtime. The text
oracle must be collected by stepping through the environment with `expert_type: handcoded`.

#### Oracle Data Extraction

```bash
export ALFWORLD_DATA=~/data/alfworld
python scripts/collect_oracle_trajectories.py \
  --split train \
  --output results/oracle/trajectories.jsonl
```

- **Source:** ALFWorld `handcoded` oracle — accessed via the Python env API at runtime
- **Output:** `results/oracle/trajectories.jsonl` (saved to local workspace disk)
- **Format:** one JSONL record per (system\_prompt, conversation) training example

#### SFT Training
- **Framework:** Unsloth + HuggingFace PEFT (LoRA)
- **Input:** `results/oracle/trajectories.jsonl`
- **Base model:** `meta-llama/Llama-3.1-8B`
- **Target:** Fine-tuned adapter saved to `results/sft_adapter/`
- **Full guide:** [`docs/sft_pipeline_guide.md`](docs/sft_pipeline_guide.md) — proven stable versions, step-by-step setup, training script, and troubleshooting

---

### Phase 3 — Evaluation with SFT Model

Re-run the identical E1 evaluation with the SFT model:

```bash
python scripts/run_baseline.py \
  --agents zero_shot few_shot react reflexion hierarchical \
  --num-games 134 --max-steps 50 \
  --model llama3.1:8b-sft \
  --split eval_out_of_distribution \
  --checkpoint results/e2_checkpoint.json \
  --output    results/e2_sft_results.json
```

Final comparison: E1 (pre-SFT baseline) vs E2 (post-SFT) success rates per agent and per task type.
