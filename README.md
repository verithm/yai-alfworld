# Week 1 — ALFWorld Baseline Evaluation

Text-based household task environment using **ALFWorld (text-only mode)** with
**Llama-3.1-8B** served locally via **Ollama** (CPU inference, Q4_K_M ~5 GB RAM).

---

## Hardware Requirements

| Component | Minimum | This Machine |
|-----------|---------|--------------|
| CPU       | AVX2 support | AMD Ryzen 7 4800U (8c/16t, AVX2 ✓) |
| RAM       | 10 GB   | 14 GB |
| Storage   | 8 GB free | 393 GB NVMe |
| GPU       | Not required | Integrated AMD Radeon (CPU-only) |

> **No NVIDIA GPU required.** Inference runs on CPU via llama.cpp inside Ollama.
> Expect ~1–3 tokens/second; 10 games per agent takes ~30–90 minutes.

---

## Project Structure

```
nlp_ws/
├── Dockerfile                    # ALFWorld + evaluation environment
├── docker-compose.yml            # Ollama (LLM) + ALFWorld (eval) services
├── requirements.txt
├── entrypoint.sh                 # Startup: wait for Ollama, pull model, run eval
├── configs/
│   └── alfworld_config.yaml      # ALFWorld env settings
├── scripts/
│   ├── run_baseline.py           # Main evaluation script
│   └── agents/
│       ├── base_agent.py         # Shared Ollama client + command matching
│       ├── zero_shot_agent.py    # Zero-shot prompting
│       ├── few_shot_agent.py     # Few-shot with 3 curated demonstrations
│       └── react_agent.py        # ReAct (Thought + Action loop)
├── data/                         # ALFWorld game files (auto-downloaded, ~800 MB)
└── results/                      # JSON results written here
```

---

## Quick Start

### Step 1 — Build the image

```bash
cd ~/nlp_ws
docker compose build
```

### Step 2 — Run the full baseline evaluation

```bash
docker compose up
```

On **first run** this will:
1. Start Ollama (CPU mode)
2. Pull `llama3.1:8b` (~4.9 GB download — one time only, cached in `ollama_data` volume)
3. Download ALFWorld game files (~800 MB — one time only, cached in `./data/`)
4. Run Zero-shot → Few-shot → ReAct evaluation (10 games each)
5. Write `results/baseline_results.json`

Subsequent runs skip the downloads and start the evaluation immediately.

### Step 3 — Check results

```bash
cat results/baseline_results.json | python3 -m json.tool
```

---

## Configuration

All parameters can be overridden via environment variables or CLI flags.

### docker-compose.yml environment block

```yaml
environment:
  MODEL_NAME: "llama3.1:8b"   # any model pulled into Ollama
  NUM_GAMES:  "10"             # games per agent (increase for final eval)
  MAX_STEPS:  "50"             # max actions per episode
```

### Run only specific agents

```bash
docker compose run --rm alfworld \
    python scripts/run_baseline.py --agents zero_shot react
```

### Run against the in-distribution split

```bash
docker compose run --rm alfworld \
    python scripts/run_baseline.py --split eval_in_distribution
```

### Run with more games for a proper benchmark

```bash
docker compose run --rm alfworld \
    python scripts/run_baseline.py --num-games 50
```

---

## Prompting Strategies

| Strategy   | Description |
|------------|-------------|
| **Zero-shot** | System instruction only; model sees current observation + recent history |
| **Few-shot**  | 3 curated ALFWorld task demonstrations prepended to every prompt |
| **ReAct**     | Explicit `Thought: …` + `Action: …` output format; full scratchpad maintained across steps |

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

**Ollama never becomes healthy**
```bash
docker compose logs ollama
# Should see "Listening on [::]:11434"
```

**Model pull stalls or fails**
```bash
# Pull manually on the host, then restart
docker compose exec ollama ollama pull llama3.1:8b
```

**ALFWorld data download fails inside container**
```bash
# Download on the host and copy to ./data/
docker compose run --rm --entrypoint bash alfworld
$ alfworld-download
```

**Out-of-memory during evaluation**
Lower `NUM_GAMES` or reduce `memory` limits in `docker-compose.yml`.

---

## Week 2 Preview

- Oracle trajectory collection (10k episodes) from `train` split
- LoRA SFT on collected data using Unsloth / HuggingFace PEFT
- Training will require the A100 cluster (data center back online)

## Week 3 Preview

- Reflexion module: on-failure → self-critique → retry loop
- Final demo: compare Baseline vs SFT vs Reflexion success rates
