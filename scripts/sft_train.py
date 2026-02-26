"""
QLoRA fine-tuning of Llama-3.1-8B on ALFWorld oracle trajectories.

Usage:
  python scripts/sft_train.py \
    --data   results/oracle/trajectories.jsonl \
    --output results/sft_adapter \
    --epochs 3
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

try:
    from unsloth import FastLanguageModel
    USE_UNSLOTH = True
except ImportError:
    from transformers import AutoModelForCausalLM
    USE_UNSLOTH = False
    print("Unsloth not available — falling back to pure PEFT+TRL")


# ── CLI ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--data",        default="results/oracle/trajectories.jsonl")
parser.add_argument("--output",      default="results/sft_adapter")
parser.add_argument("--base",        default="meta-llama/Llama-3.1-8B")
parser.add_argument("--epochs",      type=int,   default=3)
parser.add_argument("--batch",       type=int,   default=4)
parser.add_argument("--lr",          type=float, default=2e-4)
parser.add_argument("--max-seq-len", type=int,   default=8192)
args = parser.parse_args()


# ── Load & format data ───────────────────────────────────────────────────────

def load_jsonl(path: str):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def format_record(record: dict, tokenizer) -> str:
    """Convert a chat record to the model's chat template format.

    Data schema (from collect_oracle_trajectories.py):
      {"messages": [{"role": "system", ...}, {"role": "user", ...}, ...],
       "gamefile": "...", "steps": N}
    The system prompt is already embedded as messages[0].
    """
    return tokenizer.apply_chat_template(
        record["messages"], tokenize=False, add_generation_prompt=False
    )

raw = load_jsonl(args.data)
print(f"Loaded {len(raw)} training records from {args.data}")


# ── Load model ───────────────────────────────────────────────────────────────

if USE_UNSLOTH:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base,
        max_seq_length=args.max_seq_len,
        dtype=None,           # auto-detect (bfloat16 on Ampere+)
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
else:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

model.print_trainable_parameters()


# ── Prepare dataset ──────────────────────────────────────────────────────────

formatted = [format_record(r, tokenizer) for r in raw]
dataset = Dataset.from_dict({"text": formatted})
dataset = dataset.train_test_split(test_size=0.05, seed=42)
print(f"Train: {len(dataset['train'])}  Eval: {len(dataset['test'])}")


# ── Training ─────────────────────────────────────────────────────────────────

output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)

training_args = SFTConfig(
    output_dir=str(output_dir),
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch,
    per_device_eval_batch_size=args.batch,
    gradient_accumulation_steps=4,
    learning_rate=args.lr,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    max_seq_length=args.max_seq_len,
    dataset_text_field="text",
    packing=False,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=training_args,
)

trainer.train()

# Save adapter
model.save_pretrained(str(output_dir))
tokenizer.save_pretrained(str(output_dir))
print(f"\nAdapter saved to: {output_dir}")
