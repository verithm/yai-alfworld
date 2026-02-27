"""
QLoRA fine-tuning of Llama-3.1-8B-Instruct on ALFWorld oracle trajectories.

Usage:
  python scripts/sft_train.py \
    --data   results/oracle/trajectories.jsonl \
    --output results/sft_adapter \
    --epochs 3 \
    --batch  1 \
    --grad-accum 16
"""

# ── Unsloth MUST be imported first to apply its patches ──────────────────────
try:
    from unsloth import FastLanguageModel
    USE_UNSLOTH = True
except ImportError:
    from transformers import AutoModelForCausalLM
    USE_UNSLOTH = False
    print("Unsloth not available — falling back to pure PEFT+TRL")

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


# ── CLI ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--data",        default="results/oracle/trajectories.jsonl")
parser.add_argument("--output",      default="results/sft_adapter")
parser.add_argument("--base",        default="unsloth/Meta-Llama-3.1-8B-Instruct")
parser.add_argument("--epochs",      type=int,   default=3)
parser.add_argument("--batch",       type=int,   default=1)
parser.add_argument("--grad-accum",  type=int,   default=16)   # effective batch = batch × grad_accum
parser.add_argument("--lr",          type=float, default=2e-4)
parser.add_argument("--max-seq-len", type=int,   default=8192)
args = parser.parse_args()

print(f"Effective batch size: {args.batch * args.grad_accum} "
      f"({args.batch} per device × {args.grad_accum} grad accum steps)")


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
        max_seq_length=args.max_seq_len,   # Unsloth API still uses max_seq_length
        dtype=None,                         # auto-detect (bfloat16 on Ampere+)
        load_in_4bit=True,
    )
    # Instruct tokenizer already has eos/pad configured; set pad = eos to be explicit
    tokenizer.pad_token = tokenizer.eos_token
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
    gradient_accumulation_steps=args.grad_accum,
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
    max_length=args.max_seq_len,        # trl 0.24.0: was max_seq_length in older versions
    dataset_text_field="text",
    packing=False,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,         # trl 0.24.0: was 'tokenizer' in older versions
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=training_args,
)

trainer.train()

# Save adapter
model.save_pretrained(str(output_dir))
tokenizer.save_pretrained(str(output_dir))
print(f"\nAdapter saved to: {output_dir}")
