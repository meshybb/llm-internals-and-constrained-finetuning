#!/usr/bin/env python
# coding: utf-8

import json
import torch
import argparse
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from peft import LoraConfig, PeftModel

# ─────────────────────────────────────────────────────────
# argparse — קבלת היפר-פרמטרים מבחוץ
# ─────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()

parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--lora_r", type=int, default=32)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--output_dir", type=str, default="./qwen_lora_run")

args = parser.parse_args()

print("\n===== Training Configuration =====")
print(f"Epochs: {args.epochs}")
print(f"Learning Rate: {args.lr}")
print(f"LoRA r: {args.lora_r}")
print(f"Batch Size: {args.batch_size}")
print(f"Output Dir: {args.output_dir}")
print("=================================\n")

# ─────────────────────────────────────────────────────────
# 1. טעינת מודל
# ─────────────────────────────────────────────────────────

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

# ─────────────────────────────────────────────────────────
# 2. טעינת דאטה + חלוקה אוטומטית 80/10/10
# ─────────────────────────────────────────────────────────

with open("train_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

train_data, temp_data = train_test_split(data, test_size=0.20, random_state=42)
val_data, check_data = train_test_split(temp_data, test_size=0.50, random_state=42)

def convert_messages_to_text(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }

train_ds = Dataset.from_list(train_data).map(convert_messages_to_text)
val_ds   = Dataset.from_list(val_data).map(convert_messages_to_text)

# ─────────────────────────────────────────────────────────
# 3. הגדרת LoRA ואימון
# ─────────────────────────────────────────────────────────

peft_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_r * 2,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    learning_rate=args.lr,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    fp16=True,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
)

trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    peft_config=peft_config,
    args=training_args,
)

trainer.train()
trainer.save_model(args.output_dir)

print("\nTraining complete.")
print(f"Adapter saved to: {args.output_dir}")

# ─────────────────────────────────────────────────────────
# 4. Evaluation אוטומטי על check_data
# ─────────────────────────────────────────────────────────

print("\n===== Running Evaluation on Check Set =====\n")

ft_model = PeftModel.from_pretrained(base_model, args.output_dir)
ft_model.eval()

SYSTEM_PROMPT = "You are a helpful assistant. Always respond in Hebrew only."

def extract_user_prompt(messages):
    for m in messages:
        if m["role"] == "user":
            return m["content"]
    raise ValueError("No user message found")

def generate(model, messages, max_new_tokens=200):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

# ─────────────────────────────────────────────────────────
# 4A. Evaluation ללא system prompt
# ─────────────────────────────────────────────────────────

eval_no_prompt = []

for ex in check_data:
    prompt = extract_user_prompt(ex["messages"])

    out = generate(ft_model, [
        {"role": "user", "content": prompt}
    ])

    eval_no_prompt.append({
        "prompt": prompt,
        "output": out
    })

with open(f"{args.output_dir}/eval_no_prompt.jsonl", "w", encoding="utf-8") as f:
    for r in eval_no_prompt:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("Saved:", f"{args.output_dir}/eval_no_prompt.jsonl")

# ─────────────────────────────────────────────────────────
# 4B. Evaluation עם system prompt
# ─────────────────────────────────────────────────────────

eval_with_prompt = []

for ex in check_data:
    prompt = extract_user_prompt(ex["messages"])

    out = generate(ft_model, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ])

    eval_with_prompt.append({
        "prompt": prompt,
        "output": out
    })

with open(f"{args.output_dir}/eval_with_prompt.jsonl", "w", encoding="utf-8") as f:
    for r in eval_with_prompt:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("Saved:", f"{args.output_dir}/eval_with_prompt.jsonl")

# ─────────────────────────────────────────────────────────
# 5. הדפסת דוגמאות
# ─────────────────────────────────────────────────────────

print("\n===== Sample Outputs (No Prompt) =====\n")
for r in eval_no_prompt[:3]:
    print("PROMPT:", r["prompt"])
    print("OUTPUT:", r["output"])
    print("-" * 60)

print("\n===== Sample Outputs (With Prompt) =====\n")
for r in eval_with_prompt[:3]:
    print("PROMPT:", r["prompt"])
    print("OUTPUT:", r["output"])
    print("-" * 60)
