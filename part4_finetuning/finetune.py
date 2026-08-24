import torch
import json
import random
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# ─────────────────────────────────────────────────────────
# טעינת נתוני האימון
# ─────────────────────────────────────────────────────────
with open('train_data.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

print(f"Loaded {len(train_data)} training examples.")

# Stronger system prompt — makes "Hebrew only" constraint more explicit
SYSTEM_PROMPT = "You are a helpful assistant. Always respond in Hebrew only. Never use English, Chinese, or any other language — Hebrew only."

def inject_system_and_format(example):
    messages = []
    for m in example["messages"]:
        if m["role"] == "system":
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        else:
            messages.append(m)
    # If no system message exists, prepend one
    if not any(m["role"] == "system" for m in example["messages"]):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    return {"text": tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )}

# פיצול ל־train/validation (90/10)
random.seed(42)
shuffled = train_data[:]
random.shuffle(shuffled)
split = int(len(shuffled) * 0.9)
train_split, val_split = shuffled[:split], shuffled[split:]

train_dataset = Dataset.from_list(train_split).map(inject_system_and_format)
val_dataset   = Dataset.from_list(val_split).map(inject_system_and_format)

print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

# ─────────────────────────────────────────────────────────
# LoRA — attention-only, low rank to prevent catastrophic forgetting.
# Adding MLP modules (gate_proj etc.) forces too much weight update
# and causes the model to lose its reasoning ability.
# ─────────────────────────────────────────────────────────
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

# ─────────────────────────────────────────────────────────
# Training arguments.
# LR 1e-4 (half of before) preserves more base-model knowledge.
# 3 epochs — README grid search confirmed 2-3 is optimal;
# more epochs overfits to Hebrew style and loses content coherence.
# ─────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./qwen_hebrew_lora_best",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    learning_rate=1e-4,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
    warmup_ratio=0.1,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
)

trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    args=training_args,
)

print("\nStarting fine-tuning...")
trainer.train()
trainer.save_model("./qwen_hebrew_lora_best")

print("\nTraining complete. Model saved.")
