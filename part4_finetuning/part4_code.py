#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

# הגדרה לטעינה ב-4-bit (יעזור אם תרצי בהמשך להגדיל דאטה)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

# טעינת הטוקנייזר
tokenizer = AutoTokenizer.from_pretrained(model_id)

# טעינת המודל
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

print("Model loaded successfully on GPU!")


# In[ ]:


import json

# 1. טעינת הקובץ
with open('train_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. הדפסת 3 דוגמאות ראשונות לבדיקה
print(f"Total examples found: {len(data)}\n")

# for i, example in enumerate(data[:3]):
#     print(f"--- Example {i+1} ---")
#     # הפורמט של ה-Trainer מצפה ל-messages, בואי נראה מה יש בפנים
#     messages = example.get('messages', [])
#     for msg in messages:
#         role = msg.get('role', 'unknown')
#         content = msg.get('content', '')
#         print(f"[{role.upper()}]: {content}")
#     print("\n")


# In[ ]:


# --- 1. טעינת הדאטה לניסוי (Sanity Check) ---
# נשתמש בשתי הדוגמאות הראשונות כדי לוודא שהאימון משנה את המודל
from datasets import Dataset

# המרה ל-Dataset של Hugging Face
full_dataset = Dataset.from_list(data)
sanity_dataset = full_dataset.select([0, 1])

# --- יצירת שדה text מתוך messages ---
def convert_messages_to_text(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

full_dataset = Dataset.from_list(data)
full_dataset = full_dataset.map(convert_messages_to_text)

# שתי הדוגמאות הראשונות ל-Sanity Check
sanity_dataset = full_dataset.select([0, 1])


# --- 2. בדיקת Baseline (לפני האימון) ---
def generate_response(prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    # הגדרת max_new_tokens כדי שלא ייתקע
    outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

print("--- Checking Baseline (Pre-Training) ---")
test_prompt = sanity_dataset[0]['messages'][0]['content'] 
print(f"Prompt: {test_prompt}")
print(f"Response: {generate_response(test_prompt)}")


# In[ ]:


from trl import SFTTrainer
from peft import LoraConfig
from transformers import TrainingArguments

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

training_args = TrainingArguments(
    output_dir="./sanity_check_results",
    per_device_train_batch_size=1,
    num_train_epochs=50,
    learning_rate=2e-4,
    logging_steps=5,
    save_strategy="no",
    fp16=True
)

trainer = SFTTrainer(
    model=model,
    train_dataset=sanity_dataset,
    peft_config=peft_config,
    args=training_args,
)

print("\n--- Starting Sanity Check Training ---")
trainer.train()
trainer.save_model("./sanity_check_results")


# In[ ]:


from peft import PeftModel

# טוענים מחדש את המודל הבסיסי
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# טוענים את האדפטר
model = PeftModel.from_pretrained(base_model, "./sanity_check_results")
model.eval()

print("Adapter loaded successfully!")


# In[ ]:


for i in range(2):
    prompt = sanity_dataset[i]['messages'][0]['content']
    print("\nPrompt:", prompt)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded_output = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    print("Fine-Tuned Response:", decoded_output)


# # Phase 3: Full Fine-Tuning on 120 Examples

# In[ ]:


import torch
import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig
from transformers import TrainingArguments

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load fresh base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load dataset (now includes system prompt in every example)
with open('train_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Training examples: {len(data)}")
print("Roles in first example:", [m['role'] for m in data[0]['messages']])

def convert_messages_to_text(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

full_dataset = Dataset.from_list(data)
full_dataset = full_dataset.map(convert_messages_to_text)

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

training_args = TrainingArguments(
    output_dir="./qwen_hebrew_lora",
    per_device_train_batch_size=2,
    num_train_epochs=10,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
)

trainer = SFTTrainer(
    model=base_model,
    train_dataset=full_dataset,
    peft_config=peft_config,
    args=training_args,
)

print("Starting full fine-tuning (10 epochs, with system prompt)...")
trainer.train()
trainer.save_model("./qwen_hebrew_lora")
print("Training complete. Adapter saved to ./qwen_hebrew_lora")


# In[ ]:


from peft import PeftModel

# Reload clean base model and attach the fine-tuned adapter
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
ft_model = PeftModel.from_pretrained(base_model, "./qwen_hebrew_lora")
ft_model.eval()

SYSTEM_PROMPT = "You are a helpful assistant. Always respond in Hebrew only, regardless of the language of the question."

def generate_ft(prompt, max_new_tokens=200):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = ft_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

test_prompts = [
    "What is the capital of France?",
    "Give two advantages and two disadvantages of social media.",
    "Write a short email to a colleague asking to reschedule a meeting.",
    "Explain how photosynthesis works.",
    "Suggest three tips for better time management.",
]

print("=== Fine-Tuned Model Verification ===\n")
for prompt in test_prompts:
    print(f"Prompt: {prompt}")
    print(f"Response: {generate_ft(prompt)}")
    print("-" * 60)

