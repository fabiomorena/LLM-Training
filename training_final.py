import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig

# --- Phase 1: Modernes Modell und Tokenizer laden ---
modell_name = "microsoft/phi-3-mini-4k-instruct"
print(f"Lade Basismodell: {modell_name}")

tokenizer = AutoTokenizer.from_pretrained(modell_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    modell_name,
    trust_remote_code=True,
)

# --- Phase 2: LoRA Konfiguration für Phi-3 ---
print("\nKonfiguriere LoRA-Adapter für Phi-3...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# --- DER FINALE PATCH ---
# ÄNDERUNG 1: Stellt die Gradienten-Kette für das Checkpointing wieder her.
model.enable_input_require_grads()
# ÄNDERUNG 2: Deaktiviert den Cache explizit, was für Checkpointing erforderlich ist.
model.config.use_cache = False

print("Trainierbare Parameter nach Anwendung von LoRA:")
model.print_trainable_parameters()

# --- Phase 3: Dataset vorbereiten ---
print("\nLade und verarbeite das Chat-Dataset...")
dataset = load_dataset('json', data_files='chat_dataset.jsonl', split='train')

def tokenisiere_chat_format(beispiele):
    formattierter_prompt = tokenizer.apply_chat_template(beispiele["messages"], tokenize=False)
    outputs = tokenizer(formattierter_prompt, truncation=True, padding="max_length", max_length=512)
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

tokenized_dataset = dataset.map(tokenisiere_chat_format)

# --- Phase 4: Training ---
trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=TrainingArguments(
        output_dir="./results_final_phi3",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        num_train_epochs=2,
        learning_rate=2e-4,
        logging_steps=5,
        optim="adafactor",
    ),
)

print("\n🚀 Starte finales Fine-Tuning mit Gradient Checkpointing...")
trainer.train()

print("\n🎉 FINALES Training erfolgreich abgeschlossen!")
trainer.save_model("./mein-finales-modell")
print("💾 Finale LoRA-Adapter wurden erfolgreich im Ordner 'mein-finales-modell' gespeichert.")
