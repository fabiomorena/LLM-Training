import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- Pfade und Modellnamen ---
pfad_finales_modell = "./mein-finales-modell"
basis_modell_name = "microsoft/phi-3-mini-4k-instruct"

# --- Lade das trainierte LoRA-Modell ---
print("Lade finales LoRA-Modell...")
basis_modell = AutoModelForCausalLM.from_pretrained(
    basis_modell_name,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(basis_modell_name)
finales_modell = PeftModel.from_pretrained(basis_modell, pfad_finales_modell)
# Sende das Modell an den MPS-Chip für die Berechnung
finales_modell.to("mps")


# --- Bereite den Prompt im Chat-Format vor ---
messages = [
    {"role": "user", "content": "Was ist der Hauptvorteil von LoRA beim Fine-Tuning großer Sprachmodelle?"}
]

# --- KORREKTUR HIER ---
# Schritt 1: Formatiere den Prompt zu einem einzigen String.
prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Schritt 2: Tokenisiere den String, um sowohl 'input_ids' als auch 'attention_mask' zu erhalten.
inputs = tokenizer(prompt_text, return_tensors="pt").to("mps")

print("\nGeneriere Antwort mit dem finalen Modell...")
# Schritt 3: Übergebe beide an die .generate-Funktion mit dem **-Operator.
output_ids = finales_modell.generate(**inputs, max_new_tokens=150)

# Schneide den ursprünglichen Prompt von der Ausgabe ab, um nur die neue Antwort zu erhalten
neue_token_ids = output_ids[0][inputs['input_ids'].shape[-1]:]
# Dekodiere nur die neuen Tokens
response = tokenizer.decode(neue_token_ids, skip_special_tokens=True)

print("\n--- 🤖 ANTWORT DES FINALEN MODELLS ---")
print(response)
print("---------------------------------------")