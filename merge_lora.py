import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- 1. Pfade und Namen definieren ---
basis_modell_name = "microsoft/phi-3-mini-4k-instruct"
pfad_lora_adapter = "./mein-finales-modell"
pfad_merged_modell = "./mein-finales-modell-merged"

print(f"Lade Basismodell: {basis_modell_name}")
# Lade das Basismodell und den Tokenizer auf eine einfachere Weise
basis_modell = AutoModelForCausalLM.from_pretrained(
    basis_modell_name,
    trust_remote_code=True,
    #torch_dtype=torch.bfloat16, # Entfernt
    #device_map="auto",          # Entfernt
    #offload_folder=pfad_offload, # Entfernt
)
tokenizer = AutoTokenizer.from_pretrained(basis_modell_name)

print(f"Lade LoRA-Adapter von: {pfad_lora_adapter}")
lora_modell = PeftModel.from_pretrained(basis_modell, pfad_lora_adapter)

# --- 2. Der Verschmelzungs-Schritt ---
print("\nVerschmelze LoRA-Adapter mit dem Basismodell...")
merged_model = lora_modell.merge_and_unload()
print("Verschmelzung abgeschlossen.")

# --- 3. Das neue, eigenständige Modell speichern ---
print(f"Speichere das verschmolzene, eigenständige Modell nach: {pfad_merged_modell}")
merged_model.save_pretrained(pfad_merged_modell)
tokenizer.save_pretrained(pfad_merged_modell)

print(f"\n✅ Fertig! Dein neues, eigenständiges Modell ist jetzt unter '{pfad_merged_modell}' verfügbar.")
