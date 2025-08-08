import gradio as gr
from llama_cpp import Llama

# --- 1. OPTIMIERTES GGUF-MODELL LADEN (bleibt unverändert) ---
print("Lade optimiertes GGUF-Modell...")

gguf_modell_pfad = "./mein-finales-modell.gguf"

llm = Llama(
    model_path=gguf_modell_pfad,
    n_ctx=2048,
    n_gpu_layers=-1,
    verbose=False
)

print("Optimiertes Modell erfolgreich geladen und bereit.")


# --- 2. DIE LOGIK-FUNKTION ANPASSEN ---
# ÄNDERUNG: Die Funktion muss jetzt 'message' und 'history' als Argumente annehmen.
# 'message' ist die neue Eingabe des Nutzers.
# 'history' ist die bisherige Konversation.
def generiere_antwort(message, history):
    print(f"Neue Anfrage erhalten: '{message}'")

    # Baue den Prompt mit dem Chat-Template des Modells
    prompt = f"<|user|>\n{message}<|end|>\n<|assistant|>"

    # Generiere die Antwort mit llama.cpp
    output = llm(
        prompt,
        max_tokens=250,
        stop=["<|end|>"],
        echo=False
    )

    response = output["choices"][0]["text"]

    print(f"Modell-Antwort: '{response}'")
    return response


# --- 3. DIE NEUE CHAT-OBERFLÄCHE ERSTELLEN ---
# ÄNDERUNG: Wir ersetzen gr.Interface durch gr.ChatInterface.
# Diese Komponente kümmert sich automatisch um die Darstellung der Chathistorie.
iface = gr.ChatInterface(
    fn=generiere_antwort,
    title="Mein eigenes KI-Modell 🤖 (Phi-3 Mini GGUF-Optimiert)",
    chatbot=gr.Chatbot(height=500),  # Gibt der Chatbox eine feste Höhe
    textbox=gr.Textbox(placeholder="Stelle hier deine Frage zum Thema KI-Transformer...", container=False, scale=7),
    theme="gradio/dracula_soft",  # Ein modernes, dunkles Design
    examples=[  # Fügt klickbare Beispiel-Fragen hinzu
        "Was ist die Hauptfunktion eines Transformer-Modells in der KI?",
        "Erkläre den Hauptvorteil von LoRA.",
        "Was sind die Nachteile von großen Sprachmodellen?",
    ],
    cache_examples=False,  # Deaktiviert das Caching für Beispiele
)

# Starte die Web-Oberfläche
iface.launch()
