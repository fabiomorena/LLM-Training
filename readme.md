# Fine-Tuning eines KI-Modells auf einem MacBook Air

Dieses Projekt dokumentiert den gesamten Prozess des Fine-Tunings und der Optimierung eines modernen Sprachmodells (`microsoft/phi-3-mini-instruct`) auf einem Apple MacBook Air. Es dient als praxisnahes Beispiel für die Herausforderungen und Lösungen bei der Arbeit mit großen Sprachmodellen auf Consumer-Hardware.

---
## ✨ Features

* **End-to-End-Workflow:** Von der Idee über das Training bis zur fertigen, interaktiven Anwendung.
* **Modernes KI-Modell:** Fine-Tuning des leistungsstarken `phi-3-mini-instruct` (3.8 Mrd. Parameter).
* **Effizientes Training:** Einsatz von **PEFT/LoRA** zur Minimierung des Speicher- und Rechenbedarfs.
* **Inferenz-Optimierung:** Konvertierung des trainierten Modells in das hochperformante **GGUF-Format** mit 8-Bit-Quantisierung.
* **Interaktive Web-App:** Eine mit **Gradio** erstellte Benutzeroberfläche für die einfache Interaktion mit dem finalen Modell.

---
## 🛠️ Verwendete Technologien

* **Python 3.11**
* **PyTorch** (mit MPS-Beschleunigung für Apple Silicon)
* **Hugging Face Libraries:**
    * `transformers`
    * `peft` (für LoRA)
    * `datasets`
    * `accelerate`
* **Inferenz & UI:**
    * `llama-cpp-python` (für GGUF-Modelle)
    * `gradio` (für die Web-App)
* **Entwicklungsumgebung:** PyCharm auf macOS

---
## 🚀 Setup & Installation

1.  **Repository klonen:**
    ```bash
    git clone [https://github.com/fabiomorena/KI-Modell-Training-Mac.git](https://github.com/fabiomorena/KI-Modell-Training-Mac.git)
    cd KI-Modell-Training-Mac
    ```

2.  **Virtuelle Umgebung erstellen und aktivieren:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Abhängigkeiten installieren:**
    ```bash
    pip install torch transformers datasets peft accelerate gradio llama-cpp-python sentencepiece
    ```

---
## 🏃‍♀️ Anwendung

Dieses Projekt besteht aus mehreren Skripten, die den gesamten Lebenszyklus des Modells abbilden.

### ### 1. Das Modell trainieren (Optional)

Das Herzstück ist das Skript `training_final.py`. Um ein eigenes Training durchzuführen:
1.  Erstelle eine `chat_dataset.jsonl`-Datei mit deinen eigenen Frage-Antwort-Paaren.
2.  Führe das Trainingsskript aus:
    ```bash
    python training_final.py
    ```
    Dies erzeugt die LoRA-Adapter im Ordner `mein-finales-modell`.

### ### 2. Das Modell optimieren (Optional)

Um das trainierte Modell schnell und klein zu machen, sind zwei Schritte nötig:
1.  **LoRA-Adapter verschmelzen:**
    ```bash
    python merge_lora.py
    ```
2.  **In GGUF konvertieren und quantisieren:**
    ```bash
    # Lade zuerst das Konvertierungsskript
    git clone [https://github.com/ggerganov/llama.cpp.git](https://github.com/ggerganov/llama.cpp.git)
    
    # Führe die Konvertierung aus
    python llama.cpp/convert_hf_to_gguf.py ./mein-finales-modell-merged --outfile ./mein-finales-modell.gguf --outtype q8_0
    ```

### ### 3. Die fertige App starten

Um mit dem vorab trainierten und optimierten Modell zu chatten, starte die Gradio-Anwendung:
```bash
python app.py
```
Öffne anschließend die lokale URL (z.B. `http://127.0.0.1:7860`) in deinem Webbrowser.

---
## 🧗‍♂️ Die Projekt-Reise: Herausforderungen & Lösungen

Dieses Projekt war eine intensive Lernreise. Die größten Hürden und ihre Lösungen waren:
* **Abhängigkeitskonflikte:** Moderne KI-Bibliotheken entwickeln sich schnell. Ein zentraler Teil des Projekts war das Debuggen von Inkompatibilitäten zwischen `transformers`, `peft` und dem `phi-3`-Modellcode. Die Lösung war die Installation eines "goldenen Trios" von Versionen, die nachweislich miteinander harmonieren.
* **Hardware-Grenzen:** Das Training eines 3.8-Milliarden-Parameter-Modells sprengte die RAM-Grenzen des MacBook Air. Dies wurde durch clevere Software-Tricks wie **Gradient Checkpointing** und einen speichersparenden **Optimizer (`adafactor`)** umgangen.
* **Optimierungs-Toolchain:** Die Quantisierung mit `bitsandbytes` scheiterte an der Inkompatibilität mit Apple Silicon. Der Wechsel zur `llama.cpp/GGUF`-Toolchain war die Lösung, um eine hochperformante, Mac-freundliche Version des Modells zu erstellen.

---
## 📄 Lizenz

Dieses Projekt steht unter der MIT-Lizenz.

---
## 👤 Autor

* **Fabio Morena**
* GitHub: [github.com/fabiomorena](https://github.com/fabiomorena)