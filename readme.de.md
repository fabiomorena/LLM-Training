# Fine-Tuning & RAG-System auf einem MacBook Air

Dieses Projekt dokumentiert den gesamten Prozess des Fine-Tunings, der Optimierung und der Erweiterung eines modernen Sprachmodells (`microsoft/phi-3-mini-instruct`) zu einem RAG-System auf einem Apple MacBook Air. Es dient als praxisnahes Beispiel für die Herausforderungen und Lösungen bei der Arbeit mit großen Sprachmodellen auf Consumer-Hardware.

---
## ✨ Features

* **End-to-End-Workflow:** Von der Idee über das Training bis zur fertigen, interaktiven Anwendung.
* **Modernes KI-Modell:** Fine-Tuning des leistungsstarken `phi-3-mini-instruct` (3.8 Mrd. Parameter).
* **Effizientes Training:** Einsatz von **PEFT/LoRA** zur Minimierung des Speicher- und Rechenbedarfs.
* **Inferenz-Optimierung:** Konvertierung des trainierten Modells in das hochperformante **GGUF-Format** mit 8-Bit-Quantisierung.
* **Retrieval-Augmented Generation (RAG):** Das Modell kann Fragen basierend auf Inhalten aus benutzerdefinierten PDF-Dokumenten beantworten.
* **Konversationelles Gedächtnis:** Die RAG-Anwendung nutzt "Query Transformation", um Folgefragen im Gesprächskontext zu verstehen.
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
* **RAG & Inferenz:**
    * `llama-cpp-python` (für GGUF-Modelle)
    * `chromadb` (Vektor-Datenbank)
    * `sentence-transformers` (für Embeddings)
    * `pypdf` (zum Lesen von PDFs)
* **UI & Entwicklung:**
    * `gradio` (für die Web-App)
    * Git & GitHub
    * PyCharm auf macOS

---
## 🚀 Setup & Installation

1.  **Repository klonen:**
    ```bash
    git clone [https://github.com/fabiomorena/LLM-Training.git](https://github.com/fabiomorena/LLM-Training.git)
    cd LLM-Training
    ```

2.  **Virtuelle Umgebung erstellen und aktivieren:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Abhängigkeiten installieren:**
    ```bash
    pip install torch transformers datasets peft accelerate gradio llama-cpp-python sentencepiece chromadb pypdf
    ```

---
## 🏃‍♀️ Anwendung

Das Herzstück des Projekts ist die finale RAG-Anwendung.

### ### Die RAG-App starten

1.  **Dokumente bereitstellen:** Erstelle einen Ordner namens `rag_dokumente` und lege die PDF-Dateien hinein, mit denen du chatten möchtest.
2.  **Wissensdatenbank erstellen:** Führe das Skript aus, um die Dokumente zu verarbeiten und in der Vektor-Datenbank zu speichern.
    ```bash
    python create_database.py
    ```
3.  **App starten:** Starte die Gradio-Anwendung.
    ```bash
    python app.py
    ```
4.  Öffne anschließend die lokale URL (z.B. `http://127.0.0.1:7860`) in deinem Webbrowser und stelle Fragen zu deinen Dokumenten.

---
## 🧗‍♂️ Die Projekt-Reise: Herausforderungen & Lösungen

Dieses Projekt war eine intensive Lernreise. Die größten Hürden und ihre Lösungen waren:
* **Abhängigkeitskonflikte:** Moderne KI-Bibliotheken entwickeln sich schnell. Ein zentraler Teil des Projekts war das Debuggen von Inkompatibilitäten zwischen `transformers`, `peft` und dem `phi-3`-Modellcode. Die Lösung war die Installation eines "goldenen Trios" von zueinander passenden Bibliotheksversionen.
* **Hardware-Grenzen:** Das Training eines 3.8-Milliarden-Parameter-Modells sprengte die RAM-Grenzen des MacBook Air. Dies wurde durch clevere Software-Tricks wie **Gradient Checkpointing** und einen speichersparenden **Optimizer (`adafactor`)** umgangen.
* **Optimierungs-Toolchain:** Die Quantisierung mit `bitsandbytes` scheiterte an der Hardware-Inkompatibilität. Der Wechsel zur `llama.cpp/GGUF`-Toolchain war die Lösung, um eine hochperformante, Mac-freundliche Version des Modells zu erstellen. Dieser Wechsel brachte eigene Herausforderungen mit sich (veraltete Skripte, fehlende Dateien), die schrittweise gelöst wurden.
* **Konversationelles Gedächtnis für RAG:** Das RAG-System konnte anfangs keine Folgefragen verstehen. Dies wurde durch die Implementierung einer **"Query Transformation"** gelöst, bei der das LLM die Nutzeranfrage basierend auf dem Gesprächsverlauf selbst präzisiert.

---
## 📄 Lizenz

Dieses Projekt steht unter der MIT-Lizenz..

---
## 👤 Autor

* **Fabio Morena**
* GitHub: [github.com/fabiomorena](https://github.com/fabiomorena)