# Fine-Tuning & RAG System on a MacBook Air

This project documents the entire process of fine-tuning, optimizing, and extending a modern language model (`microsoft/phi-3-mini-instruct`) into a RAG system on an Apple MacBook Air. It serves as a practical example of the challenges and solutions when working with large language models on consumer hardware.

---
## ✨ Features

* **End-to-end workflow:** From idea to training to the finished interactive application.
* **Modern AI model:** Fine-tuning the powerful `phi-3-mini-instruct` (3.8B parameters).
* **Efficient training:** Use of **PEFT/LoRA** to minimize memory and compute requirements.
* **Inference optimization:** Conversion of the trained model into the high-performance **GGUF format** with 8-bit quantization.
* **Retrieval-Augmented Generation (RAG):** The model can answer questions based on content from custom PDF documents.
* **Conversational memory:** The RAG application uses "Query Transformation" to understand follow-up questions in conversational context.
* **Interactive web app:** A **Gradio**-built interface for easy interaction with the final model.

---
## 🛠️ Technologies Used

* **Python 3.11**
* **PyTorch** (with MPS acceleration for Apple Silicon)
* **Hugging Face libraries:**
    * `transformers`
    * `peft` (for LoRA)
    * `datasets`
    * `accelerate`
* **RAG & inference:**
    * `llama-cpp-python` (for GGUF models)
    * `chromadb` (vector database)
    * `sentence-transformers` (for embeddings)
    * `pypdf` (for reading PDFs)
* **UI & development:**
    * `gradio` (for the web app)
    * Git & GitHub
    * PyCharm on macOS

---
## 🚀 Setup & Installation

1.  **Clone repository:**
    ```bash
    git clone https://github.com/fabiomorena/LLM-Training.git
    cd LLM-Training
    ```

2.  **Create and activate virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install torch transformers datasets peft accelerate gradio llama-cpp-python sentencepiece chromadb pypdf
    ```

---
## 🏃‍♀️ Usage

The core of the project is the final RAG application.

### Start the RAG app

1.  **Provide documents:** Create a folder named `rag_dokumente` and place the PDF files you want to chat with inside.
2.  **Build knowledge base:** Run the script to process the documents and store them in the vector database.
    ```bash
    python create_database.py
    ```
3.  **Start app:** Launch the Gradio application.
    ```bash
    python app.py
    ```
4.  Open the local URL (e.g., `http://127.0.0.1:7860`) in your web browser and ask questions about your documents.

---
## 🧗‍♂️ Project Journey: Challenges & Solutions

* **Dependency conflicts:** Debugged incompatibilities between `transformers`, `peft`, and the `phi-3` model code by installing a compatible set of versions.
* **Hardware limits:** Overcame RAM constraints using **gradient checkpointing** and a memory-efficient **optimizer (`adafactor`)**.
* **Optimization toolchain:** Switched from `bitsandbytes` to the `llama.cpp/GGUF` toolchain for Mac-friendly, high-performance inference.
* **Conversational memory for RAG:** Implemented **"Query Transformation"** to make the model understand follow-up questions.

---
## 📄 License

MIT License.

---
## 👤 Author

* **Fabio Morena**
* GitHub: [github.com/fabiomorena](https://github.com/fabiomorena)