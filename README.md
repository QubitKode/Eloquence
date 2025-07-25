# Eloquence

A **Streamlit**‑based Agentic RAG to upload PDF documents and interact with them using advanced AI‑powered retrieval and language models. Powered by **Google Gemini API** and **HuggingFace** embeddings, Eloquence allows you to ask natural‑language questions about any PDF - tailored to different user personas.


https://github.com/user-attachments/assets/4f853156-cda9-402a-afb5-8e74e0f42a70




---

## 🚀 Features

* 📄 **PDF Upload & Parsing**

  * Extract text, tables, graphs, and images.
* 🤖 **AI‑Powered Q\&A**

  * Ask natural language questions and get context‑aware answers.
* 👥 **Persona‑Based Agentic Responses**

  * Choose between **Expert** or **Intermediate** modes.
* 🖼️ **Rich Citations & Visuals**

  * Inline citations, diagrams, tables, and extracted images.
* ⚙️ **Config Panel**

  * Enter your API key, select persona, toggle force reprocess.
* 💾 **Caching**

  * Reuse previously processed PDFs for speed.

---

## 📥 Installation & Setup

### Prerequisites

* **Python** ≥ 3.8
* **Google Gemini API key** (Get yours at [Google AI Studio](https://ai.google/studio))

### Clone & Install

```bash
git clone https://github.com/your-username/eloquence.git
```

```bash
cd eloquence
```

```bash
python -m venv venv
```

```bash
# On Windows
venv\Scripts\activate
```

```bash
# On macOS/Linux
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

### Configure API Key

1. Open `config.py`.
2. Replace the placeholder:

   ```python
   GENAI_API_KEY = "YOUR_API_KEY_HERE"
   ```

   with your actual key.

---

## ▶️ Running the App

```bash
streamlit run app_with_persona.py
```

This will launch Eloquence in your default browser.

---

## 🖱️ Usage

1. **Upload PDF** via the sidebar uploader.
2. **Force Reprocessing** (optional) to ignore cache.
3. **Select Persona**:

   * **Expert** for deep, technical answers.
   * **Intermediate** for simplified, high‑level explanations.
4. **Ask Questions** in the input box.
5. **View Answers** with inline citations and visuals.

---

## 📂 Project Structure

```
├── app_with_persona.py      # Streamlit app entrypoint
├── config.py                # API keys & settings
├── ingest.py                # PDF parsing & extraction
├── storage.py               # Document caching & indexing
├── retrieval.py             # Chunk retrieval logic
├── persona_retrieval.py     # Persona‑based query handling
├── agent.py                 # AI agent orchestration
├── tools.py                 # Utility functions
├── vector_db_cache/         # Cached embeddings & indexes
└── output/                  # Temp files & extracted assets
```

---

## 🤝 Contributing

Contributions are welcomed! Please:

1. Fork the repository
2. Create a feature branch

   ```bash
   git checkout -b feature/YourFeature
   ```
3. Commit your changes

   ```bash
   git commit -m "Add some feature"
   ```
4. Push to the branch

   ```bash
   git push origin feature/YourFeature
   ```
5. Open a Pull Request

---

---

> “The art of being wise is the art of knowing what to overlook.”
> ― William James
