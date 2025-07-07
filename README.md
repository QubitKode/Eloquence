[![Python Version](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-✔️-orange)](https://streamlit.io/)  
[![Google Gemini API](https://img.shields.io/badge/Google%20Gemini%20API-Configured-blueviolet)](https://ai.google/studio)  

# Eloquence

A **Streamlit**-based web application to upload PDF documents and interact with them using advanced AI-powered retrieval and language models. Powered by **Google Gemini API** and **HuggingFace** embeddings, Eloquence allows you to ask natural-language questions about any PDF—tailored to different user personas.

---

## 🚀 Features

- 📄 **PDF Upload & Parsing**  
  - Extract text, tables, graphs, and images.  
- ⚡ **Fast Retrieval**  
  - Index & cache document data for instant lookup.  
- 🤖 **AI-Powered Q&A**  
  - Ask natural language questions and get context-aware answers.  
- 👥 **Persona-Based Responses**  
  - Choose between **Expert** or **Intermediate** modes.  
- 🖼️ **Rich Citations & Visuals**  
  - Inline citations, diagrams, tables, and extracted images.  
- ⚙️ **Config Panel**  
  - Enter your API key, select persona, toggle force reprocess.  
- 💾 **Caching**  
  - Reuse previously processed PDFs for speed.

---

## 📥 Installation & Setup

### Prerequisites

- **Python** ≥ 3.8  
- **Google Gemini API key** (Get yours at [Google AI Studio](https://ai.google/studio))

### Clone & Install

```bash
# Clone the repo
git clone https://github.com/your-username/eloquence.git
cd eloquence

# (Optional) Create & activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
Configure API Key
Open config.py.

Replace the placeholder GENAI_API_KEY = "YOUR_API_KEY_HERE" with your actual key.

▶️ Running the App
bash
Copy
Edit
streamlit run app_with_persona.py
This will launch Eloquence in your default browser.

🖱️ Usage
Upload PDF via the sidebar uploader.

Force Reprocessing (optional) to ignore cache.

Select Persona:

Expert for deep, technical answers.

Intermediate for simplified, high-level explanations.

Ask Questions in the input box.

View Answers with inline citations and visuals.


📂 Project Structure
bash
Copy
Edit
├── app_with_persona.py      # Streamlit app entrypoint
├── config.py                # API keys & settings
├── ingest.py                # PDF parsing & extraction
├── storage.py               # Document caching & indexing
├── retrieval.py             # Chunk retrieval logic
├── persona_retrieval.py     # Persona-based query handling
├── agent.py                 # AI agent orchestration
├── tools.py                 # Utility functions
├── vector_db_cache/         # Cached embeddings & indexes
└── output/                  # Temp files & extracted assets
🤝 Contributing
Contributions are welcomed! Please:

Fork the repository

Create a feature branch (git checkout -b feature/YourFeature)

Commit your changes (git commit -m 'Add some feature')

Push to the branch (git push origin feature/YourFeature)

Open a Pull Request
