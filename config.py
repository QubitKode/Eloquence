# config.py
# Configuration for API keys and model names

# Google Gemini API Key (for google-generativeai). Insert your API key here.
GENAI_API_KEY = "YOUR_GEMINI_API_KEY"  # e.g., "YOUR-GEMINI-API-KEY"

# Google Gemini model to use (e.g., "gemini-2.0", "gemini-2.0-chat", or specific version)
GENAI_MODEL = "gemini-1.5-flash"

# Embedding model name (HuggingFace SentenceTransformer)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Number of top relevant chunks to retrieve for answering
TOP_K = 5

# Temporary directory for storing extracted images (and possibly intermediate files)
TEMP_DIR = "output"
