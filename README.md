Eloquence is a Streamlit-based web application that allows users to upload PDF documents and ask questions about their content using advanced AI-powered retrieval and language models. The app leverages Google Gemini API and HuggingFace embeddings to provide accurate and context-aware answers. It supports different user personas (Expert and Intermediate) to tailor responses accordingly.

## Features

- Upload and parse PDF documents for content extraction along with the visuals (Table, Graph, Images).
- Index and cache document data for fast retrieval.
- Ask natural language questions about the uploaded documents.
- Persona-based Agentic responses: choose between Expert and Intermediate user types.
- Display answers with citations, visual content (images, diagrams, tables), and additional explanations.
- Configuration panel for API key input and persona selection.
- Caching mechanism to avoid reprocessing PDFs unnecessarily.

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key (Get Your free API key from Google AI Studio)

### Installation Steps

1. Clone the repository or download the source code.

2. (Optional but recommended) Create and activate a virtual environment:

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
Install the required dependencies:
pip install -r requirements.txt
Configure your Google Gemini API key:
Open config.py
Replace the placeholder value of GENAI_API_KEY with your actual API key.

Running the Application
Start the Streamlit app by running:

streamlit run app_with_persona.py
This will open the web app in your default browser.

Using the App
Upload a PDF document using the file uploader in the sidebar.

Optionally, select "Force reprocessing" to re-index the document even if cached data exists.

Choose the user persona (Expert or Intermediate) to tailor the responses.

Ask questions about the document in the input box.

View answers with citations, visual content, and additional explanations depending on the persona.

Previously processed PDFs are cached and can be loaded from the sidebar for faster access.

Project Structure Overview
app_with_persona.py: Main Streamlit app file.
config.py: Configuration for API keys and model settings.
ingest.py: PDF parsing and content extraction.
storage.py: Document storage and indexing.
retrieval.py: Retrieval of relevant document chunks.
persona_retrieval.py: Persona-based retrieval logic.
agent.py, tools.py: Supporting modules for AI interaction and utilities.
vector_db_cache/: Directory for cached document data.
output/: Temporary files and extracted images.
Contributing
Contributions are welcome! Please open issues or submit pull requests for improvements or bug fixes.
