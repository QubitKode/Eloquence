# app_with_persona.py
import streamlit as st
import os
import pickle
import re
from ingest import parse_pdf
from storage import store_documents
from retrieval import retrieve_relevant_chunks  # Add this import
from persona_retrieval import retrieve_with_persona, PersonaType, get_persona_description
import config

# Streamlit page configuration
st.set_page_config(
    page_title="PDF Question-Answering System with Personas",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a more professional look
st.markdown("""
<style>
    .main {
        padding: 2rem;
        background-color: #f8f9fa;
        width: 100vw;
        min-height: 100vh;
    }
    .stApp {
        max-width: 100vw;
        min-height: 100vh;
        margin: 0;
        padding: 0;
    }
    h1 {
        color: #0066cc;
        margin-bottom: 1.5rem;
    }
    .stButton button {
        background-color: #0066cc;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stTextInput>div>div>input {
        border-radius: 4px;
    }
    .success-message {
        padding: 1rem;
        background-color: #d4edda;
        color: #155724;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .divider {
        height: 1px;
        background-color: #e9ecef;
        margin: 1.5rem 0;
    }
    .reference-content {
        margin-top: 1.5rem;
        padding: 1.5rem;
        background-color: #f8f8f8;
        border-radius: 8px;
        border-left: 4px solid #0066cc;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .reference-content h3 {
        color: #0066cc;
        margin-bottom: 1.2rem;
        font-size: 1.4rem;
    }
    
    .reference-content strong {
        color: #333;
        font-size: 1.1rem;
        display: block;
        margin: 12px 0 8px 0;
        padding-bottom: 5px;
        border-bottom: 1px solid #eaeaea;
    }

    h2 {
        color: #0066cc;
        font-size: 1.4rem;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #e0e0e0;
    }
    
    h3 {
        color: #333;
        font-size: 1.2rem;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
    }
    
    ul {
        margin-bottom: 1rem;
    }
    
    li {
        margin-bottom: 0.3rem;
    }
    
    /* New styles for persona selection */
    .persona-selector {
        background-color: #f0f7ff;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border: 1px solid #d1e7ff;
    }
    
    .persona-selector h3 {
        color: #0066cc;
        margin-top: 0;
        margin-bottom: 0.8rem;
    }
    
    .persona-description {
        font-size: 0.9rem;
        color: #666;
        font-style: italic;
        margin-top: 0.5rem;
    }
    
    /* Visualization container */
    .visualization-container {
        margin-top: 2rem;
        padding: 1.5rem;
        background-color: #f9f9f9;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    
    .visualization-container h3 {
        color: #0066cc;
        margin-top: 0;
        margin-bottom: 1rem;
    }
    
    /* Search results display */
    .search-results-container {
        margin-top: 1.5rem;
        padding: 1rem;
        background-color: #f4f4f4;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    
    .search-term {
        font-weight: bold;
        color: #28a745;
        margin-bottom: 0.5rem;
    }
            
    .block-container {
    padding-top: 1rem !important;
    margin-top: 1rem !important;
}
</style>
           

""", unsafe_allow_html=True)

# Constants
DB_DIRECTORY = "vector_db_cache"
os.makedirs(DB_DIRECTORY, exist_ok=True)

def get_cached_db_path(filename):
    """Generate a consistent path for storing cached database files"""
    safe_filename = "".join(c for c in filename if c.isalnum() or c in "._- ").replace(" ", "_")
    return os.path.join(DB_DIRECTORY, f"{safe_filename}_cache.pkl")

# Title and description
st.markdown(
    "<h1 style='background-color: black; color: white; padding: 1rem; border-radius: 8px; text-align: center;'>Eloquence</h1>",
    unsafe_allow_html=True
)
# Add a divider for better separation
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
# Close the centered container div
st.markdown("</div>", unsafe_allow_html=True)
# Move all configuration features to the sidebar for a true "Configuration Panel"
with st.sidebar:
    st.markdown("<h2>Configuration</h2>", unsafe_allow_html=True)

    # API Key input (if not set in config.py, allow user to input)
    if not config.GENAI_API_KEY:
        api_key_input = st.text_input("Enter your Google Gemini API key", type="password")
        if api_key_input:
            config.GENAI_API_KEY = api_key_input
            # Reconfigure the API with new key
            import google.generativeai as genai
            genai.configure(api_key=api_key_input)

    # File uploader for PDF
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    # Allow force reprocessing with checkbox
    force_reprocess = st.checkbox("Force reprocessing of PDF", value=False, 
                                 help="Check this if you want to regenerate the database for this PDF")

    # Display available cached PDFs
    cached_files = [f.replace("_cache.pkl", "") for f in os.listdir(DB_DIRECTORY) if f.endswith("_cache.pkl")]
    if cached_files:
        st.markdown("### Previously Processed PDFs")
        selected_cached_file = st.selectbox(
            "Select a previously processed PDF to query:",
            ["None"] + cached_files,
            index=0
        )
        if selected_cached_file != "None":
            load_cached = st.button("Load Selected PDF")
            if load_cached:
                cache_path = os.path.join(DB_DIRECTORY, f"{selected_cached_file}_cache.pkl")
                with open(cache_path, "rb") as f:
                    cached_data = pickle.load(f)
                    st.session_state.contents = cached_data["contents"]
                    st.session_state.faiss_index = cached_data["faiss_index"]
                    st.session_state.id_to_chunk = cached_data["id_to_chunk"]
                    st.session_state.db_conn = cached_data["db_conn"]
                    st.session_state.current_file_name = selected_cached_file
                st.success(f"Loaded '{selected_cached_file}' successfully!")

    # Persona selection - put this in the sidebar for better organization
    st.markdown("### Choose User Type")

    selected_persona = st.radio(
        "Choose the type of user persona:",
        [PersonaType.EXPERT.value, PersonaType.INTERMEDIATE.value],
        index=0,
        key="persona_selection"
    )

    # Show description of selected persona
    persona = PersonaType(selected_persona)
    st.markdown(
        f"<div class='persona-description'>{get_persona_description(persona)}</div>",
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    /* Centers the entire Streamlit app content horizontally */
    .stApp {
        display: flex;
        flex-direction: column;
        align-items: center; /* Centers all direct children horizontally */
        padding-top: 0px !important; /* Removes default top padding */
        padding-left: 350px;
        padding-right: 350px;
    }
    </style>
    <div class="centered-container">""",
   unsafe_allow_html=True
)

# Processing logic when a file is uploaded
if uploaded_file:
    # Check if we need to reprocess the file
    # This happens when: file is new OR session state doesn't have required attributes OR force_reprocess is checked
    required_attrs = ['contents', 'faiss_index', 'id_to_chunk', 'db_conn']
    missing_attrs = [attr for attr in required_attrs if attr not in st.session_state]
    
    # Look for cached file
    cache_path = get_cached_db_path(uploaded_file.name)
    cached_exists = os.path.exists(cache_path) and not force_reprocess
    
    need_processing = (
        'current_file_name' not in st.session_state or 
        st.session_state.current_file_name != uploaded_file.name or
        missing_attrs or
        force_reprocess
    )
    
    if need_processing:
        st.session_state.current_file_name = uploaded_file.name
        # Parse the PDF
        with st.spinner("Parsing PDF and indexing content..."):
            contents = parse_pdf(uploaded_file)
            faiss_index, db_conn, id_to_chunk = store_documents(contents)
        
        # Store in session state for reuse during Q&A
        st.session_state.contents = contents
        st.session_state.faiss_index = faiss_index
        st.session_state.id_to_chunk = id_to_chunk
        st.session_state.db_conn = db_conn
        
        # Cache the results for future use
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "contents": contents,
                    "faiss_index": faiss_index,
                    "id_to_chunk": id_to_chunk,
                    "db_conn": db_conn
                }, f)
        except Exception as e:
            st.warning(f"Could not save cache: {e}")
        
        st.success(f"PDF parsed successfully! You can now ask questions about the document.")
    
    elif cached_exists and ('contents' not in st.session_state or force_reprocess):
        # Load from cache
        with st.spinner("Loading cached document..."):
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
                st.session_state.contents = cached_data["contents"]
                st.session_state.faiss_index = cached_data["faiss_index"]
                st.session_state.id_to_chunk = cached_data["id_to_chunk"]
                st.session_state.db_conn = cached_data["db_conn"]
        st.success(f"Loaded cached data for '{uploaded_file.name}'. You can now ask questions.")

# Query section - display if a document is loaded
if 'contents' in st.session_state and st.session_state.contents:
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.subheader("Ask a question about the document")
    query = st.text_input("Enter your question", placeholder="e.g., What percentage of ransomware attacks were there in 2023-2024?")
    
    if query:
        # Get the current persona selection
        persona = PersonaType(st.session_state.persona_selection)
        
        with st.spinner(f"Retrieving answer using {persona.value} persona..."):
            # Use persona-based retrieval
            response = retrieve_with_persona(
                query, 
                st.session_state.faiss_index, 
                st.session_state.db_conn, 
                st.session_state.id_to_chunk,
                persona
            )
            
            if "answer" in response:
                # Format the answer for better structure
                answer = response["answer"]
                
                # Display answer in a professional format
                st.markdown("<div style='background-color:#f0f7ff; padding:20px; border-radius:5px; border-left:4px solid #0066cc;'>", unsafe_allow_html=True)
                st.markdown(answer)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Display sources - extract only the pages cited in the answer
                if "(Page" in answer:
                    # Extract pages that were actually cited in the answer
                    page_refs = re.findall(r'\(Page\s+(\d+)\)', answer)
                    if page_refs:
                        cited_pages = sorted(set(page_refs))
                        st.markdown(f"**Sources:** Pages {', '.join(cited_pages)}")
                
                # Display references from the document (images, diagrams, tables, etc.)
                if "chunks_found" in response and response["chunks_found"] > 0:
                    # Get the results (they're not in the response, so we need to retrieve them again)
                    results = retrieve_relevant_chunks(query, 
                                                        st.session_state.faiss_index, 
                                                        st.session_state.db_conn, 
                                                        st.session_state.id_to_chunk)
                    
                    # Get cited pages
                    cited_pages = set()
                    if "(Page" in answer:
                        cited_pages = set(re.findall(r'\(Page\s+(\d+)\)', answer))
                    else:
                        # If no explicit citations, include all pages that had content
                        cited_pages = set(str(chunk['page']) for chunk in results if 'page' in chunk)
                    
                    # Filter for visual content types
                    visual_chunks = [
                        chunk for chunk in results 
                        if chunk.get("type") in ["image", "diagram", "figure", "table"] 
                        and str(chunk.get('page', '')) in cited_pages
                        and "path" in chunk and chunk["path"] and os.path.exists(chunk["path"])
                    ]
                    
                    # Only display the reference content section if we have visual chunks
                    if visual_chunks:
                        st.markdown("### Visual Content")
                        
                        # Track unique visuals by path to avoid duplicates
                        unique_visuals = {}
                        
                        for chunk in visual_chunks:
                            # Skip if this visual has already been displayed
                            visual_path = chunk.get("path", "")
                            if not visual_path or visual_path in unique_visuals:
                                continue
                            
                            # Mark this visual as displayed
                            unique_visuals[visual_path] = True
                            
                            # Create a section divider between items
                            if len(unique_visuals) > 1:
                                st.markdown("<hr style='margin: 15px 0; border-top: 1px solid #e0e0e0;'>", unsafe_allow_html=True)
                                
                            # Display the image
                            st.image(visual_path, width=400)
                            
                            # Create a header with page information
                            st.markdown(f"**{chunk.get('type', 'Image').capitalize()} (Page {chunk.get('page', 'unknown')})**")
                            
                            # Display caption if available
                            if "caption" in chunk and chunk["caption"]:
                                st.markdown(f"*{chunk['caption']}*")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                
                # For intermediate persona, show additional info
                if persona == PersonaType.INTERMEDIATE:
                    # Display web search results if available
                    if "search_results" in response and response["search_results"]:
                        st.markdown("<div class='search-results-container'>", unsafe_allow_html=True)
                        st.markdown("### Term Explanations")
                        st.markdown("Additional information retrieved to explain technical terms:")
                        
                        for term, result in response["search_results"].items():
                            st.markdown(f"<div class='search-term'>{term}</div>", unsafe_allow_html=True)
                            if isinstance(result, str):
                                # Display the result directly if it's already a string
                                st.markdown(f"{result}")
                            elif isinstance(result, dict) and "results" in result:
                                # Format the results from the search API
                                for i, item in enumerate(result.get("results", []), 1):
                                    st.markdown(f"**{i}.** {item.get('title', '')}")
                                    st.markdown(f"{item.get('snippet', '')}")
                            else:
                                # Fallback for other formats
                                st.markdown(f"Information about {term}")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Display visualizations if available
                    if "visualizations" in response and response["visualizations"]:
                        st.markdown("<div class='visualization-container'>", unsafe_allow_html=True)
                        st.markdown("### Visualizations")
                        
                        for viz in response["visualizations"]:
                            topic = viz.get("topic", "")
                            style = viz.get("style", "")
                            
                            st.markdown(f"**{topic}** ({style})")
                            
                            # Check if result exists and has filepath
                            if "result" in viz and isinstance(viz["result"], dict) and "filepath" in viz["result"]:
                                try:
                                    filepath = viz["result"]["filepath"]
                                    if os.path.exists(filepath):
                                        st.image(filepath, caption=topic, width=600)
                                    else:
                                        st.warning(f"Image file not found: {filepath}")
                                except Exception as e:
                                    st.warning(f"Could not display image: {str(e)}")
                            # Check if result is a string (like an explanation)
                            elif "result" in viz and isinstance(viz["result"], str):
                                st.info(viz["result"])
                            # Check for error message
                            elif "error" in viz:
                                st.warning(f"Error generating visualization: {viz['error']}")
                            else:
                                st.info(f"A visualization for '{topic}' would be shown here.")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                
            else:
                st.error("No answer was generated. Please try a different question.")

# Display a message if no document is loaded
elif 'contents' not in st.session_state:
    st.info("Please upload a PDF document or select a previously processed PDF to start asking questions.")