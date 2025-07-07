# retrieval.py
import numpy as np
import re
from storage import _embedding_model  # use the same loaded model
import config

def retrieve_relevant_chunks(query, faiss_index, db_conn, id_to_chunk):
    """Embed the query and retrieve top-K relevant document chunks from the FAISS index."""
    # Check if query mentions a specific section number
    section_match = re.search(r'(\d+\.\d+(?:\.\d+)*)', query)
    section_number = section_match.group(1) if section_match else None
    
    # Embed the query to vector
    query_vec = _embedding_model.encode([query], convert_to_numpy=True).astype('float32')
    if faiss_index is None or faiss_index.ntotal == 0:
        return []  # no data to search
    
    # Search FAISS index - use more results if we're looking for a section
    k = config.TOP_K * 2 if section_number else config.TOP_K
    D, I = faiss_index.search(query_vec, k)  # distances and indices
    
    # I is an array of shape (1, k) of IDs
    top_ids = [int(x) for x in I[0] if x != -1]
    results = []
    if not top_ids:
        return results
    
    # Fetch corresponding chunks from id_to_chunk (which contains full info)
    for doc_id in top_ids:
        if doc_id in id_to_chunk:
            chunk = id_to_chunk[doc_id]
            results.append(chunk)
    
    # If we're looking for a specific section, post-process to include visual elements
    if section_number:
        # Get page numbers where this section appears
        section_pages = []
        for chunk in results:
            if chunk["type"] == "text" and section_number in chunk.get("content", ""):
                section_pages.append(chunk["page"])
        
        # If we found the section on specific pages, include all visual elements from those pages
        if section_pages:
            visual_elements = []
            for doc_id, chunk in id_to_chunk.items():
                if (chunk["type"] in ["image", "table", "diagram", "figure"] and 
                    chunk["page"] in section_pages and 
                    doc_id not in top_ids):
                    visual_elements.append(chunk)
            
            # Add visual elements to results
            results.extend(visual_elements)
    
    # Check if query asks for a specific figure or table
    fig_ref_match = re.search(r'(figure|fig\.|table|chart|diagram)\s*(\d+(?:\.\d+)?)', query, re.IGNORECASE)
    if fig_ref_match:
        ref_type = fig_ref_match.group(1).lower()
        ref_num = fig_ref_match.group(2)
        ref_key = f"{ref_type}{ref_num}"
        
        # Look for visual elements with this reference number
        for doc_id, chunk in id_to_chunk.items():
            if chunk["type"] in ["image", "table", "diagram", "figure"]:
                # Check if this element has reference_numbers
                if "reference_numbers" in chunk and ref_key in chunk["reference_numbers"]:
                    if doc_id not in top_ids:
                        results.append(chunk)
                # Also check captions and surrounding text
                elif chunk.get("content", "").lower().find(ref_key) >= 0:
                    if doc_id not in top_ids:
                        results.append(chunk)
    
    # Improve handling of visual elements mentioned in the query
    visual_keywords = {
        "image": ["image", "picture", "photo", "illustration"],
        "table": ["table", "grid", "tabular", "column", "row"],
        "diagram": ["diagram", "chart", "graph", "plot", "figure", "fig"]
    }

    # Check if query is asking about visual elements
    visual_type_mentioned = None
    for v_type, keywords in visual_keywords.items():
        if any(kw in query.lower() for kw in keywords):
            visual_type_mentioned = v_type
            break

    # If the query specifically asks about visual elements, prioritize them
    if visual_type_mentioned:
        # Find all elements of the mentioned type
        visual_elements = []
        for doc_id, chunk in id_to_chunk.items():
            if chunk["type"] == visual_type_mentioned or (
                visual_type_mentioned == "diagram" and chunk["type"] in ["diagram", "chart", "figure"]
            ):
                # Check if not already in results
                if doc_id not in top_ids:
                    visual_elements.append(chunk)
        
        # Add to beginning of results to prioritize them
        results = visual_elements + results

    return results
