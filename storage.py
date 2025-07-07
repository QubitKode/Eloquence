# storage.py
import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json

import config

# Initialize embedding model (this can be slow, so ideally load once)
# We assume using a smaller model for speed; ensure the model name is correct in config.
_embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)

def init_db(conn):
    """Initialize the SQLite database schema (documents table)."""
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            page INTEGER,
            type TEXT,
            content TEXT,
            image_path TEXT,
            table_csv TEXT,
            structured_data TEXT,
            related_id INTEGER
        )
    """)
    conn.commit()

def store_documents(contents):
    """
    Store parsed document contents into SQLite and build a FAISS index.
    Returns the FAISS index, SQLite connection, and a dict mapping doc IDs to content dict.
    """
    # Use in-memory database (or persistent file if needed)
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    cur = conn.cursor()
    id_to_chunk = {}  # map from row id to content dict
    
    # Prepare data for embedding
    texts = []
    ids = []
    for chunk in contents:
        page = chunk["page"]
        ctype = chunk["type"]
        text = chunk["content"]
        img_path = chunk.get("path", "") if ctype == "image" else ""
        table_csv = ""
        if ctype == "table":
            # For tables, we already have CSV text in content; optionally store separately
            table_csv = chunk["content"]  # here content is already CSV text
        # Insert into DB
        cur.execute(
            "INSERT INTO documents(page, type, content, image_path, table_csv) VALUES (?, ?, ?, ?, ?)",
            (page, ctype, text, img_path, table_csv)
        )
        doc_id = cur.lastrowid
        # Map id to full chunk (including data like DataFrame if present) for later use
        id_to_chunk[doc_id] = chunk
        # Collect text for embedding
        texts.append(text)
        ids.append(doc_id)
    
    # Special handling for tables to improve retrieval
    for chunk in contents:
        if chunk["type"] == "table":
            # Store caption separately for better matching
            if "caption" in chunk and chunk["caption"]:
                caption_text = chunk["caption"]
                
                # Create a separate embedding for the caption
                caption_embedding = _embedding_model.encode([caption_text], convert_to_numpy=True).astype('float32')
                
                # Add caption embedding to index with reference to original table
                if caption_embedding.shape[0] > 0:
                    caption_doc_id = cur.lastrowid + 1
                    cur.execute(
                        "INSERT INTO documents(page, type, content, related_id) VALUES (?, ?, ?, ?)",
                        (chunk["page"], "table_caption", caption_text, doc_id)
                    )
                    
                    # Store mapping to original table
                    id_to_chunk[caption_doc_id] = {"type": "table_caption", "related_id": doc_id}
                    
                    # Add to embeddings collection
                    texts.append(caption_text)
                    ids.append(caption_doc_id)
            
            # Store tabular data in structured form if available
            if "tabular_data" in chunk:
                # Store serialized structured data for faster querying
                cur.execute(
                    "UPDATE documents SET structured_data = ? WHERE id = ?",
                    (json.dumps(chunk["tabular_data"]), doc_id)
                )
    
    conn.commit()
    
    # Compute embeddings for all texts
    if texts:
        embeddings = _embedding_model.encode(texts, convert_to_numpy=True)
        embeddings = embeddings.astype('float32')
    else:
        embeddings = np.array([], dtype='float32').reshape(0, 384)  # no content case
    
    # Build FAISS index
    if embeddings.shape[0] > 0:
        dim = embeddings.shape[1]
    else:
        dim = 384  # default dim for model (fallback if no content)
    index = faiss.IndexFlatL2(dim)
    # Use IndexIDMap to store IDs
    index_id_map = faiss.IndexIDMap(index)
    if embeddings.shape[0] > 0:
        index_id_map.add_with_ids(embeddings, np.array(ids, dtype='int64'))
    else:
        index_id_map = index  # empty index (no content)
    return index_id_map, conn, id_to_chunk
