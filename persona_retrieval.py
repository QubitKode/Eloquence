"""
Persona-based retrieval system.
Supports expert and intermediate personas using different retrieval strategies.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import time

# Import existing retrieval system
from retrieval import retrieve_relevant_chunks
from llm import generate_answer
from agent import run_agent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PersonaType(str, Enum):
    """Enum for different persona types."""
    EXPERT = "expert"
    INTERMEDIATE = "intermediate"

def get_persona_description(persona: PersonaType) -> str:
    """Get a description of the persona."""
    descriptions = {
        PersonaType.EXPERT: (
            "Expert level persona that provides comprehensive, technical answers. "
            "Assumes deep domain knowledge and includes specific details."
        ),
        PersonaType.INTERMEDIATE: (
            "Intermediate level persona that explains concepts clearly with additional context. "
            "Includes explanations of technical terms and helpful visualizations."
        )
    }
    return descriptions.get(persona, "Unknown persona")

def retrieve_with_persona(
    question: str, 
    faiss_index: Any, 
    db_conn: Any, 
    id_to_chunk: Dict[int, Dict[str, Any]], 
    persona: PersonaType = PersonaType.EXPERT
) -> Dict[str, Any]:
    """
    Retrieve information based on the selected persona.
    
    Args:
        question: The user's question
        faiss_index: FAISS index for retrieval
        db_conn: Database connection
        id_to_chunk: Mapping of chunk IDs to content
        persona: The persona type to use for retrieval
        
    Returns:
        Dict containing answer and any additional information based on persona
    """
    # Step 1: Retrieve relevant chunks (same for all personas)
    start_time = time.time()
    logger.info(f"Retrieving with {persona} persona for question: {question}")
    
    results = retrieve_relevant_chunks(question, faiss_index, db_conn, id_to_chunk)
    
    if not results:
        logger.warning("No relevant content found for question")
        return {
            "answer": "I couldn't find relevant information in the document to answer your question.",
            "persona": persona,
            "chunks_found": 0
        }
    
    # Step 2: Process based on persona
    if persona == PersonaType.EXPERT:
        # Use the default retrieval and answer generation
        answer = generate_answer(question, results)
        
        retrieval_time = time.time() - start_time
        logger.info(f"Expert retrieval completed in {retrieval_time:.2f} seconds")
        
        return {
            "answer": answer,
            "persona": persona,
            "chunks_found": len(results)
        }
    
    elif persona == PersonaType.INTERMEDIATE:
        # Use the agent-based retrieval for intermediate users
        agent_response = run_agent(question, results)
        
        retrieval_time = time.time() - start_time
        logger.info(f"Intermediate retrieval completed in {retrieval_time:.2f} seconds")
        
        return {
            "answer": agent_response.get("answer", "Error generating answer"),
            "persona": persona,
            "chunks_found": len(results),
            "search_results": agent_response.get("search_results", {}),
        }
    
    else:
        # Fallback to expert persona for unknown types
        logger.warning(f"Unknown persona type: {persona}, falling back to expert")
        answer = generate_answer(question, results)
        
        return {
            "answer": answer,
            "persona": PersonaType.EXPERT,
            "chunks_found": len(results)
        }