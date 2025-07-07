# llm.py
import google.generativeai as genai
import config

if not config.GENAI_API_KEY:
    raise ValueError("Add your Gemini key.GENAI_API_KEY")

genai.configure(api_key=config.GENAI_API_KEY)

MODEL_NAME = config.GENAI_MODEL or "gemini-pro"          
_GEMINI = genai.GenerativeModel(MODEL_NAME)


def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Parameters
    ----------
    question : str
        The user's natural-language question.
    retrieved_chunks : list of dicts
        Each dict has keys:
            page    → int   (1-indexed PDF page)
            type    → "text" | "image" | "table"
            content → str   (text, OCR, or CSV)
    Returns
    -------
    str : the assistant's answer, with page-number citations.
    """
    # 1) System instruction (steers citation behaviour)
    system_instruction = (
        "You are a helpful assistant answering questions about a PDF. "
        "Base your answer ONLY on the excerpts provided. "
        "IMPORTANT: Provide comprehensive, detailed answers with clear structure and organization. "
        "Always use headings (## Main Heading and ### Subheading) to organize your response. "
        "Break your answer into relevant sections such as: Summary, Key Details, Analysis, and Implications. "
        "Pay SPECIAL ATTENTION to numerical data, statistics, and percentages mentioned in diagrams and charts. "
        "When asked about specific numbers or percentages, ALWAYS quote the EXACT figures from the provided content. "
        "Do not round or approximate numerical values. For example, use 25.79% instead of ~26%. "
        "NEVER state that information is unavailable if it exists in the provided content. "
        "If a question asks for numerical data and the answer is in the content, you MUST provide that exact data. "
        "Cite page numbers like \"(Page 3)\" for every factual claim. "
        "For images and tables, always reference them as \"Table on page X\" or \"Image on page X\" in your answer. "
        "If information comes from diagrams or charts, explicitly mention this in your answer, e.g., \"According to the chart on page 5...\". "
        "When appropriate, use bullet points or numbered lists to organize information clearly. "
        "Ensure your answers provide thorough context and meaningful insights from the provided content."
    )

    # 2) Build a single prompt string from the retrieved chunks
    context_lines = []
    
    # First, check if visual elements are specifically mentioned in the question
    visual_elements_focus = any(term in question.lower() for term in 
                               ["image", "picture", "table", "diagram", "chart", 
                                "figure", "graph", "illustration", "visual"])

    # Prioritize visual chunks if they're specifically asked about
    if visual_elements_focus:
        # Move visual chunks to the front of the context
        visual_chunks = [chunk for chunk in retrieved_chunks if chunk.get("type", "") in 
                         ("image", "diagram", "figure", "table", "chart")]
        text_chunks = [chunk for chunk in retrieved_chunks if chunk.get("type", "") not in 
                       ("image", "diagram", "figure", "table", "chart")]
        retrieved_chunks = visual_chunks + text_chunks

    # First, organize chunks by type and page for better context understanding
    text_chunks = []
    visual_chunks = []
    
    for chunk in retrieved_chunks:
        page = chunk["page"]
        ctype = chunk.get("type", "")
        
        if ctype in ("image", "diagram", "figure", "table"):
            visual_chunks.append(chunk)
        else:
            text_chunks.append(chunk)
    
    # Add text chunks first (typically contain the main content)
    for chunk in text_chunks:
        page = chunk["page"]
        text = chunk.get("content", "")
        
        # Truncate very long text
        if len(text) > 800:
            text = text[:800] + " …"
        context_lines.append(f"[Page {page}] TEXT: {text}")
    
    # Then add visual chunks with special formatting
    for chunk in visual_chunks:
        page = chunk["page"]
        ctype = chunk.get("type", "")
        text = chunk.get("content", "")
        path = chunk.get("path", "")
        
        # Add reference information to help the model know this is a visual element
        if path:
            visual_reference = f"[VISUAL ELEMENT: {ctype} from page {page}]"
            context_lines.append(visual_reference)
        
        # For images/diagrams/figures, integrate numerical data with analysis
        if ctype in ("image", "diagram", "figure"):
            # Extract both numerical data and analysis together
            analysis = ""
            
            # Parse content with a focus on preserving numerical data
            if "Analysis:" in text:
                analysis_part = text.split("Analysis:", 1)[1].split("OCR (if any):", 1)[0].strip()
                analysis = analysis_part
            else:
                analysis = text
            
            # Add integrated context that preserves numerical data
            context_lines.append(f"[Page {page}] {ctype.upper()}: {analysis}")
        else:
            context_lines.append(f"[Page {page}] {ctype.upper()}: {text}")
    
    context_block = "\n\n".join(context_lines)

    prompt = (
        f"{system_instruction}\n\n"
        f"Document Excerpts:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer (with page citations, organized with clear headings and structure):"
    )

    # 3) Call Gemini's generate_content
    try:
        response = _GEMINI.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=10000,
            ),
        )
        return response.text.strip()
    except Exception as exc:
        # Surface the exception so you see it in Streamlit, but keep the app alive
        return f"*(Gemini error: {exc})*"
