import os
import base64
import pymupdf as fitz
import numpy as np
from PIL import Image
import concurrent.futures
import functools
import time
import matplotlib.pyplot as plt
import cv2
from io import BytesIO
import logging
import pandas as pd
from config import TEMP_DIR, GENAI_API_KEY, GENAI_MODEL
import google.generativeai as genai
from paddleocr import PaddleOCR
import warnings
import re

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Gemini API
if not GENAI_API_KEY:
    raise ValueError("Add your Gemini key to config.GENAI_API_KEY")
genai.configure(api_key=GENAI_API_KEY)
_GEMINI = genai.GenerativeModel(GENAI_MODEL)

# Ensure temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

# Lazy-loaded image captioning model
_blip_processor = _blip_model = None

def get_blip_model():
    """Lazy-load BLIP model only when needed"""
    global _blip_processor, _blip_model
    if (_blip_processor is None or _blip_model is None):
        try:
            from transformers import AutoProcessor, BlipForConditionalGeneration
            _blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            _blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            logger.info("BLIP model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load BLIP model: {e}")
    return _blip_processor, _blip_model

def describe_image(image_path):
    """Generate a descriptive caption for an image using BLIP model (if available)."""
    processor, model = get_blip_model()
    if processor is None or model is None:
        return ""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        logger.warning(f"Image captioning failed: {e}")
        return ""

# Initialize PaddleOCR (lazy loading)
_paddle_ocr = None

def get_paddle_ocr():
    """Enhanced PaddleOCR with more optimization options"""
    global _paddle_ocr
    if _paddle_ocr is None:
        try:
            # More optimization options
            _paddle_ocr = PaddleOCR(
                use_angle_cls=False, 
                lang='en', 
                use_gpu=False,
                rec_batch_num=6,           # Batch size for recognition
                rec_model_dir=None,        # Use default model
                det_model_dir=None,        # Use default model
                show_log=False,            # Disable verbose logging
                use_mp=True                # Enable multiprocessing
            )
            logger.info("PaddleOCR model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load PaddleOCR model: {e}")
    return _paddle_ocr

def smart_ocr(image_path, min_size=10000, confidence_threshold=0.7, handwritten_only=False):
    """
    Perform OCR using PaddleOCR with smart filtering
    If handwritten_only is True, only perform OCR if the image appears to contain handwritten content
    Returns extracted text or empty string if no text detected
    """
    try:
        img = Image.open(image_path)
        
        # Skip tiny images
        if img.width * img.height < min_size:
            return ""
            
        # Convert to numpy array for analysis
        img_array = np.array(img.convert('L'))
        
        # Check image variance/contrast (low variance often means no text)
        std_dev = np.std(img_array)
        if std_dev < 20:
            return ""
        
        # If handwritten_only flag is set, try to detect if content is handwritten
        if handwritten_only:
            # Simple heuristic: handwritten content often has higher local variance but lower global structure
            edges = cv2.Canny(img_array, 50, 150)
            edge_density = np.sum(edges > 0) / (img.width * img.height)
            
            # Check for characteristics of machine-printed vs handwritten text
            is_likely_printed = edge_density < 0.05 or std_dev > 60
            
            if is_likely_printed:
                # Skip OCR for what appears to be machine-printed content when handwritten_only is True
                return ""
        
        # Get PaddleOCR instance
        ocr = get_paddle_ocr()
        if ocr is None:
            return ""  # Model failed to load
            
        # Run OCR
        result = ocr.ocr(image_path, cls=False)
        
        if not result or len(result) == 0 or result[0] is None:
            return ""
        
        # Filter by confidence and extract text
        extracted_text = []
        for line in result[0]:
            if len(line) >= 2:
                bbox, (text, confidence) = line
                
                if confidence > confidence_threshold and len(text.strip()) > 0:
                    extracted_text.append(text.strip())
        
        return "\n".join(extracted_text)
    except Exception as e:
        logger.warning(f"OCR error: {e}")
        return ""

@functools.lru_cache(maxsize=32)
def cached_analyze_diagram(diagram_path):
    """Cached version of diagram analysis to avoid duplicate API calls"""
    return analyze_diagram(diagram_path)

def analyze_diagram(diagram_path: str) -> str:
    """
    Use Gemini to analyze the cropped diagram.
    This function encodes the image as base64 and sends it as input to Gemini.
    Returns an analysis string.
    """
    try:
        # Read and encode the image as base64
        with open(diagram_path, "rb") as img_file:
            img_bytes = img_file.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        
        # Gemini expects image input as a dict with mime type and data
        gemini_image = {
            "mime_type": "image/png",
            "data": img_b64
        }
        prompt = (
            "Analyze this chart/diagram and provide a single, integrated explanation. "
            "If it contains numerical data, incorporate ALL exact numbers and percentages "
            "directly into your narrative. DO NOT list numbers separately. "
            "Begin your analysis with a clear description of what the chart shows. "
            "Keep your explanation concise and make sure all numbers are accurately contextualized."
            "Keep everything in a structured format."
        )
        response = _GEMINI.generate_content(
            [
                {"role": "user", "parts": [
                    {"text": prompt},
                    {"inline_data": gemini_image}
                ]}
            ],
            generation_config={"temperature": 0.1, "max_output_tokens": 5000}
        )
        analysis = response.text.strip() if response and hasattr(response, "text") else ""
        return analysis or "No analysis available."
    except Exception as e:
        logger.error(f"Error in diagram analysis: {e}")
        return f"Error in diagram analysis: {e}"

@functools.lru_cache(maxsize=32)
def cached_analyze_visual_element(element_path, element_type):
    """Cached version of visual element analysis to avoid duplicate API calls"""
    return analyze_visual_element(element_path, element_type)

def analyze_visual_element(element_path: str, element_type: str) -> str:
    """
    Use Gemini to analyze any visual element (image, table, or diagram).
    This function encodes the image as base64 and sends it as input to Gemini.
    Returns an analysis string.
    """
    try:
        # Read and encode the image as base64
        with open(element_path, "rb") as img_file:
            img_bytes = img_file.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        
        # Gemini expects image input as a dict with mime type and data
        gemini_image = {
            "mime_type": "image/png",
            "data": img_b64
        }
        
        # Customize prompt based on element type
        if element_type == "diagram":
            prompt = (
                "Analyze this chart/diagram and provide a single, integrated explanation. "
                "If it contains numerical data, incorporate ALL exact numbers and percentages "
                "directly into your narrative. DO NOT list numbers separately. "
                "Begin your analysis with a clear description of what the chart shows. "
                "Keep your explanation concise and make sure all numbers are accurately contextualized."
                "Keep everything in a structured format."
            )
        elif element_type == "table":
            prompt = (
                "Analyze this table and extract its key information. "
                "First describe what the table shows. "
                "Then extract the data in a structured format. "
                "Include ALL numbers, dates, and categories exactly as they appear. "
                "Maintain the relationships between rows and columns in your explanation. "
                "Be concise but complete."
            )
        else:  # image
            prompt = (
                "Describe this image in detail. What does it show? "
                "If there are people, objects, or text visible, describe them. "
                "If it appears to be a figure from a document, describe what it's illustrating. "
                "Be specific and concise."
            )
        
        response = _GEMINI.generate_content(
            [
                {"role": "user", "parts": [
                    {"text": prompt},
                    {"inline_data": gemini_image}
                ]}
            ],
            generation_config={"temperature": 0.1, "max_output_tokens": 5000}
        )
        analysis = response.text.strip() if response and hasattr(response, "text") else ""
        return analysis or f"No analysis available for this {element_type}."
    except Exception as e:
        logger.error(f"Error in {element_type} analysis: {e}")
        return f"Error in {element_type} analysis: {e}"

def process_diagrams_sequentially(diagram_paths):
    """Process diagrams one by one to avoid API limits, with improved throttling"""
    results = {}
    
    logger.info(f"Processing {len(diagram_paths)} diagrams sequentially...")
    
    # Calculate adaptive delay based on number of diagrams
    # More diagrams -> longer delay to avoid rate limits
    base_delay = 0.5
    adaptive_delay = min(5.0, max(0.5, 0.5 * len(diagram_paths) / 10))
    
    for i, path in enumerate(diagram_paths):
        try:
            # Check if we already have a cached result
            if hasattr(cached_analyze_diagram, 'cache_info'):
                cache_info = cached_analyze_diagram.cache_info()
                logger.info(f"Cache info: {cache_info}")
            
            # Use cached result if available
            results[path] = cached_analyze_diagram(path)
            logger.info(f"Processed diagram {i+1}/{len(diagram_paths)}: {os.path.basename(path)}")
            
            # Add a delay between API calls to avoid rate limits
            if i < len(diagram_paths) - 1:
                time.sleep(adaptive_delay)
                
                # Add a longer pause after every 5 calls to avoid quota limits
                if (i + 1) % 5 == 0:
                    logger.info(f"Taking a longer pause after processing {i+1} diagrams...")
                    time.sleep(adaptive_delay * 3)
                
        except Exception as e:
            results[path] = f"Error analyzing diagram: {e}"
            logger.error(f"Failed to analyze diagram {path}: {e}")
            
            # On error, take a longer pause to allow API to recover
            time.sleep(adaptive_delay * 2)
    
    return results

def detect_image_type(img_path):
    """
    Detect if an image is a diagram, table, or regular image
    Returns: "table", "diagram", or "image"
    """
    try:
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            return "image"
        
        # Get image properties
        height, width = img.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Edge detection for table/diagram detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (width * height)
        
        # Line detection
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 
            threshold=100, 
            minLineLength=50, 
            maxLineGap=10
        )
        
        # Table detection - count horizontal and vertical lines
        if lines is not None and len(lines) > 5:
            h_lines = 0
            v_lines = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 10:  # Horizontal line
                    h_lines += 1
                if abs(x2 - x1) < 10:  # Vertical line
                    v_lines += 1
            
            # If there are multiple horizontal and vertical lines, likely a table
            if h_lines >= 3 and v_lines >= 3:
                return "table"
        
        # Diagram detection
        # Count circles
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=10, maxRadius=100
        )
        
        # Diagram criteria
        has_many_lines = lines is not None and len(lines) > 10
        has_circles = circles is not None
        
        if has_many_lines or has_circles or edge_density > 0.1:
            return "diagram"
        
        return "image"
    except Exception as e:
        logger.warning(f"Image type detection failed: {e}")
        return "image"

def detect_table_areas(page_image):
    """
    Detect table regions in a page image using OpenCV with high precision.
    Uses adaptive thresholding and grid structure checks to avoid images/diagrams.
    Returns a list of (x, y, w, h) tuples.
    """
    # Convert to grayscale if needed
    if len(page_image.shape) == 3:
        gray = cv2.cvtColor(page_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = page_image

    # Adaptive thresholding for better separation
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )

    # Morphological operations to enhance lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    grid = cv2.add(horizontal_lines, vertical_lines)

    # Find contours on the grid image
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    table_regions = []
    min_area = 1200  # Lowered for small tables

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0

        # Filter out very wide/tall regions (likely not tables)
        if not (0.3 < aspect_ratio < 10):
            continue

        roi = grid[y:y+h, x:x+w]
        if roi.size == 0 or min(roi.shape) < 10:
            continue

        # Count horizontal and vertical lines in ROI
        hor_proj = np.sum(roi, axis=0) // 255
        ver_proj = np.sum(roi, axis=1) // 255
        h_lines = np.sum(hor_proj > (0.5 * h))
        v_lines = np.sum(ver_proj > (0.5 * w))

        # Require at least 2 horizontal and 2 vertical lines (grid structure)
        if h_lines >= 2 and v_lines >= 2:
            table_regions.append((x, y, w, h))

    return table_regions

def extract_tables_from_page_direct(doc, page, page_num, contents, page_content_map, processed_regions, visual_elements):
    """Extract tables by directly cropping from the PDF, avoiding duplication and strictly tables only."""
    tables_found = []
    try:
        zoom_factor = 1.3
        mat = fitz.Matrix(zoom_factor, zoom_factor)
        pix = page.get_pixmap(matrix=mat)
        page_width = page.rect.width
        page_height = page.rect.height
        page_area = page_width * page_height

        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        table_regions = detect_table_areas(img)

        for i, (x, y, w, h) in enumerate(table_regions):
            x0 = x / zoom_factor
            y0 = y / zoom_factor
            x1 = (x + w) / zoom_factor
            y1 = (y + h) / zoom_factor

            table_area = (x1 - x0) * (y1 - y0)
            table_percentage = (table_area / page_area) * 100

            if is_region_processed(page_num, [x0, y0, x1, y1]):
                continue

            register_processed_region(page_num, [x0, y0, x1, y1])

            # Always crop the detected table region (never the full page unless it's >90%)
            if table_percentage > 90:
                table_rect = fitz.Rect(0, 0, page_width, page_height)
                table_img_path = os.path.join(TEMP_DIR, f"page{page_num}_full_page_table{i}.png")
            else:
                table_rect = fitz.Rect(x0, y0, x1, y1)
                table_img_path = os.path.join(TEMP_DIR, f"page{page_num}_table{i}_direct.png")

            table_pix = page.get_pixmap(matrix=mat, clip=table_rect)
            table_pix.save(table_img_path)

            # Add to visual elements for analysis
            if visual_elements is not None:
                visual_elements.append((table_img_path, "table"))

            table_text = smart_ocr(table_img_path)

            table_obj = {
                "page": page_num,
                "type": "table",
                "ocr_text": table_text,
                "path": table_img_path,
                "coords": [x0, y0, x1, y1] if table_percentage <= 70 else [0, 0, page_width, page_height],
                "parent_page": page_num,
                "sequence": len(tables_found),
                "is_full_page": table_percentage > 70
            }

            tables_found.append(table_obj)
            if "tables" not in page_content_map[page_num]:
                page_content_map[page_num]["tables"] = []
            page_content_map[page_num]["tables"].append(table_obj)

    except Exception as e:
        logger.warning(f"Direct table extraction failed for page {page_num}: {str(e)}")

    return tables_found

def extract_tables_in_parallel_direct(doc, num_pages, contents, page_content_map, processed_regions, visual_elements=None, max_workers=2):
    """Extract tables from all pages in parallel using direct cropping"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_page = {
            executor.submit(
                extract_tables_from_page_direct, 
                doc, doc[page_num-1], page_num, 
                contents, page_content_map, processed_regions, visual_elements
            ): page_num
            for page_num in range(1, num_pages + 1)
        }
        
        for future in concurrent.futures.as_completed(future_to_page):
            page_num = future_to_page[future]
            try:
                future.result()  # We don't need the result as it's added directly to contents
            except Exception as e:
                logger.error(f"Table extraction failed for page {page_num}: {str(e)}")

def detect_and_map_relations(contents, page_content_map):
    """Detect relationships between text and visual elements with improved accuracy"""
    # Gather text chunks by page for relation mapping
    text_by_page = {}
    # Track section numbers and their pages
    section_map = {}
    
    # First pass - identify section numbers and their locations
    for chunk in contents:
        if chunk["type"] == "text":
            page = chunk["page"]
            content = chunk["content"]
            
            # Look for section numbers (e.g., 1.3, 2.4.1)
            section_matches = re.findall(r'(\d+\.\d+(?:\.\d+)*)[.\s]', content)
            for section in section_matches:
                if section not in section_map:
                    section_map[section] = []
                section_map[section].append(page)
            
            if page not in text_by_page:
                text_by_page[page] = []
            text_by_page[page].append(chunk)
    
    # Track elements that have been processed to avoid duplicates
    processed_elements = set()
    
    # For each visual element, find the most relevant text chunks
    for page_num, page_data in page_content_map.items():
        for element_type in ["images", "tables", "diagrams"]:
            if element_type not in page_data:
                continue
                
            for element in page_data[element_type]:
                # Skip if this element has already been processed
                element_id = f"{element_type}_{element.get('path', '')}_{element.get('sequence', '')}"
                if element_id in processed_elements:
                    continue
                
                processed_elements.add(element_id)
                element["related_text"] = []
                element["related_sections"] = []  # Track related section numbers
                
                # Check if we have coordinates for proximity-based matching
                element_coords = element.get("coords", None)
                
                # Find text chunks from the same page
                if page_num in text_by_page:
                    relevant_chunks = []
                    
                    for text_chunk in text_by_page[page_num]:
                        text_content = text_chunk["content"].lower()
                        relevance_score = 0
                        
                        # Score based on keyword matches
                        if element_type == "tables" and any(kw in text_content for kw in ["table", "data", "statistics", "figures"]):
                            relevance_score += 5
                        elif element_type == "images" and any(kw in text_content for kw in ["figure", "image", "photo", "picture", "illustration"]):
                            relevance_score += 5
                        elif element_type == "diagrams" and any(kw in text_content for kw in ["diagram", "chart", "graph", "plot", "visualization"]):
                            relevance_score += 5
                        
                        # Look for number references that might indicate a figure/table number
                        if re.search(r'(figure|fig\.|table|chart)\s*\d+', text_content, re.IGNORECASE):
                            relevance_score += 3
                        
                        # Look for section numbers in the text
                        section_matches = re.findall(r'(\d+\.\d+(?:\.\d+)*)[.\s]', text_chunk["content"])
                        if section_matches:
                            for section in section_matches:
                                element["related_sections"].append(section)
                            relevance_score += 4
                        
                        # Check if the text mentions something contained in the element's OCR text
                        element_text = element.get("content", "").lower()
                        if element_text:
                            # Extract significant words from the element text
                            element_words = set(re.findall(r'\b\w+\b', element_text))
                            text_words = set(re.findall(r'\b\w+\b', text_content))
                            # Check word overlap
                            common_words = element_words.intersection(text_words)
                            if len(common_words) >= 2:  # If they share at least 2 significant words
                                relevance_score += len(common_words)
                        
                        # Proximity score - if we have coordinates for both the element and text
                        text_coords = text_chunk.get("coords", None)
                        if element_coords and text_coords:
                            # Calculate vertical distance (text above or below the element)
                            vert_distance = min(
                                abs(text_coords[3] - element_coords[1]),  # text bottom to element top
                                abs(text_coords[1] - element_coords[3])   # text top to element bottom
                            )
                            # Text close to the element gets higher score
                            if vert_distance < 50:  # Close proximity
                                relevance_score += 10
                            elif vert_distance < 100:  # Medium proximity
                                relevance_score += 5
                        
                        if relevance_score > 0:
                            relevant_chunks.append((text_chunk, relevance_score))
                    
                    # Sort chunks by relevance score and take top 3
                    relevant_chunks.sort(key=lambda x: x[1], reverse=True)
                    element["related_text"] = [chunk["content"] for chunk, score in relevant_chunks[:3]]
                
                # Also check the section map to see if this visual element is on a page with known sections
                for section, pages in section_map.items():
                    if page_num in pages and section not in element["related_sections"]:
                        element["related_sections"].append(section)

    # Create a mapping of figure/table references to actual elements
    visual_reference_map = {}

    # First pass - extract figure/table numbers from text
    for chunk in contents:
        if chunk["type"] == "text":
            content = chunk["content"]
            # Look for figure/table references with numbers
            fig_matches = re.findall(r'(figure|fig\.|table|chart|diagram)\s*(\d+(?:\.\d+)?)', content, re.IGNORECASE)
            for match_type, match_num in fig_matches:
                ref_key = f"{match_type.lower()}{match_num}"
                if ref_key not in visual_reference_map:
                    visual_reference_map[ref_key] = []
                visual_reference_map[ref_key].append(chunk["page"])

    # After the nested loops, add this code to map elements to their reference numbers
    for page_num, page_data in page_content_map.items():
        for element_type in ["images", "tables", "diagrams"]:
            if element_type not in page_data:
                continue
                
            for element in page_data[element_type]:
                element["reference_numbers"] = []
                
                # Check if any of the visual references match this element's page
                for ref_key, ref_pages in visual_reference_map.items():
                    if page_num in ref_pages:
                        # Also check proximity and other factors if you want more certainty
                        element["reference_numbers"].append(ref_key)
    
    return contents

# Global list to track processed regions
processed_regions = []

def is_region_processed(page_num, bbox, threshold=0.5):
    """Check if a region on a page has already been processed"""
    for proc_page, proc_bbox in processed_regions:
        if proc_page == page_num and has_significant_overlap(bbox, proc_bbox, threshold):
            return True
    return False

# Add this function to register a processed region
def register_processed_region(page_num, bbox):
    """Register a region as processed to avoid duplication"""
    processed_regions.append((page_num, bbox))

def parse_pdf(pdf_file) -> list:
    """
    Parse a PDF file (path or file-like) and extract text, tables, images, diagrams.
    Returns a list of content chunks with improved relationships.
    """
    start_time = time.time()
    contents = []
    
    # Clean temp directory
    for f in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, f)
        if os.path.isfile(file_path):
            os.unlink(file_path)
    
    # Initialize processed regions tracker
    processed_regions = []
    
    # Open the PDF
    if isinstance(pdf_file, (str, bytes, os.PathLike)):
        doc = fitz.open(pdf_file)
        pdf_path = pdf_file if isinstance(pdf_file, str) else None
    else:
        file_bytes = pdf_file.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pdf_path = None
    
    # If we don't have a direct file path, save a temporary copy
    if pdf_path is None:
        pdf_path = os.path.join(TEMP_DIR, "uploaded_doc.pdf")
        with open(pdf_path, "wb") as f:
            f.write(doc.write())
    
    num_pages = doc.page_count
    logger.info(f"Processing PDF with {num_pages} pages")
    
    # Higher resolution for image operations
    zoom_factor = 2.0
    mat = fitz.Matrix(zoom_factor, zoom_factor)
    
    # Structure to track page content hierarchically
    page_content_map = {}
    
    # Collect all visual elements for analysis
    visual_elements = []  # Will store tuples of (path, element_type)
    
    # First pass: Extract text and images (but NOT drawings yet)
    for page_number in range(num_pages):
        page = doc.load_page(page_number)
        page_index = page_number + 1
        
        logger.info(f"Processing page {page_index}/{num_pages}")
        
        # Initialize page in content map
        page_content_map[page_index] = {
            "text": [],
            "images": [],
            "tables": [],
            "diagrams": []
        }
        
        # Extract text
        text = page.get_text("text")
        if text:
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            for para in paragraphs:
                if len(para) < 5:
                    continue
                chunk = {
                    "page": page_index,
                    "type": "text",
                    "content": para,
                    "parent_page": page_index,
                    "section": "body",
                    "sequence": len(page_content_map[page_index]["text"])
                }
                contents.append(chunk)
                page_content_map[page_index]["text"].append(chunk)
        
        # Extract images using PyMuPDF - now with parallel processing
        image_list = page.get_images(full=True)
        if image_list:
            image_results = process_images_in_parallel(
                image_list, page, page_index, contents, 
                page_content_map, visual_elements, processed_regions
            )

    # Second pass: Extract tables directly from the PDF (before drawing extraction)
    logger.info("Extracting tables by direct cropping...")
    extract_tables_in_parallel_direct(doc, num_pages, contents, page_content_map, processed_regions, visual_elements)
    
    # Add tables to visual elements for analysis
    for page_num, page_data in page_content_map.items():
        if "tables" in page_data:
            for table in page_data["tables"]:
                visual_elements.append((table["path"], "table"))

    # Third pass: Extract drawings (vector graphics, charts, etc.)
    for page_number in range(num_pages):
        page = doc.load_page(page_number)
        page_index = page_number + 1
        drawings = page.get_drawings()
        if drawings and len(drawings) > 3:  # Threshold to avoid false positives
            # Compute the union of drawing rectangles
            union_rect = None
            for d in drawings:
                rect = fitz.Rect(d["rect"])
                if union_rect is None:
                    union_rect = rect
                else:
                    union_rect = union_rect.include_rect(rect)
            
            if union_rect:
                # Convert to list coordinates
                bbox = [union_rect.x0, union_rect.y0, union_rect.x1, union_rect.y1]
                
                # Check if this region overlaps with any previously processed element (skip if so)
                overlap_with_existing = False
                # Check tables
                for t in page_content_map[page_index]["tables"]:
                    if has_significant_overlap(bbox, t["coords"], threshold=0.3):
                        overlap_with_existing = True
                        break
                # Check images if no overlap with tables found
                if not overlap_with_existing and "images" in page_content_map[page_index]:
                    for img in page_content_map[page_index]["images"]:
                        if "coords" in img and has_significant_overlap(bbox, img["coords"], threshold=0.3):
                            overlap_with_existing = True
                            break
                if overlap_with_existing:
                    continue  # Skip this drawing region, it's already processed
                
                # Register this region as processed
                register_processed_region(page_index, bbox)
                
                # Inflate slightly to capture surrounding elements
                union_rect.x0 = max(0, union_rect.x0 - 5)
                union_rect.y0 = max(0, union_rect.y0 - 5)
                union_rect.x1 += 5
                union_rect.y1 += 5
                
                # Crop the diagram region
                diag_pix = page.get_pixmap(matrix=mat, clip=union_rect)
                diag_img_path = os.path.join(TEMP_DIR, f"page{page_index}_drawing.png")
                diag_pix.save(diag_img_path)
                
                # Add to visual elements collection for batch processing
                visual_elements.append((diag_img_path, "diagram"))
                ocr_text = smart_ocr(diag_img_path)
                
                diagram_obj = {
                    "page": page_index,
                    "type": "diagram",
                    "path": diag_img_path,
                    "ocr_text": ocr_text,
                    "coords": [union_rect.x0, union_rect.y0, union_rect.x1, union_rect.y1],
                    "parent_page": page_index,
                    "sequence": len(page_content_map[page_index]["diagrams"])
                }
                page_content_map[page_index]["diagrams"].append(diagram_obj)

    # Process all visual elements sequentially to avoid API limits
    if visual_elements:
        logger.info(f"Processing {len(visual_elements)} visual elements...")
        element_analyses = process_visual_elements_sequentially(visual_elements)
        
        # Add the processed elements to contents
        for page_num, page_data in page_content_map.items():
            # Update images
            if "images" in page_data:
                for image in page_data["images"]:
                    path = image["path"]
                    if path in element_analyses:
                        analysis = element_analyses[path]
                        content_text = (
                            f"Image on page {image['page']}:\n"
                            f"Analysis: {analysis}\n"
                            f"OCR text: {image.get('ocr_text', '')}"
                        )
                        image["content"] = content_text
                        image["ai_analysis"] = analysis  # Store AI analysis separately
                        if image not in contents:
                            contents.append(image)
            
            # Update tables
            if "tables" in page_data:
                for table in page_data["tables"]:
                    path = table["path"]
                    if path in element_analyses:
                        analysis = element_analyses[path]
                        content_text = (
                            f"Table on page {table['page']}:\n"
                            f"Analysis: {analysis}\n"
                            f"OCR text: {table.get('ocr_text', '')}"
                        )
                        table["content"] = content_text
                        table["ai_analysis"] = analysis  # Store AI analysis separately
                        if table not in contents:
                            contents.append(table)
            
            # Update diagrams
            if "diagrams" in page_data:
                for diagram in page_data["diagrams"]:
                    path = diagram["path"]
                    if path in element_analyses:
                        analysis = element_analyses[path]
                        content_text = (
                            f"Diagram on page {diagram['page']}:\n"
                            f"Analysis: {analysis}\n"
                            f"OCR text: {diagram.get('ocr_text', '')}"
                        )
                        diagram["content"] = content_text
                        diagram["ai_analysis"] = analysis  # Store AI analysis separately
                        if diagram not in contents:
                            contents.append(diagram)
    
    # Map relations between text and visual elements with improved algorithm
    logger.info("Mapping relationships between elements...")
    contents = detect_and_map_relations(contents, page_content_map)
    
    # Close the document
    doc.close()
    
    end_time = time.time()
    logger.info(f"PDF processing completed in {end_time - start_time:.2f} seconds")
    
    return contents

def has_significant_overlap(rect1, rect2, threshold=0.5):
    """
    Check if two rectangles have significant overlap.
    rect1, rect2: [x0, y0, x1, y1] coordinates
    threshold: minimum overlap ratio to consider significant
    """
    # Calculate intersection
    x0 = max(rect1[0], rect2[0])
    y0 = max(rect1[1], rect2[1])
    x1 = min(rect1[2], rect2[2])
    y1 = min(rect1[3], rect2[3])
    
    if x0 >= x1 or y0 >= y1:
        return False  # No intersection
    
    intersection_area = (x1 - x0) * (y1 - y0)
    
    # Calculate areas of both rectangles
    area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    area2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
    
    # Calculate overlap ratio relative to the smaller rectangle
    smaller_area = min(area1, area2)
    overlap_ratio = intersection_area / smaller_area if smaller_area > 0 else 0
    
    return overlap_ratio >= threshold

def process_images_in_parallel(image_list, page, page_index, contents, page_content_map, visual_elements, processed_regions, max_workers=4):
    """Process images in parallel to speed up extraction"""
    
    # Get document from the page
    doc = page.parent
    
    def process_single_image(img_info, img_index):
        try:
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Save the image
            img_ext = base_image["ext"]
            img_name = f"page{page_index}_img{img_index}.{img_ext}"
            img_path = os.path.join(TEMP_DIR, img_name)
            
            # Get image bbox
            bbox = page.get_image_bbox(img_info)
            if not bbox:
                return None
                
            # Check if this region has already been processed
            if is_region_processed(page_index, [bbox.x0, bbox.y0, bbox.x1, bbox.y1]):
                return None
                
            # Register this region as processed
            register_processed_region(page_index, [bbox.x0, bbox.y0, bbox.x1, bbox.y1])
            
            with open(img_path, "wb") as img_file:
                img_file.write(image_bytes)
            
            # Determine image type
            img_type = detect_image_type(img_path)
            
            result = {
                "path": img_path,
                "type": img_type,
                "bbox": [bbox.x0, bbox.y0, bbox.x1, bbox.y1],
                "index": img_index
            }
            
            return result
        except Exception as e:
            logger.warning(f"Error processing image {img_index} on page {page_index}: {e}")
            return None
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_img = {
            executor.submit(process_single_image, img_info, img_index): img_index
            for img_index, img_info in enumerate(image_list)
        }
        
        for future in concurrent.futures.as_completed(future_to_img):
            img_index = future_to_img[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Image processing failed for index {img_index}: {str(e)}")
    
    # Process the results
    for result in results:
        img_path = result["path"]
        img_type = result["type"]
        img_index = result["index"]
        bbox = result["bbox"]
        
        # Get OCR text for all image types
        ocr_text = smart_ocr(img_path)
        
        if img_type == "diagram":
            # Add to visual elements collection for batch processing
            visual_elements.append((img_path, "diagram"))
            
            diagram_obj = {
                "page": page_index,
                "type": "diagram",
                "path": img_path,
                "ocr_text": ocr_text,
                "coords": bbox,
                "parent_page": page_index,
                "sequence": len(page_content_map[page_index]["diagrams"])
            }
            page_content_map[page_index]["diagrams"].append(diagram_obj)
            # Content will be added after analysis
        
        elif img_type == "table":
            # Add to visual elements collection for batch processing
            visual_elements.append((img_path, "table"))
            
            # Create table object (content will be updated after analysis)
            table_obj = {
                "page": page_index,
                "type": "table",
                "path": img_path,
                "ocr_text": ocr_text,
                "coords": bbox,
                "parent_page": page_index,
                "sequence": len(page_content_map[page_index]["tables"])
            }
            page_content_map[page_index]["tables"].append(table_obj)
        
        else:  # Regular image
            # Add to visual elements collection for batch processing
            visual_elements.append((img_path, "image"))
            
            # Create image object (content will be updated after analysis)
            image_obj = {
                "page": page_index,
                "type": "image",
                "path": img_path,
                "ocr_text": ocr_text,
                "coords": bbox,
                "parent_page": page_index,
                "sequence": len(page_content_map[page_index]["images"])
            }
            page_content_map[page_index]["images"].append(image_obj)
    
    return results

def find_table_caption(page, coords):
    """Find caption text above or below a table"""
    try:
        # Simple implementation - look for text near the table coordinates
        table_rect = fitz.Rect(coords)
        # Look above the table
        above_rect = fitz.Rect(table_rect.x0, table_rect.y0-50, table_rect.x1, table_rect.y0)
        above_text = page.get_text("text", clip=above_rect).strip()
        # Look below the table
        below_rect = fitz.Rect(table_rect.x0, table_rect.y1, table_rect.x1, table_rect.y1+50)
        below_text = page.get_text("text", clip=below_rect).strip()
        
        # Return caption if found (prefer above text)
        if "table" in above_text.lower() or "fig" in above_text.lower():
            return above_text
        if "table" in below_text.lower() or "fig" in below_text.lower():
            return below_text
        return ""
    except Exception as e:
        logger.warning(f"Error finding table caption: {e}")
        return ""
        
def is_scanned_page(pdf_path, page_num):
    """Returns True if the page appears to be a scanned document."""
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num-1]
        
        # Get text content - very little text suggests a scanned page
        text = page.get_text("text").strip()
        if len(text) < 100:  # Minimal text on the page
            # Check for images
            image_list = page.get_images(full=True)
            # If few text but has images, likely scanned
            if len(image_list) > 0:
                doc.close()
                return True
        
        # Alternative check: analyze the page for image coverage
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        # Convert to grayscale
        if pix.n == 4:  # RGBA
            gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        else:  # RGB
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Calculate histogram features
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist / hist.sum()
        # Scanned pages often have specific histogram patterns
        # (e.g., peaks at certain gray levels and more uniform distribution)
        hist_std = np.std(hist_norm)
        
        doc.close()
        # Low text content with high image variation suggests a scanned page
        return len(text) < 500 and hist_std < 0.03
    except Exception as e:
        logger.warning(f"Error checking if page {page_num} is scanned: {e}")
        return False  # Default to non-scanned if there's an error

def extract_tables_with_camelot(pdf_path, page_num, contents, page_content_map):
    """Extract tables using Camelot for better performance and text accuracy"""
    import camelot
    tables_found = []
    
    try:
        # Extract tables with Camelot (much faster for text-based PDFs)
        tables = camelot.read_pdf(
            pdf_path, 
            pages=str(page_num),
            flavor='lattice'  # For tables with visible lines
        )
        
        if len(tables) == 0:
            # Try stream mode if lattice fails
            tables = camelot.read_pdf(
                pdf_path, 
                pages=str(page_num),
                flavor='stream'
            )
        
        for i, table in enumerate(tables):
            # Get table as text
            table_text = table.df.to_string(index=False)
            
            # Extract table coordinates
            coords = table._bbox  # [x0, y0, x1, y1]
            
            # Save table as image (for visual display in frontend)
            table_img_path = os.path.join(TEMP_DIR, f"page{page_num}_table{i}_camelot.png")
            
            # Get the table region from PDF
            doc = fitz.open(pdf_path)
            page = doc[page_num-1]
            table_rect = fitz.Rect(coords)
            mat = fitz.Matrix(1.5, 1.5)  # Higher quality for better visibility
            table_pix = page.get_pixmap(matrix=mat, clip=table_rect)
            table_pix.save(table_img_path, output="jpeg", quality=85)  # Use jpeg for smaller file size
            
            # Look for caption (text above or below the table)
            caption = find_table_caption(page, coords)
            
            table_obj = {
                "page": page_num,
                "type": "table",
                "content": f"Table on page {page_num}:\n{table_text}",
                "caption": caption,
                "tabular_data": table.df.to_dict(),  # Store structured data for efficient querying
                "path": table_img_path,
                "coords": coords,
                "parent_page": page_num,
                "sequence": i
            }
            
            tables_found.append(table_obj)
            contents.append(table_obj)
            if "tables" not in page_content_map[page_num]:
                page_content_map[page_num]["tables"] = []
            page_content_map[page_num]["tables"].append(table_obj)
            
            doc.close()
    
    except Exception as e:
        logger.warning(f"Failed to extract tables with Camelot for page {page_num}: {e}")
        return []
    
    return tables_found

def extract_tables_hybrid(pdf_path, num_pages, contents, page_content_map):
    """Extract tables using a hybrid approach (Camelot for text PDFs, OpenCV for scanned)"""
    for page_num in range(1, num_pages + 1):
        # Try Camelot first (faster)
        tables = extract_tables_with_camelot(pdf_path, page_num, contents, page_content_map)
        
        # If no tables found with Camelot, try OpenCV approach for this page
        if not tables and is_scanned_page(pdf_path, page_num):
            doc = fitz.open(pdf_path)
            extract_tables_from_page_direct(doc, doc[page_num-1], page_num, contents, page_content_map, [])
            doc.close()

def process_visual_elements_sequentially(elements):
    """
    Process visual elements (images, tables, diagrams) one by one to avoid API limits.
    
    Args:
        elements: List of tuples (path, element_type)
    
    Returns:
        Dictionary mapping paths to analysis results
    """
    results = {}
    
    logger.info(f"Processing {len(elements)} visual elements sequentially...")
    
    # Calculate adaptive delay based on number of elements
    base_delay = 0.5
    adaptive_delay = min(5.0, max(0.5, 0.5 * len(elements) / 10))
    
    for i, (path, element_type) in enumerate(elements):
        try:
            # Check if we already have a cached result
            if hasattr(cached_analyze_visual_element, 'cache_info'):
                cache_info = cached_analyze_visual_element.cache_info()
                logger.info(f"Cache info: {cache_info}")
            
            # Use cached result if available
            results[path] = cached_analyze_visual_element(path, element_type)
            logger.info(f"Processed {element_type} {i+1}/{len(elements)}: {os.path.basename(path)}")
            
            # Add a delay between API calls to avoid rate limits
            if i < len(elements) - 1:
                time.sleep(adaptive_delay)
                
                # Add a longer pause after every 5 calls to avoid quota limits
                if (i + 1) % 5 == 0:
                    logger.info(f"Taking a longer pause after processing {i+1} elements...")
                    time.sleep(adaptive_delay * 3)
                
        except Exception as e:
            results[path] = f"Error analyzing {element_type}: {e}"
            logger.error(f"Failed to analyze {element_type} {path}: {e}")
            
            # On error, take a longer pause to allow API to recover
            time.sleep(adaptive_delay * 2)
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        print(f"Processing PDF: {pdf_path}")
        
        start_time = time.time()
        contents = parse_pdf(pdf_path)
        end_time = time.time()
        
        duration = end_time - start_time
        num_pages = len(set(chunk["page"] for chunk in contents))
        
        print(f"Processed {num_pages} pages in {duration:.2f} seconds")
        print(f"Average time per page: {duration/num_pages:.2f} seconds")
        print(f"Extracted {len(contents)} content chunks:")
        
        # Count by type
        type_counts = {}
        for chunk in contents:
            ctype = chunk["type"]
            type_counts[ctype] = type_counts.get(ctype, 0) + 1
            
        for ctype, count in type_counts.items():
            print(f"  - {ctype}: {count}")
    else:
        print("Please provide a PDF path as argument")
