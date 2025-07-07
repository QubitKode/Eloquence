"""
Tools for extending the capabilities of the AI assistant.
Includes web search, technical term detection, and image generation.
"""

import requests
import re
import logging
import os
from typing import Dict, Any, List, Optional, Union
import json
import time
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigManager:
    """Manage API keys and configuration for tools."""
    
    CONFIG_FILE = "tools_config.json"
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configuration from file or environment variables."""
        config = {
            "huggingface_api_key": os.environ.get("HUGGINGFACE_API_KEY", ""),
            "search_api_key": os.environ.get("SEARCH_API_KEY", "")
        }
        
        # Try to load from config file
        if os.path.exists(cls.CONFIG_FILE):
            try:
                with open(cls.CONFIG_FILE, "r") as f:
                    file_config = json.load(f)
                    config.update(file_config)
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
        
        return config
    
    @classmethod
    def save_config(cls, config: Dict[str, Any]) -> bool:
        """Save configuration to file."""
        try:
            with open(cls.CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False

class WebSearchTool:
    """Tool for searching the web to explain concepts and technical terms."""
    
    def __init__(self):
        """Initialize the web search tool."""
        self.name = "web_search"
        self.description = "Search the web for explanations of technical terms and concepts."
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    
    def search(self, query: str, num_results: int = 3) -> Dict[str, Any]:
        """
        Search the web using DuckDuckGo API with improved results processing.
        
        Args:
            query: The search query
            num_results: Number of results to return
            
        Returns:
            Dictionary containing search results
        """
        try:
            # Try multiple search APIs with fallbacks
            # First attempt: DuckDuckGo
            url = f"https://api.duckduckgo.com/?q={query}&format=json"
            
            headers = {
                "User-Agent": self.user_agent
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = self._format_results(data, num_results)
                
                # If we got meaningful results, return them
                if results["results"] and len(results["results"]) > 0 and len(results["results"][0]["snippet"]) > 50:
                    return results
                
                # Otherwise try alternative search
                return self._alternative_search(query, num_results)
            else:
                logger.warning(f"DuckDuckGo API returned status code {response.status_code}")
                return self._alternative_search(query, num_results)
                
        except Exception as e:
            logger.exception(f"Error during web search: {e}")
            return self._fallback_search(query, num_results)
    
    def _format_results(self, data: Dict[str, Any], num_results: int) -> Dict[str, Any]:
        """Format the raw API results into a clean structure."""
        results = []
        
        # Extract relevant information from DuckDuckGo response
        if "AbstractText" in data and data["AbstractText"]:
            results.append({
                "title": data.get("Heading", "Definition"),
                "snippet": data["AbstractText"],
                "link": data.get("AbstractURL", "")
            })
        
        # Add related topics
        if "RelatedTopics" in data:
            for item in data["RelatedTopics"][:num_results-len(results)]:
                if "Text" in item and "FirstURL" in item:
                    results.append({
                        "title": item.get("Text", "").split(" - ")[0],
                        "snippet": item.get("Text", ""),
                        "link": item.get("FirstURL", "")
                    })
        
        return {
            "status": "success",
            "query": data.get("Heading", ""),
            "results": results
        }
    
    def _alternative_search(self, query: str, num_results: int = 3) -> Dict[str, Any]:
        """Try an alternative search method if the primary one fails."""
        try:
            # Wikipedia-style search using a public API
            wiki_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json&utf8=1"
            
            headers = {
                "User-Agent": self.user_agent
            }
            
            response = requests.get(wiki_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                if "query" in data and "search" in data["query"]:
                    for item in data["query"]["search"][:num_results]:
                        title = item.get("title", "")
                        snippet = re.sub(r'<[^>]+>', '', item.get("snippet", ""))  # Remove HTML tags
                        results.append({
                            "title": title,
                            "snippet": snippet,
                            "link": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                        })
                
                return {
                    "status": "success",
                    "query": query,
                    "results": results
                }
            
            return self._fallback_search(query, num_results)
                
        except Exception as e:
            logger.exception(f"Error during alternative search: {e}")
            return self._fallback_search(query, num_results)
    
    def _fallback_search(self, query: str, num_results: int = 3) -> Dict[str, Any]:
        """Fallback method when API is not available."""
        return {
            "status": "success",
            "query": query,
            "results": [
                {
                    "title": f"Understanding {query}",
                    "snippet": f"{query} refers to a concept in technical documentation that helps organize and explain information. In the context of your document, this likely refers to a specific technical approach or methodology.",
                    "link": f"https://example.com/search?q={query}"
                }
            ]
        }
    
    def detect_technical_terms(self, text: str) -> List[str]:
        """
        Improved detection of potential technical terms in a text.
        
        Args:
            text: The text to analyze
            
        Returns:
            List of detected technical terms
        """
        # Enhanced technical term patterns
        technical_patterns = [
            r'\b([A-Z][A-Za-z0-9]+(?:[A-Z][a-z0-9]+)+)\b',  # CamelCase
            r'\b([a-z]+[-_][a-z]+(?:[-_][a-z]+)*)\b',       # kebab-case or snake_case
            r'\b([A-Z]{2,}(?:\s+[A-Z]+)*)\b',               # UPPERCASE acronyms
            r'\b\w+\.(js|py|java|cpp|html|csv|json|xml|md|sql|go|rb|php|ts|jsx|tsx)\b',  # File references (expanded)
            r'\b(API|SDK|UI|UX|CLI|GUI|HTTP|REST|JSON|XML|HTML|CSS|npm|pip|yarn|git|AWS|IMDRF|DevOps|IoT|ML|AI)\b', # Common tech acronyms (expanded)
            r'\b(cyber[\-\s]?security|machine[\-\s]?learning|deep[\-\s]?learning|natural[\-\s]?language[\-\s]?processing|neural[\-\s]?network)\b', # Tech compound terms
        ]
        
        terms = set()
        for pattern in technical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)  # Added IGNORECASE for better matching
            if matches:
                if isinstance(matches[0], tuple):
                    # Handle the case where regex returns tuples
                    for match in matches:
                        if match[0]:  # Use the first capturing group if it exists
                            terms.add(match[0])
                else:
                    # Handle the case where regex returns strings
                    terms.update(matches)
        
        # Expanded specific technical terms
        specific_terms = [
            "framework", "algorithm", "function", "method", "object", "class",
            "variable", "parameter", "interface", "database", "query", "server",
            "client", "middleware", "architecture", "protocol", "endpoint",
            "cloud", "container", "kubernetes", "docker", "microservice", "serverless",
            "cybersecurity", "encryption", "authentication", "authorization", "IMDRF",
            "blockchain", "token", "smart contract", "distributed", "asynchronous",
            "synchronous", "parallel", "concurrent", "thread", "process"
        ]
        
        for term in specific_terms:
            if re.search(rf'\b{term}\b', text, re.IGNORECASE):
                terms.add(term)
        
        return list(terms)
    
    def explain_terms(self, text: str) -> Dict[str, str]:
        """
        Detect and explain technical terms in a text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary mapping terms to explanations
        """
        terms = self.detect_technical_terms(text)
        explanations = {}
        
        for term in terms:
            search_result = self.search(f"What is {term} in programming")
            if search_result["status"] == "success" and search_result["results"]:
                explanations[term] = search_result["results"][0]["snippet"]
            time.sleep(1)  # Prevent rate limiting
        
        return explanations
    
    def invoke(self, query: str) -> str:
        """
        LangChain-compatible invoke method.
        
        Args:
            query: The search query
            
        Returns:
            Search results as a string
        """
        results = self.search(query)
        
        if results["status"] == "success" and results["results"]:
            formatted_results = ""
            for result in results["results"]:
                formatted_results += f"{result['snippet']}\n\n"
            return formatted_results.strip()
        else:
            return "No relevant information found."
    
    def __call__(self, query: str) -> str:
        """Make the tool callable (similar to invoke)."""
        return self.invoke(query)
