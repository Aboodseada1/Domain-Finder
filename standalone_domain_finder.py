#!/usr/bin/env python3
"""
standalone_domain_finder.py

Takes a company name and attempts to find its official website URL using
SearXNG web search and LLM analysis for verification.
"""

import json
import sys
import os
import random
import time
import re
import argparse
import logging
import traceback
import html
from urllib.parse import urlparse, quote_plus
from pathlib import Path

# --- Library Imports ---
import requests
try:
    from termcolor import colored
except ImportError:
    print("Warning: 'termcolor' not installed (pip install termcolor). Colored output will be disabled.")
    # Define a dummy function if termcolor is not available
    def colored(text, *args, **kwargs):
        return text

# Optional dependencies (imported later when needed/checked)
playwright = None # Keep placeholder from ultimate_scraper if needed, but not used here
selenium = None   # Keep placeholder from ultimate_scraper if needed, but not used here
webdriver_manager = None # Keep placeholder from ultimate_scraper if needed, but not used here
genai = None # For Gemini
openai = None # For OpenAI/Groq

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(name)s] %(message)s', stream=sys.stdout)
logger = logging.getLogger("domain_finder")

# --- Configuration ---
# Max search results to feed to LLM
MAX_RESULTS_FOR_LLM = 15
# LLM Model Fallbacks (if specific model isn't provided) - Align with CEO finder
GEMINI_MODEL_FALLBACK = ["gemini-1.5-flash", "gemini-pro"]
OPENAI_MODEL_FALLBACK = ["gpt-4o-mini", "gpt-3.5-turbo"]
GROQ_MODEL_FALLBACK = ["llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]
OLLAMA_DEFAULT_MODEL = "llama3:8b"
MAX_PROMPT_LENGTH = 30000 # Limit prompt length for LLM
MAX_RETRIES = 2 # Retries for LLM calls
INITIAL_RETRY_DELAY = 1
MAX_RETRY_DELAY = 5

# --- Integrated URL Cleaning & Extraction ---
def clean_url(url):
    """Clean a URL: remove escapes, trailing chars, normalize."""
    if not url or not isinstance(url, str):
        return None # Return None instead of "null"

    try:
        # Basic cleaning
        url = url.strip().replace('\\n', '').replace('\\r', '').replace('\\t', '')
        url = re.sub(r'[.,;:\'"\)\]}>]+$', '', url) # Remove trailing punctuation
        while url.endswith('\\'): url = url[:-1]
        while url.endswith('/') and len(url.split('//', 1)[-1]) > 1: url = url[:-1] # Remove trailing slash unless it's just http://

        # Add scheme if missing for parsing
        if not url.startswith(('http://', 'https://', '//')):
            url = 'https://' + url

        parsed = urlparse(url)
        # Reconstruct with just scheme, netloc, path
        # Ensure netloc exists
        if not parsed.netloc:
             logger.debug(f"URL '{url}' has no network location after parsing.")
             return None

        clean = f"{parsed.scheme or 'https'}://{parsed.netloc}{parsed.path or ''}"
        # Final trailing slash removal
        while clean.endswith('/') and len(clean.split('//', 1)[-1]) > 1: clean = clean[:-1]

        # Prevent returning excessively long invalid URLs sometimes caught by regex
        if len(clean) > 1000:
             logger.warning(f"URL seems excessively long after cleaning, might be invalid: {clean[:100]}...")
             # Optional: could try to truncate or just return None
             # Let's try to find the first likely end point (.com, .org etc. + optional path segment)
             match = re.match(r'(https?://[^/]+/[^/?#\s]+)', clean) or re.match(r'(https?://[^/]+)', clean)
             if match:
                 return match.group(1)
             return None # If it's extremely long and unparsable

        return clean
    except Exception as e:
        logger.debug(f"Error parsing/cleaning URL '{url}': {e}")
        return None # Return None on error

def extract_urls(text):
    """Extract valid URLs, filter common non-official domains."""
    if not text or not isinstance(text, str):
        return []

    # Basic text cleaning for regex
    cleaned_text = text.replace('\\n', ' ').replace('\\r', ' ').replace('\\t', ' ')
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    # Regex for URLs (handles common cases)
    url_pattern = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[-\w%/.~!$&\'()*+,;=:@?#]*)?')
    urls_found = url_pattern.findall(cleaned_text)

    exclude_domains = [
        'google.com', 'youtube.com', 'youtu.be', 'facebook.com', 'fb.com',
        'twitter.com', 'x.com', 'instagram.com', 'linkedin.com', 'tiktok.com',
        'pinterest.com', 'snapchat.com', 'reddit.com', 'tumblr.com', 'vimeo.com',
        'dailymotion.com', 'medium.com', 'wordpress.com', 'blogspot.com', 'wix.com',
        'wikipedia.org', 'wikimedia.org', 'amazon.com', 'ebay.com', 'apple.com', # Exclude very large unrelated sites
        'microsoft.com', # Often appears in dev links
        'support.google.com', 'maps.google.com', 'play.google.com', # Google subdomains
        'yelp.com', 'tripadvisor.com', 'trustpilot.com', # Review/directory sites
        'github.com', # Usually not the *company* site unless it's GitHub itself
        'wa.me', 't.me', # Messaging apps
        'maps.apple.com',
        # Add TLDs often associated with spam or non-official sites
        '.info', '.biz', '.tv', '.cc', '.online', '.site', '.xyz', '.icu', '.top', '.loan'
    ]

    filtered_urls = set()
    for url in urls_found:
        cleaned = clean_url(url)
        if not cleaned:
            continue

        try:
            parsed = urlparse(cleaned)
            domain = parsed.netloc.lower() if parsed.netloc else ''
            path = parsed.path or ''

            # Skip if domain is empty or doesn't look like a domain
            if not domain or '.' not in domain:
                 continue

            # Skip if domain is in exclude list or ends with an excluded TLD
            if any(ex_domain in domain for ex_domain in exclude_domains) or \
               any(domain.endswith(tld) for tld in exclude_domains if tld.startswith('.')):
                logger.debug(f"Filtering excluded domain/TLD: {cleaned}")
                continue

            # Optional: Skip URLs with very generic paths often found in directories/articles
            # if any(p in path.lower() for p in ['/directory/', '/article/', '/profile/', '/company/', '/biz/']):
            #      logger.debug(f"Filtering potentially non-official path: {cleaned}")
            #      continue

            filtered_urls.add(cleaned)
        except Exception as e:
            logger.debug(f"Error processing extracted URL '{url}' for filtering: {e}")
            continue

    return sorted(list(filtered_urls)) # Return sorted list

# --- Integrated SearXNG Client Logic ---
class SearXNGClient:
    """Handles communication with a SearXNG instance."""
    def __init__(self, base_url):
        if not base_url or not base_url.startswith(('http://', 'https://')):
             raise ValueError("Invalid SearXNG base URL provided.")
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Accept': 'application/json'
        })
        logger.debug(f"SearXNGClient initialized with base URL: {self.base_url}")

    def search(self, query, max_pages=1, timeout=10): # Only 1 page needed usually
        """Search SearXNG and return simplified results list."""
        all_results = []
        page = 1
        logger.debug(f"Searching SearXNG for '{query}' (Max pages: {max_pages})")
        while page <= max_pages:
            encoded_query = quote_plus(query)
            url = f"{self.base_url}/search?q={encoded_query}&format=json&pageno={page}"
            logger.debug(f"Fetching page {page}: {url}")
            try:
                response = self.session.get(url, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                page_results = data.get('results', [])
                if page_results:
                    logger.debug(f"Got {len(page_results)} results from page {page}.")
                    for result in page_results:
                        simplified = {
                            'title': result.get('title', ''),
                            'url': result.get('url', ''),
                            'content': result.get('content', '')
                        }
                        all_results.append(simplified)
                    page += 1
                    time.sleep(0.2)
                else:
                    logger.debug(f"No more results found on page {page}.")
                    break
            except requests.exceptions.RequestException as e:
                logger.error(f"SearXNG request error page {page}, query '{query}': {e}")
                return None # Indicate failure
            except json.JSONDecodeError:
                logger.error(f"SearXNG JSON decode error page {page}, query '{query}'.")
                return None # Indicate failure
            except Exception as e:
                logger.error(f"Unexpected SearXNG error page {page}, query '{query}': {e}")
                logger.debug(traceback.format_exc())
                return None # Indicate failure

        logger.debug(f"SearXNG search for '{query}' complete. Found {len(all_results)} results.")
        return all_results # Return the list of result dicts

def search_web_standalone(query, searx_base_url, pages=1):
    """Standalone search returning list of result dicts or None."""
    if not searx_base_url:
        logger.error("No SearXNG base URL provided.")
        return None
    try:
        client = SearXNGClient(searx_base_url)
        results_list = client.search(query, max_pages=pages)
        return results_list # Can be None if search failed
    except ValueError as e:
        logger.error(f"Failed to initialize SearXNGClient: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during search_web_standalone: {e}")
        logger.debug(traceback.format_exc())
        return None

# --- Integrated LLM Call Logic ---
def sanitize_prompt(prompt):
    """Basic prompt sanitization."""
    if not isinstance(prompt, str): prompt = str(prompt)
    prompt = prompt[:MAX_PROMPT_LENGTH] # Truncate
    prompt = prompt.replace('\0', '') # Remove null bytes
    prompt = html.escape(prompt) # Escape HTML
    # Remove most control characters except whitespace
    prompt = ''.join(c for c in prompt if ord(c) >= 32 or c in '\n\r\t')
    prompt = re.sub(r'\s+', ' ', prompt).strip() # Normalize whitespace
    return prompt

def _ensure_llm_libs(provider):
    """Checks and imports necessary LLM libraries on demand."""
    global genai, openai
    provider = provider.lower()
    if provider == 'gemini':
        if genai is None:
            try:
                import google.generativeai as genai_lib
                genai = genai_lib
                logger.debug("Gemini library imported.")
                return True
            except ImportError:
                logger.error("google.generativeai package not installed (pip install google-generativeai). Cannot use Gemini.")
                return False
        return True # Already imported
    elif provider in ['openai', 'groq']:
        if openai is None:
            try:
                import openai as openai_lib
                openai = openai_lib
                logger.debug("OpenAI library imported.")
                return True
            except ImportError:
                logger.error("openai package not installed (pip install openai). Cannot use OpenAI or Groq.")
                return False
        return True # Already imported
    elif provider == 'ollama':
         return True # Ollama uses requests, which is already imported
    else:
         logger.error(f"Unsupported LLM provider: {provider}")
         return False

def _call_llm(prompt, provider, api_key=None, model_name=None):
    """Routes the LLM call, imports libs JIT, handles errors."""
    sanitized_prompt = sanitize_prompt(prompt)
    if not sanitized_prompt:
        return None, "Invalid or empty prompt after sanitization"

    provider = provider.lower()
    if not _ensure_llm_libs(provider):
        return None, f"Missing library for {provider}"

    if model_name:
        logger.info(f"Attempting LLM call: Provider={provider}, Model={model_name}")
    else:
        logger.info(f"Attempting LLM call: Provider={provider} (using fallback)")

    response = None
    model_used_info = f"{provider}:failed" # Default if call fails

    # --- Inner functions for specific providers (similar to CEO finder) ---
    def _get_gemini(p, key, model):
        genai.configure(api_key=key)
        models_to_try = [model] if model else GEMINI_MODEL_FALLBACK
        last_exc = None
        for m in models_to_try:
            logger.debug(f"Trying Gemini model: {m}")
            try:
                llm = genai.GenerativeModel(m)
                resp = llm.generate_content(p, request_options={"timeout": 45}) # Add timeout
                logger.info(f"Gemini ({m}) success.")
                return json.dumps({"response": resp.text}), f"gemini:{m}"
            except Exception as e:
                last_exc = e
                logger.warning(f"Gemini ({m}) failed: {e}")
                time.sleep(INITIAL_RETRY_DELAY) # Simple delay on failure
        logger.error(f"Gemini failed all models. Last error: {last_exc}")
        return None, f"gemini:failed ({last_exc})"

    def _get_openai(p, key, model):
        client = openai.OpenAI(api_key=key, timeout=45.0)
        models_to_try = [model] if model else OPENAI_MODEL_FALLBACK
        last_exc = None
        for m in models_to_try:
            logger.debug(f"Trying OpenAI model: {m}")
            try:
                resp = client.chat.completions.create(model=m, messages=[{"role": "user", "content": p}])
                logger.info(f"OpenAI ({m}) success.")
                return json.dumps({"response": resp.choices[0].message.content}), f"openai:{m}"
            except Exception as e:
                last_exc = e
                logger.warning(f"OpenAI ({m}) failed: {e}")
                time.sleep(INITIAL_RETRY_DELAY)
        logger.error(f"OpenAI failed all models. Last error: {last_exc}")
        return None, f"openai:failed ({last_exc})"

    def _get_groq(p, key, model):
        client = openai.OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1", timeout=45.0)
        models_to_try = [model] if model else GROQ_MODEL_FALLBACK
        last_exc = None
        for m in models_to_try:
            logger.debug(f"Trying Groq model: {m}")
            try:
                resp = client.chat.completions.create(model=m, messages=[{"role": "user", "content": p}])
                logger.info(f"Groq ({m}) success.")
                return json.dumps({"response": resp.choices[0].message.content}), f"groq:{m}"
            except Exception as e:
                last_exc = e
                logger.warning(f"Groq ({m}) failed: {e}")
                time.sleep(INITIAL_RETRY_DELAY)
        logger.error(f"Groq failed all models. Last error: {last_exc}")
        return None, f"groq:failed ({last_exc})"

    def _get_ollama(p, model):
        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/api/generate")
        m = model if model else OLLAMA_DEFAULT_MODEL
        logger.debug(f"Trying Ollama model: {m} at {ollama_url}")
        try:
            resp = requests.post(ollama_url, json={"model": m, "prompt": p, "stream": False}, timeout=60)
            resp.raise_for_status()
            resp_json = resp.json()
            logger.info(f"Ollama ({m}) success.")
            return json.dumps({"response": resp_json.get('response', '').strip()}), f"ollama:{m}"
        except Exception as e:
            logger.error(f"Ollama ({m}) failed: {e}")
            return None, f"ollama:{m}_failed ({e})"

    # --- Route call based on provider ---
    try:
        if provider == "gemini":
            if not api_key: raise ValueError("API key required for Gemini.")
            response, model_used_info = _get_gemini(sanitized_prompt, api_key, model_name)
        elif provider == "openai":
            if not api_key: raise ValueError("API key required for OpenAI.")
            response, model_used_info = _get_openai(sanitized_prompt, api_key, model_name)
        elif provider == "groq":
            if not api_key: raise ValueError("API key required for Groq.")
            response, model_used_info = _get_groq(sanitized_prompt, api_key, model_name)
        elif provider == "ollama":
            response, model_used_info = _get_ollama(sanitized_prompt, model_name)
        else:
             # Should have been caught by _ensure_llm_libs, but as fallback
             raise ValueError(f"Unsupported provider: {provider}")

        if response:
            return response, model_used_info
        else:
            # Failure message generated within provider functions
            return None, model_used_info

    except ValueError as e: # Config errors
        logger.error(f"Configuration error for {provider}: {e}")
        return None, f"{provider}:config_error ({e})"
    except Exception as e: # Catch-all for unexpected errors during call
        logger.error(f"Unexpected error during {provider} call: {e}")
        logger.debug(traceback.format_exc())
        return None, f"{provider}:unexpected_error ({e})"


# --- Domain Finder Core Logic ---
class URLSearcher:
    def __init__(self, searx_url, llm_provider, llm_api_key=None, llm_model=None):
        if not searx_url: raise ValueError("SearXNG URL required.")
        if not llm_provider: raise ValueError("LLM provider required.")
        self.searx_url = searx_url
        self.llm_provider = llm_provider.lower()
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        log_model = f", Model: {self.llm_model}" if self.llm_model else " (using fallback)"
        logger.debug(f"URLSearcher initialized: SearXNG={searx_url}, LLM={self.llm_provider}{log_model}")

    def create_llm_prompt(self, company_name, search_results):
        """Generate prompt for LLM to identify the official website."""
        if not search_results: return None

        prompt = f"TASK: Identify the official website URL for the company \"{company_name}\".\n\n"
        prompt += f"SEARCH RESULTS (Top {MAX_RESULTS_FOR_LLM}):\n"
        prompt += "--------------------\n"
        for i, result in enumerate(search_results[:MAX_RESULTS_FOR_LLM]):
            prompt += f"Result {i+1}:\n"
            prompt += f"  Title: {result.get('title', 'N/A')}\n"
            prompt += f"  URL: {result.get('url', 'N/A')}\n"
            # Limit content length per result to keep prompt manageable
            content_snippet = (result.get('content', '') or '')[:250]
            if len(result.get('content', '')) > 250: content_snippet += "..."
            prompt += f"  Snippet: {content_snippet}\n\n"
        prompt += "--------------------\n"
        prompt += "INSTRUCTIONS:\n"
        prompt += f"1. Carefully analyze the search results provided above.\n"
        prompt += f"2. Determine the single, most likely official website URL for the company named \"{company_name}\".\n"
        prompt += "3. Prioritize URLs that clearly belong to the company (check domain name against company name, title, snippet content).\n"
        prompt += "4. Strongly prefer primary domains (e.g., `company.com`) over subdomains unless the subdomain is clearly the main site (e.g., `app.company.com`).\n"
        prompt += "5. EXCLUDE URLs that are clearly:\n"
        prompt += "   - Social media (LinkedIn, Facebook, Twitter/X, Instagram, etc.)\n"
        prompt += "   - News articles or press releases about the company\n"
        prompt += "   - Directory listings (Yelp, Crunchbase, etc.)\n"
        prompt += "   - Job boards (Indeed, Glassdoor, etc.)\n"
        prompt += "   - Wikipedia pages\n"
        prompt += "   - Stock information sites\n"
        prompt += "   - Generic blog platforms or forums\n"
        prompt += "6. Return ONLY the full, validated official URL (starting with http:// or https://). Clean it by removing any trailing punctuation or unnecessary characters.\n"
        prompt += "7. If multiple strong candidates exist, choose the one that seems most official or primary based on the title and snippet.\n"
        prompt += "8. If NO reliable official website can be determined from the results, return ONLY the single word: null\n\n"
        prompt += "OFFICIAL URL:"

        return prompt

    def extract_clean_url_from_llm_response(self, response_text):
        """Extracts and cleans the single URL from the LLM response."""
        if not response_text: return None

        # Handle if response is tuple (json_str, model_info)
        if isinstance(response_text, tuple):
            json_str = response_text[0]
            try:
                 response_data = json.loads(json_str)
                 response_str = response_data.get("response", "")
            except (json.JSONDecodeError, TypeError):
                 logger.warning(f"Could not decode JSON from LLM response tuple: {json_str}")
                 return None # Cannot parse
        elif isinstance(response_text, str):
             response_str = response_text # Assume raw response text
             # Try to parse if it looks like the JSON structure we expect
             if response_str.strip().startswith('{'):
                 try:
                      response_data = json.loads(response_str)
                      response_str = response_data.get("response", response_str)
                 except json.JSONDecodeError:
                      pass # Keep original string if it wasn't valid JSON
        else:
             logger.warning(f"Unexpected LLM response type: {type(response_text)}")
             return None

        # Clean whitespace and potential markdown backticks
        cleaned_response = response_str.strip().strip('`')

        # Check for explicit "null"
        if cleaned_response.lower() == "null":
            logger.info("LLM explicitly returned 'null'.")
            return None

        # Try extracting URLs using the regex
        extracted_urls = extract_urls(cleaned_response)

        if extracted_urls:
            # The prompt asked for ONLY the URL. Take the first valid one found.
            final_url = clean_url(extracted_urls[0]) # Clean again for good measure
            if final_url:
                logger.info(f"LLM response yielded URL: {final_url}")
                return final_url
            else:
                 logger.warning(f"URL extracted from LLM response failed final cleaning: {extracted_urls[0]}")
                 return None
        else:
            # If regex fails, but response isn't 'null', treat the whole response as potential URL
            # This handles cases where LLM just returns the URL without http://
            logger.warning(f"Could not extract standard URL from LLM response '{cleaned_response}'. Attempting to clean the response itself.")
            final_url = clean_url(cleaned_response) # Try cleaning the whole thing
            if final_url and urlparse(final_url).scheme and urlparse(final_url).netloc:
                 logger.info(f"Cleaned LLM response directly to URL: {final_url}")
                 return final_url
            else:
                 logger.error(f"LLM response was not 'null' but could not be interpreted as a valid URL: {cleaned_response}")
                 return None


    def find_company_url(self, company_name):
        """Find the company URL using SearXNG and LLM verification."""
        if not company_name:
            logger.error("No company name provided.")
            return None

        # Simple query focusing on the official site
        query = f"{company_name} official website homepage"
        logger.info(f"Searching SearXNG for: {query}")
        results = search_web_standalone(query, self.searx_url)

        if results is None: # Indicates search failure
            logger.error("SearXNG search failed. Cannot proceed.")
            return None
        if not results: # Empty results list
            logger.warning("SearXNG search returned no results.")
            # Optionally try a broader query here?
            # query = company_name
            # results = search_web_standalone(query, self.searx_url)
            # if not results: return None # Give up if broader search also fails
            return None # For now, fail if initial search is empty

        # --- LLM Analysis ---
        logger.info(f"Found {len(results)} search results. Preparing prompt for LLM analysis...")
        prompt = self.create_llm_prompt(company_name, results)

        if not prompt:
            logger.error("Failed to create LLM prompt.")
            return None # Should not happen if results exist

        logger.info("Calling LLM for official URL verification...")
        llm_response_tuple = _call_llm(
            prompt,
            self.llm_provider,
            self.llm_api_key,
            self.llm_model
        ) # Returns (response_json_str, model_used_info) or (None, error_msg)

        if not llm_response_tuple or not llm_response_tuple[0]:
             logger.error(f"LLM call failed or returned empty response. Error/Info: {llm_response_tuple[1] if llm_response_tuple else 'N/A'}")
             # --- Fallback to simple extraction ---
             logger.warning("Falling back to extracting first valid URL from search results...")
             for result in results:
                  urls_in_result = extract_urls(f"{result.get('url','')} {result.get('title','')} {result.get('content','')}")
                  if urls_in_result:
                       potential_url = clean_url(urls_in_result[0])
                       if potential_url:
                            logger.info(f"Using fallback URL from search result: {potential_url}")
                            return potential_url
             logger.error("Fallback failed: No valid URLs found in search results.")
             return None # Fallback also failed
        else:
            # LLM call was successful, attempt to extract URL
            final_url = self.extract_clean_url_from_llm_response(llm_response_tuple)
            if final_url:
                logger.info(f"LLM verification successful. Final URL: {final_url}")
                return final_url
            else:
                logger.warning("LLM response processed, but no valid URL identified by LLM or extraction failed.")
                 # --- Fallback to simple extraction ---
                logger.warning("Falling back to extracting first valid URL from search results...")
                for result in results:
                    urls_in_result = extract_urls(f"{result.get('url','')} {result.get('title','')} {result.get('content','')}")
                    if urls_in_result:
                         potential_url = clean_url(urls_in_result[0])
                         if potential_url:
                              logger.info(f"Using fallback URL from search result: {potential_url}")
                              return potential_url
                logger.error("Fallback failed: No valid URLs found in search results after failed LLM verification.")
                return None # Fallback also failed

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description="Standalone Domain Finder: Finds the official website URL for a company name.",
        epilog="Example: python standalone_domain_finder.py \"Example Corporation\" -s http://localhost:8080 -p openai --api-key YOUR_KEY -o result.json"
    )
    parser.add_argument("company_name", help="The company name to search for.")
    parser.add_argument("-s", "--searx-url", required=True, help="Base URL of the SearXNG instance.")
    parser.add_argument("-p", "--llm-provider", required=True, choices=['gemini', 'openai', 'groq', 'ollama'], help="LLM provider for verification.")
    parser.add_argument("-m", "--llm-model", help="Optional: Specific LLM model name to use.", default=None)
    parser.add_argument("-k", "--api-key", dest="llm_api_key", help="API key for the selected LLM provider (required for gemini, openai, groq).")
    parser.add_argument("-o", "--output-file", help="Path to save the output file (JSON format).", default=None)
    parser.add_argument("-l", "--log-level", default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set logging level.")

    args = parser.parse_args()

    # --- Configure Logging ---
    log_level_map = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO, 'WARNING': logging.WARNING, 'ERROR': logging.ERROR, 'CRITICAL': logging.CRITICAL}
    log_level = log_level_map.get(args.log_level.upper(), logging.INFO)
    logger.setLevel(log_level)
    # Quieten noisy libraries
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


    # --- Validate API Key Requirement ---
    if args.llm_provider in ['gemini', 'openai', 'groq'] and not args.llm_api_key:
        logger.critical(f"Error: API key (--api-key) is required for LLM provider '{args.llm_provider}'.")
        sys.exit(1)
    # Check if libraries are installed for the chosen provider
    if not _ensure_llm_libs(args.llm_provider):
         # Error message printed by _ensure_llm_libs
         sys.exit(1)

    logger.info("="*30)
    logger.info(" Starting Domain Finder ".center(30,"="))
    logger.info("="*30)
    start_time = time.time()

    # --- Instantiate and Run ---
    try:
        searcher = URLSearcher(
            searx_url=args.searx_url,
            llm_provider=args.llm_provider,
            llm_api_key=args.llm_api_key,
            llm_model=args.llm_model
        )
        found_url = searcher.find_company_url(args.company_name)
    except ValueError as e:
         logger.critical(f"Initialization error: {e}")
         sys.exit(1)
    except Exception as e:
         logger.critical(f"An unexpected error occurred during processing: {e}")
         logger.debug(traceback.format_exc())
         found_url = None # Ensure it's None on unexpected error

    end_time = time.time()
    logger.info(f"Domain search finished in {end_time - start_time:.2f} seconds.")

    # --- Prepare Output ---
    output_data = {
        "company_name_input": args.company_name,
        "official_url": found_url, # Will be None if not found
        "status": "success" if found_url else "not_found"
    }

    output_json = json.dumps(output_data, indent=2, ensure_ascii=False)

    # --- Save or Print Output ---
    if args.output_file:
        try:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output_json)
            logger.info(f"Output successfully saved to: {args.output_file}")
        except Exception as e:
            logger.error(f"Failed to write output to file '{args.output_file}': {e}")
            print("\n--- Output (Error writing to file, fallback to Console) ---")
            print(output_json)
            print("--- End Output ---")
    else:
        print("\n--- Result ---")
        print(output_json)
        print("--- End Result ---")

    if found_url is None:
        logger.warning(f"Could not find an official URL for '{args.company_name}'.")
        sys.exit(1) # Exit with error code if not found
    else:
        logger.info(f"Found URL: {found_url}")
        sys.exit(0)

if __name__ == "__main__":
    # Add initial hints for optional dependencies
    try: import google.generativeai as genai
    except ImportError: logger.info("Hint: Gemini provider requires: pip install google-generativeai")
    try: import openai
    except ImportError: logger.info("Hint: OpenAI/Groq providers require: pip install openai")

    main()