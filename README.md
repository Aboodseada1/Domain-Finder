# Standalone Domain Finder

A Python script to find the official website URL for a given company name. It uses SearXNG for initial web searches and leverages an LLM (like Gemini, OpenAI, Groq, or Ollama) to analyze search results and identify the most likely official domain, filtering out common non-official links.

## Features

* Takes a company name as input.
* Uses SearXNG for web search.
* Employs LLM analysis (configurable provider/model) to verify the official website from search results.
* Filters out common non-official links (social media, directories, news, etc.).
* Cleans and normalizes the final URL.
* Provides fallback mechanism (extracting first likely URL from search results if LLM fails).
* Outputs result in JSON format (including status and found URL).
* Standalone CLI tool.
* Configurable logging level.
* Supports Gemini, OpenAI, Groq, and Ollama LLM providers.

## Prerequisites

1. **Python:** Python 3.8+ recommended.
2. **Pip:** Python package installer.
3. **SearXNG Instance:** Access to a running SearXNG instance and its base URL.
4. **Core Libraries:** `requests`, `termcolor` (Install via `requirements.txt`).
5. **Optional LLM Libraries:** Install based on the `--llm-provider` you intend to use:
6. * `google-generativeai` for Gemini (`pip install google-generativeai`)
   * `openai` for OpenAI or Groq (`pip install openai`)
7. **LLM API Keys (Conditional):** Required for `gemini`, `openai`, or `groq` providers.
8. **Ollama (Conditional):** If using `ollama`, ensure Ollama is installed and running.

## Installation

1. **Clone the repository or download the scripts:**
2. ```bash
   git clone https://github.com/Aboodseada1/Domain-Finder
   cd https://github.com/Aboodseada1/Domain-Finder
   ```
3. Or simply download `standalone_domain_finder.py` and `requirements.txt`.
4. **(Recommended)** Create and activate a Python virtual environment:
5. ```bash
   python -m venv venv
   source venv/bin/activate # On Windows use `venv\Scripts\activate`
   ```
6. **Install dependencies:** This installs core libs. Manually install optional LLM libs if needed (see Prerequisites).
7. ```bash
   pip install -r requirements.txt
   # Example: pip install google-generativeai # If using Gemini
   ```

## Usage

Run from your terminal:

```bash
python standalone_domain_finder.py <company_name> -s <searx_url> -p <llm_provider> [options]
```

**Arguments:**

* `company_name`: (Required) The name of the company to find the domain for (e.g., `"Example Corporation"`).
* `-s`, `--searx-url`: (Required) The base URL of your SearXNG instance.
* `-p`, `--llm-provider`: (Required) LLM provider for verification: `gemini`, `openai`, `groq`, `ollama`.
* `-m`, `--llm-model`: (Optional) Specific LLM model name (e.g., `gpt-4o-mini`). Uses provider's default/fallback if omitted.
* `-k`, `--api-key`: (Optional) API key for the selected LLM provider. **Required** for `gemini`, `openai`, `groq`.
* `-o`, `--output-file`: (Optional) Path to save the output JSON file.
* `-l`, `--log-level`: (Optional) Set logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Default: `INFO`.

## Examples

*(Replace placeholders)*

```bash
# Find domain for Google using OpenAI and print JSON to console
python standalone_domain_finder.py "Google" -s http://localhost:8080 -p openai -k YOUR_OPENAI_KEY

# Find domain for Microsoft using Gemini, specify model, save to file
python standalone_domain_finder.py "Microsoft Corporation" -s https://searx.example.org -p gemini -m gemini-1.5-flash -k YOUR_GEMINI_KEY -o microsoft_domain.json

# Find domain using local Ollama (default model) with debug logging
python standalone_domain_finder.py "Hugging Face" -s http://localhost:8080 -p ollama -l DEBUG
```

## Output Format

The script always outputs a JSON object containing:

* `company_name_input`: The original company name provided.
* `official_url`: The found official URL (string) or `null` if not found.
* `status`: Either `"success"` or `"not_found"`.

### Example Success Output:

```json
{
  "company_name_input": "Example Corporation",
  "official_url": "https://example.com",
  "status": "success"
}
```

### Example Failure Output:

```json
{
  "company_name_input": "NonExistent Company XYZ",
  "official_url": null,
  "status": "not_found"
}
```

## How It Works

1. Takes the company name as input.
2. Performs a targeted search on SearXNG (e.g., `"Company Name" official website homepage`).
3. If search results are found, it constructs a prompt containing the company name and key details from the search results (titles, URLs, snippets).
4. Sends the prompt to the specified LLM provider (Gemini, OpenAI, Groq, Ollama).
5. The LLM analyzes the results based on instructions to identify the most likely official URL, filtering out social media, directories, etc.
6. The script extracts and cleans the URL provided by the LLM.
7. If the LLM fails or doesn't return a valid URL, it falls back to extracting the first plausible, non-excluded URL directly from the initial search results.
8. Outputs the final result (found URL or null) in JSON format.

## Dependencies

Core: `requests`, `termcolor`. Optional LLM libs: `google-generativeai`, `openai`. See `requirements.txt`.

## Contributing

Feel free to suggest improvements or report bugs via issues or pull requests on [GitHub](https://github.com/Aboodseada1?tab=repositories).

## Support Me

Find this useful? Consider supporting via [PayPal](http://paypal.me/aboodseada1999). Thanks!

## License

MIT License.

```
MIT License

Copyright (c) 2025 Abood

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```