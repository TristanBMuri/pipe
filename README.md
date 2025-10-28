# Project Verification Pipeline

This project verifies a draft text against a ground truth transcript using NER and NLI checks.

## 1. Prerequisites

You must have [Ollama](https://ollama.com/) installed and running locally for the Natural Language Inference (NLI) step.

1.  [Download and install Ollama](https://ollama.com/).
2.  Pull the NLI model specified in the scripts (e.g., `llama3`):
    ```bash
    ollama pull llama3
    ```

### For bazzite python 3.12
    ```bash
    distrobox enter py312-project
    ```
## 2. Setup

1.  **Clone the repository:**
    ```bash
    git clone [your-repo-url]
    cd PythonProject
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Copy the example file and add your secret keys.
    ```bash
    cp .env.example .env
    ```
    Now, edit the `.env` file and paste in your API keys for Langfuse and OpenAI.

## 3. Usage

You can run the full verification pipeline directly.

```bash
# This script runs a pass/fail example defined inside it
python prod/prod_pipeline.py

# Run with Qwen embeddings (local):
python prod/helpers/nli_langfuse.py --transcript /path/to/truth.txt --draft /path/to/draft.txt --embedder-type "qwen" --embed-model "Qwen/Qwen3-Embedding-0.6B"

# Run with OpenAI embeddings (API):
python prod/helpers/nli_langfuse.py --transcript /path/to/truth.txt --draft /path/to/draft.txt --embedder-type "openai" --embed-model "text-embedding-3-large"