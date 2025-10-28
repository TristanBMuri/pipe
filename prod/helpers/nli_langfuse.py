# nli_langfuse.py
from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass
from typing import List, Protocol
import dotenv
import os
from openai import OpenAI

import nltk
import numpy as np

# --- New Imports for Transformers ---
import torch
from langfuse._client.observe import observe
from transformers import AutoModel, AutoTokenizer

from .nli_dataclasses import (
    PassCriteria, SentenceResult, CoverageResult,
    TopContradiction, VerificationReport, EvidenceScores, Thresholds
)

from .LangfusePrompter_v3 import LangfusePrompter


class Embedder(Protocol):
    """A protocol for any embedder class."""
    model_name: str

    def encode(self, sentences: List[str]) -> np.ndarray:
        ...


# ----------------------------
# Qwen Embedder Class
# ----------------------------

class QwenEmbedder:
    """
    Encapsulates the Qwen embedding model from transformers.
    Handles tokenization, EOS token, last-token pooling, and normalization.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str = "cpu", max_length: int = 30000):
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device(device)
        logging.info(f"Loading embedding model '{model_name}' onto {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.eos_token = self.tokenizer.eos_token
        if not self.eos_token:
            logging.warning("Could not find EOS token, embedding quality may be low.")
            self.eos_token = ""
        logging.info("Embedding model loaded.")

    def encode(self, sentences: List[str]) -> np.ndarray:
        """
        Encodes a list of sentences into a numpy array of embeddings.
        """
        if not sentences:
            return np.array([])

        # 1. Add the required EOS token to each sentence
        sentences_with_eos = [f"{s}{self.eos_token}" for s in sentences]

        # 2. Tokenize
        batch_dict = self.tokenizer(
            sentences_with_eos,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        # 3. Get model output
        with torch.no_grad():
            outputs = self.model(**batch_dict)

        # 4. Perform Last-Token Pooling
        last_hidden_state = outputs.last_hidden_state
        lengths = batch_dict['attention_mask'].sum(dim=1)
        last_token_indices = lengths - 1

        indices = last_token_indices.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, self.model.config.hidden_size
        )
        embeddings = torch.gather(last_hidden_state, 1, indices).squeeze(1)

        # 5. Normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # 6. Return as numpy array on CPU
        return embeddings.cpu().numpy()


# ----------------------------
# OpenAI Embedder Class (NEW)
# ----------------------------

class OpenAIEmbedder:
    """
    Encapsulates the OpenAI embedding models via API.
    """

    def __init__(self, model_name: str = "text-embedding-3-small", device: str = "cpu", max_length: int = 8191):
        # device and max_length are for compatibility, not actively used by OpenAI API
        self.model_name = model_name
        self.max_length = max_length
        self.device = device  # Unused, but keeps signature
        logging.info(f"Initializing embedding model '{model_name}' (OpenAI)...")

        # The OpenAI client automatically reads the OPENAI_API_KEY environment variable.
        # This requires dotenv.load_dotenv() to have been called.
        try:
            self.client = OpenAI()
            logging.info("OpenAI embedding client initialized.")
        except Exception as e:
            logging.error(f"Failed to initialize OpenAI client. Is OPENAI_API_KEY set? Error: {e}")
            raise

    def encode(self, sentences: List[str]) -> np.ndarray:
        """
        Encodes a list of sentences into a numpy array of embeddings using OpenAI API.
        """
        if not sentences:
            return np.array([])

        # OpenAI API handles batching
        try:
            # Replace newlines, as recommended by OpenAI
            sentences_cleaned = [s.replace("\n", " ") for s in sentences]

            res = self.client.embeddings.create(input=sentences_cleaned, model=self.model_name)

            # Extract embeddings and convert to numpy array
            embeddings = [item.embedding for item in res.data]
            return np.array(embeddings).astype(np.float32)

        except Exception as e:
            logging.error(f"OpenAI embedding request failed: {e}")
            return np.array([])


# ----------------------------
# Utilities (I/O and Text Processing)
# ----------------------------

def load_text(path: str) -> str:
    """Loads text from a file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@observe(name="nli", as_type="generation")
def safe_nli(llm, premise, hypothesis):
    # This call returns an AIMessage object
    r = llm.text_prompt("", prem=premise, hypothesis=hypothesis)

    msg = ""
    if hasattr(r, 'content'):
        # Access the .content attribute of the AIMessage object
        msg = r.content.strip()
    elif isinstance(r, dict):
        # Keep original logic as a fallback for other response types
        msg = (r.get("message") or {}).get("content", "").strip()

    if not msg:
        logging.warning("    - Warning: Could not extract message content from LLM response.")
        return None

    try:
        # The content *is* the JSON string
        return json.loads(msg)
    except Exception:
        # Fallback to find JSON within a potentially messy string
        m = re.search(r"\{.*\}", msg, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        else:
            logging.warning(f"    - Warning: Could not parse JSON from LLM response: {msg[:100]}...")
            return None


def load_draft_from_json(path: str, keys: List[str]) -> str:
    """Loads and concatenates specified keys from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    parts = []
    for k in keys:
        node = data
        for seg in k.split("."):
            node = node.get(seg, "") if isinstance(node, dict) else ""
        if isinstance(node, str):
            parts.append(node)
    text = " ".join(parts)
    return re.sub(r"\s+", " ", text).strip()


def default_transcript_cleaner(text: str) -> str:
    """Cleans transcript text by removing common artifacts."""
    text = re.sub(r"\\", " ", text)
    text = re.sub(r"\[(?:START|ENDE)\s*Transkriptionstext\]", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def ensure_nltk_punkt() -> None:
    """Downloads the 'punkt' tokenizer model if not found."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def sentence_splitter(text: str, language: str = "german") -> List[str]:
    """Splits text into sentences."""
    ensure_nltk_punkt()
    return nltk.sent_tokenize(text, language=language)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculates cosine similarity between two numpy vectors."""
    # Check for zero vectors to avoid division by zero
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm_v1 * norm_v2)


@dataclass
class RetrieverIndex:
    """Index for semantic retrieval using embeddings."""
    chunks: List[str]
    chunk_embeddings: np.ndarray


def build_retriever(
        chunks: List[str],
        embedder: Embedder  # <-- MODIFIED: Use generic Embedder protocol
) -> RetrieverIndex:
    """Generates embeddings for all chunks to create a retriever index."""
    logging.info(f"Generating embeddings for {len(chunks)} chunks using '{embedder.model_name}'...")

    # The .encode method handles batching efficiently
    try:
        embeddings = embedder.encode(chunks)
    except Exception as e:
        logging.error(f"Failed to generate embeddings: {e}")
        # Create an empty index on failure
        return RetrieverIndex(chunks=chunks, chunk_embeddings=np.array([]))

    return RetrieverIndex(chunks=chunks, chunk_embeddings=embeddings)


def retrieve_evidence(
        query: str,
        retr: RetrieverIndex,
        embedder: Embedder,  # <-- MODIFIED: Use generic Embedder protocol
        top_k: int
) -> List[str]:
    """Retrieves top_k evidence chunks based on semantic similarity."""

    # Check for empty index
    if retr.chunk_embeddings.shape[0] == 0:
        logging.warning("Retrieving from an empty index.")
        return []

    query_embedding = embedder.encode([query])[0]

    sims = [cosine_similarity(query_embedding, chunk_emb) for chunk_emb in retr.chunk_embeddings]
    top_indices = sorted(range(len(sims)), key=lambda j: sims[j], reverse=True)[:top_k]
    return [retr.chunks[i] for i in top_indices]


# ----------------------------
# NLI Wrapper
# ----------------------------

def nli_scores_for_evidence(
        hypothesis: str,
        premises: List[str]
) -> EvidenceScores:
    """Performs NLI for a hypothesis against multiple premises using an Ollama model."""
    if not premises:
        return EvidenceScores(0.0, 0.0, "N/A", "N/A")

    max_ent, max_contra = 0.0, 0.0
    sup_ev, con_ev = "N/A", "N/A"

    llm = LangfusePrompter("natural_language_inference")

    for prem in premises:
        user_prompt = f'Premise: "{prem}"\n\nHypothesis: "{hypothesis}"'
        try:
            result = safe_nli(llm, prem, hypothesis)
            if not result:
                logging.warning("    - Warning: empty/invalid JSON from model; skipping premise.")
                continue

            label = str(result.get("label", "")).lower()
            score = float(result.get("score", 0.0))

            if label in {"contradicted", "contradict"}:
                label = "contradiction"

            if label == "entailment" and score > max_ent:
                max_ent, sup_ev = score, prem
            elif label == "contradiction" and score > max_contra:
                max_contra, con_ev = score, prem

        except Exception as e:
            logging.error(f"    - Warning: NLI failed for this premise: {e}")
            continue

    return EvidenceScores(max_ent, max_contra, sup_ev, con_ev)


# ----------------------------
# Coverage Calculation
# ----------------------------

def semantic_coverage(
        transcript_chunks: List[str],
        transcript_embs: np.ndarray,
        draft_sents: List[str],
        embedder: Embedder,  # <-- MODIFIED: Use generic Embedder protocol
        max_sim_threshold: float
) -> CoverageResult:
    """Calculates how well the draft covers the transcript."""
    if not draft_sents:
        return CoverageResult(0.0, len(transcript_chunks), transcript_chunks[:10])

    if transcript_embs.shape[0] == 0:
        logging.warning("Calculating coverage against zero transcript embeddings.")
        return CoverageResult(0.0, 0, [])

    draft_embs = embedder.encode(draft_sents)

    if draft_embs.shape[0] == 0:
        logging.warning("No draft embeddings generated for coverage check.")
        return CoverageResult(0.0, len(transcript_chunks), transcript_chunks[:10])

    uncovered = []
    for i, chunk_emb in enumerate(transcript_embs):
        sims = [cosine_similarity(chunk_emb, draft_emb) for draft_emb in draft_embs]
        max_sim = max(sims) if sims else 0.0
        if max_sim < max_sim_threshold:
            uncovered.append(transcript_chunks[i])

    cov = 1.0 - (len(uncovered) / max(1, len(transcript_chunks)))
    return CoverageResult(cov, len(uncovered), uncovered[:10])


# ----------------------------
# Core Verification Pipeline
# ----------------------------

def verify_draft_against_transcript(
        transcript_text: str,
        draft_text: str,
        language: str,
        thresholds: Thresholds,
        criteria: PassCriteria,
        embedder_type: str = "openai",
        embedder_model_name: str | None = None
) -> VerificationReport:
    """Main function to verify a draft against a transcript using Ollama."""

    # --- 1) Setup ---

    try:
        if embedder_type.lower() == "openai":
            model_name = embedder_model_name or "text-embedding-3-small"
            embedder: Embedder = OpenAIEmbedder(model_name=model_name)
        else:
            model_name = embedder_model_name or "Qwen/Qwen3-Embedding-0.6B"
            embedder: Embedder = QwenEmbedder(model_name=model_name)

    except ImportError as e:
        logging.error(f"FATAL: Could not import dependencies for embedder. {e}")
        logging.error("Please install dependencies (e.g., transformers, torch, openai)")
        raise SystemExit(1)
    except Exception as e:
        logging.error(f"FATAL: Could not load embedding model: {e}")
        logging.error("This may be due to a missing API key (OPENAI_API_KEY) or dependency conflict.")
        raise SystemExit(1)

    # 1.b) Setup NLI Client (using ollama)
    # (No changes needed here)

    # 1.c) Process texts
    cleaned = default_transcript_cleaner(transcript_text)
    transcript_chunks = sentence_splitter(cleaned, language=language)
    draft_sents = [s for s in sentence_splitter(draft_text, language=language) if s.strip()]

    if not transcript_chunks:
        print("FATAL: Transcript is empty or could not be chunked.")
        raise SystemExit(1)
    if not draft_sents:
        print("WARNING: Draft is empty or could not be chunked.")
        # Return a failing report
        return VerificationReport(0.0, 0.0, 0, CoverageResult(0.0, len(transcript_chunks), transcript_chunks[:10]),
                                  TopContradiction(None, None, -1.0), [], [], [], False)

    # 2) Build Retriever
    retr = build_retriever(transcript_chunks, embedder)

    if retr.chunk_embeddings.shape[0] == 0:
        print("FATAL: Could not build retriever index. Embeddings failed.")
        raise SystemExit(1)

    # 3) Per-sentence NLI
    per_sentence, contested, unsupported = [], [], []
    supported, hallucinated, contradiction_hits = 0, 0, 0
    top_contra = TopContradiction(None, None, -1.0)

    for i, sent in enumerate(draft_sents):
        logging.info(f"Verifying sentence {i + 1}/{len(draft_sents)}: {sent[:80]}...")

        evidence = retrieve_evidence(sent, retr, embedder, thresholds.top_k_retrieval)

        if not evidence:
            logging.warning("  - No evidence retrieved.")
            res = SentenceResult(sent, False, 0.0, 0.0, None, None, "No evidence retrieved")
            hallucinated += 1
            unsupported.append(res)
            per_sentence.append(res)
            continue

        # Use NLI client
        es = nli_scores_for_evidence(sent, evidence)
        if es.contradiction > top_contra.score:
            top_contra = TopContradiction(sent, es.conflicting_evidence, es.contradiction)

        is_supported = es.entailment >= thresholds.entailment_support and es.contradiction <= thresholds.contradiction_support

        if is_supported:
            supported += 1
            res = SentenceResult(sent, True, es.entailment, es.contradiction, es.supporting_evidence,
                                 es.conflicting_evidence)
        else:
            hallucinated += 1
            res = SentenceResult(sent, False, es.entailment, es.contradiction, es.supporting_evidence,
                                 es.conflicting_evidence, "Low entailment or high contradiction")
            unsupported.append(res)

        if es.contradiction > 0.5: contradiction_hits += 1
        per_sentence.append(res)

    supported_rate = (supported / len(draft_sents)) if draft_sents else 1.0
    hallucination_rate = (hallucinated / len(draft_sents)) if draft_sents else 0.0

    # 4) Coverage
    logging.info("Calculating coverage...")
    coverage = semantic_coverage(
        transcript_chunks, retr.chunk_embeddings, draft_sents,
        embedder, thresholds.max_sim_coverage
    )

    # 5) Final Decision
    passed = all([
        supported_rate >= criteria.min_supported_rate,
        hallucination_rate <= criteria.max_hallucination_rate,
        coverage.coverage >= criteria.min_coverage,
        (contradiction_hits == 0) if criteria.zero_contradictions else True
    ])
    logging.info(f"Verification complete. Overall pass: {passed}")

    return VerificationReport(
        supported_rate, hallucination_rate, contradiction_hits, coverage,
        top_contra, contested, unsupported, per_sentence, passed
    )


if __name__ == "__main__":
    import argparse
    from dataclasses import asdict
    from pathlib import Path  # <-- NEW (for dotenv fix)

    # --- NEW: Load .env file from project root ---
    # This is crucial for both Langfuse and OpenAI keys
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    ENV_PATH = PROJECT_ROOT / ".env"

    if ENV_PATH.exists():
        dotenv.load_dotenv(dotenv_path=ENV_PATH)
        logging.info(f"Loaded environment variables from {ENV_PATH}")
    else:
        # Fallback for safety, though it will likely fail
        logging.warning(f".env file not found at {ENV_PATH}. Trying default load.")
        dotenv.load_dotenv()
    # --- End of new block ---

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(
        description="Verify a draft against a transcript using local Transformers + Ollama NLI.")
    # Inputs
    parser.add_argument("--transcript", type=str, default=None,
                        help="Path to plaintext transcript (.txt). If omitted, uses demo text.")
    parser.add_argument("--draft", type=str, default=None,
                        help="Path to plaintext draft (.txt). If omitted, uses demo text.")
    parser.add_argument("--draft-json", type=str, default=None,
                        help="Path to JSON file to extract draft text from (mutually exclusive with --draft).")
    parser.add_argument("--draft-json-keys", type=str, default=None,
                        help='Comma-separated dotted keys for --draft-json (e.g., "title,summary.body").')

    # Language
    parser.add_argument("--language", type=str, default="german",
                        help='Language for sentence splitting (e.g., "german", "english").')

    # Model config
    # --- MODIFIED: Split embed-model and added embedder-type ---
    parser.add_argument("--embedder-type", type=str, default="qwen",
                        choices=["qwen", "openai"],
                        help="The embedding provider to use.")
    parser.add_argument("--embed-model", type=str, default=None,
                        help="Override default embedding model name (e.g., 'text-embedding-3-large' or 'Qwen/Qwen3-Embedding-0.6B').")
    # ---
    parser.add_argument("--nli-model", type=str, default="llama3",
                        help="[UNUSED] This argument is for future use if NLI model is configurable.")
    parser.add_argument("--cpu-host", type=str, default="http://localhost:11435",
                        help="[UNUSED] This argument is ignored.")
    parser.add_argument("--gpu-host", type=str, default="http://localhost:11434",
                        help="[UNUSED] This argument is ignored.")

    # Thresholds
    parser.add_argument("--entailment-support", type=float, default=0.55,
                        help="Minimum entailment score to count as supported.")
    parser.add_argument("--contradiction-support", type=float, default=0.35,
                        help="Maximum contradiction score allowed for a sentence to still count as supported.")
    parser.add_argument("--top-k-retrieval", type=int, default=5,
                        help="Number of evidence chunks retrieved per sentence.")
    parser.add_argument("--max-sim-coverage", type=float, default=0.55,
                        help="Similarity threshold for coverage computation.")

    # Pass criteria
    parser.add_argument("--min-supported-rate", type=float, default=0.70,
                        help="Minimum fraction of draft sentences supported.")
    parser.add_argument("--max-hallucination-rate", type=float, default=0.30,
                        help="Maximum fraction of unsupported sentences allowed.")
    parser.add_argument("--min-coverage", type=float, default=0.70,
                        help="Minimum transcript coverage required.")
    parser.add_argument("--zero-contradictions", action="store_true",
                        help="Fail if any high-contradiction sentence is found (>0.5).")

    # Output
    parser.add_argument("--out", type=str, default=None,
                        help="Optional path to write the JSON report. Prints to stdout otherwise.")

    args = parser.parse_args()

    # --- Load inputs ---
    if args.transcript:
        transcript_text = load_text(args.transcript)
    else:
        # Minimal German demo transcript
        logging.info("No transcript provided, using demo text.")
        transcript_text = (
            "Herr Müller, 58 Jahre alt, stellte sich mit seit drei Tagen bestehenden Kopfschmerzen vor. "
            "Er nahm Paracetamol mit mäßiger Linderung. Fieber wurde verneint."
        )

    if args.draft and args.draft_json:
        raise SystemExit("Please provide either --draft or --draft-json, not both.")

    if args.draft:
        draft_text = load_text(args.draft)
    elif args.draft_json:
        if not args.draft_json_keys:
            raise SystemExit("--draft-json requires --draft-json-keys (comma-separated dotted paths).")
        keys = [k.strip() for k in args.draft_json_keys.split(",") if k.strip()]
        draft_text = load_draft_from_json(args.draft_json, keys)
    else:
        # Minimal German demo draft
        logging.info("No draft provided, using demo text.")
        draft_text = (
            "Der 58-jährige Herr Müller berichtet über Kopfschmerzen seit drei Tagen. "
            "Er nahm Paracetamol mit etwas Wirkung. Fieber besteht nicht."
        )

    thresholds = Thresholds(
        entailment_support=args.entailment_support,
        contradiction_support=args.contradiction_support,
        top_k_retrieval=args.top_k_retrieval,
        max_sim_coverage=args.max_sim_coverage,
    )

    criteria = PassCriteria(
        min_supported_rate=args.min_supported_rate,
        max_hallucination_rate=args.max_hallucination_rate,
        min_coverage=args.min_coverage,
        zero_contradictions=args.zero_contradictions,
    )

    # --- Run verification ---
    try:
        report = verify_draft_against_transcript(
            transcript_text=transcript_text,
            draft_text=draft_text,
            language=args.language,
            thresholds=thresholds,
            criteria=criteria,
            embedder_type=args.embedder_type,  # <-- MODIFIED
            embedder_model_name=args.embed_model  # <-- MODIFIED
        )

        # Convert dataclass (with nested dataclasses) to JSON-serializable dict
        report_dict = asdict(report)

        payload = json.dumps(report_dict, ensure_ascii=False, indent=2)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(payload)
            print(f"Wrote report to {args.out}")
        else:
            print(payload)

    except SystemExit:
        # Raised by the script for known fatal errors
        print("Verification halted.")
    except Exception as e:
        # Surface a clear failure
        import traceback

        print(f"[FATAL] An unexpected error occurred: {e}")
        traceback.print_exc()