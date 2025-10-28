from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Thresholds:
    entailment_support: float = 0.5
    contradiction_support: float = 0.35
    max_sim_coverage: float = 0.3
    top_k_retrieval: int = 5

@dataclass
class PassCriteria:
    min_supported_rate: float = 0.90
    max_hallucination_rate: float = 0.05
    min_coverage: float = 0.90
    zero_contradictions: bool = True  # no contradiction > 0.5

@dataclass
class EvidenceScores:
    entailment: float
    contradiction: float
    supporting_evidence: str
    conflicting_evidence: str

@dataclass
class SentenceResult:
    sentence: str
    supported: bool
    entailment: float
    contradiction: float
    supporting_evidence: Optional[str] = None
    conflicting_evidence: Optional[str] = None
    note: Optional[str] = None  # "contested" / "unsupported" reasons

@dataclass
class CoverageResult:
    coverage: float
    uncovered_chunks: int
    uncovered_examples: List[str]

@dataclass
class TopContradiction:
    sentence: Optional[str]
    evidence: Optional[str]
    score: float

@dataclass
class VerificationReport:
    supported_rate: float
    hallucination_rate: float
    contradiction_hits: int
    coverage: CoverageResult
    top_contradiction: TopContradiction
    contested: List[SentenceResult]
    unsupported: List[SentenceResult]
    per_sentence: List[SentenceResult]
    passed: bool