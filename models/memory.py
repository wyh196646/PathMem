"""
Self-Evolving Diagnostic Memory Module

This module implements the memory system for the Self-Evolving Memory-Centric Agent:
1. DiagnosticMemory: Long-term memory storing successful diagnostic experiences
2. EpisodicMemory: Short-term memory for current case evidence accumulation
3. MemoryEntry: Data structure for a single memory entry
"""

import os
import json
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """L2-normalize a single embedding vector."""
    array = np.asarray(embedding, dtype=np.float32)
    return array / (np.linalg.norm(array) + 1e-8)


def normalize_embedding_matrix(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize a batch of embeddings row-wise."""
    array = np.asarray(embeddings, dtype=np.float32)
    return array / (np.linalg.norm(array, axis=-1, keepdims=True) + 1e-8)


def cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """Cosine similarity between two embeddings."""
    lhs_norm = normalize_embedding(lhs)
    rhs_norm = normalize_embedding(rhs)
    return float(np.dot(lhs_norm.flatten(), rhs_norm.flatten()))


@dataclass
class MemoryEntry:
    """
    Memory entry structure: m_i = {e_i^slide, e_i^q, E_i^vis, R_i, C_i, a_i}

    - slide_emb: slide embedding (averaged patch embeddings)
    - question_emb: question embedding
    - visual_evidence: key visual evidence patches and their embeddings
    - reasoning_trace: intermediate reasoning trajectory
    - diagnostic_criteria: discriminative criteria used for diagnosis
    - answer: final answer
    """
    slide_id: str
    question: str
    slide_emb: np.ndarray  # e_i^slide
    question_emb: np.ndarray  # e_i^q
    visual_evidence: List[Dict]  # E_i^vis: [{patch_name, embedding, description}, ...]
    reasoning_trace: str  # R_i
    diagnostic_criteria: str  # C_i
    answer: str  # a_i
    answer_label: Optional[str] = None
    compact_diagnostic_cues: Optional[List[str]] = None
    support_count: int = 1

    def to_dict(self) -> Dict:
        """Convert to serializable dictionary."""
        return {
            "slide_id": self.slide_id,
            "question": self.question,
            "slide_emb": self.slide_emb.tolist(),
            "question_emb": self.question_emb.tolist(),
            "visual_evidence": [
                {
                    "patch_name": v["patch_name"],
                    "embedding": v["embedding"].tolist() if isinstance(v["embedding"], np.ndarray) else v["embedding"],
                    "description": v["description"]
                }
                for v in self.visual_evidence
            ],
            "reasoning_trace": self.reasoning_trace,
            "diagnostic_criteria": self.diagnostic_criteria,
            "answer": self.answer,
            "answer_label": self.answer_label or self.answer,
            "compact_diagnostic_cues": self.compact_diagnostic_cues or [self.diagnostic_criteria],
            "support_count": self.support_count,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MemoryEntry":
        """Create MemoryEntry from dictionary."""
        return cls(
            slide_id=data["slide_id"],
            question=data["question"],
            slide_emb=np.array(data["slide_emb"]),
            question_emb=np.array(data["question_emb"]),
            visual_evidence=[
                {
                    "patch_name": v["patch_name"],
                    "embedding": np.array(v["embedding"]),
                    "description": v["description"]
                }
                for v in data["visual_evidence"]
            ],
            reasoning_trace=data["reasoning_trace"],
            diagnostic_criteria=data["diagnostic_criteria"],
            answer=data["answer"],
            answer_label=data.get("answer_label", data.get("answer")),
            compact_diagnostic_cues=data.get(
                "compact_diagnostic_cues",
                [data.get("diagnostic_criteria", "")]
            ),
            support_count=int(data.get("support_count", 1)),
        )


@dataclass
class RetrievedPrototype:
    """
    Compact retrieval result exposed to the inference pipeline.

    evidence_embeddings are the top evidence patch embeddings from similar cases.
    diagnostic_cues remain text so the caller can reuse the existing text encoder.
    """
    entries: List[Tuple["MemoryEntry", float]]
    evidence_embeddings: List[np.ndarray]
    diagnostic_cues: List[str]


class DiagnosticMemory:
    """
    Long-term diagnostic memory that stores successful experiences.
    Supports:
    - Adding new memory entries from successful predictions
    - Retrieving top-K similar entries based on slide/question embeddings
    - Persistence (save/load)
    """

    def __init__(self, memory_path: Optional[str] = None):
        self.entries: List[MemoryEntry] = []
        self.memory_path = memory_path

        if memory_path and os.path.exists(memory_path):
            self.load(memory_path)

    def add_entry(self, entry: MemoryEntry):
        """Add a successful experience to memory."""
        self.entries.append(entry)

    def retrieve_top_k(
        self,
        slide_emb: np.ndarray,
        question_emb: np.ndarray,
        k: int = 3,
        lambda_slide: float = 0.5,
        lambda_question: float = 0.5
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Retrieve top-K similar memory entries.
        Similarity = lambda_s * sim(slide_emb) + lambda_q * sim(question_emb)

        Returns: List of (MemoryEntry, similarity_score)
        """
        if not self.entries:
            return []

        slide_emb = normalize_embedding(slide_emb)
        question_emb = normalize_embedding(question_emb)

        weight_sum = lambda_slide + lambda_question
        if weight_sum <= 0:
            lambda_slide = 0.5
            lambda_question = 0.5
            weight_sum = 1.0
        lambda_slide /= weight_sum
        lambda_question /= weight_sum

        scores = []
        for entry in self.entries:
            entry_slide = normalize_embedding(entry.slide_emb)
            entry_q = normalize_embedding(entry.question_emb)

            sim_slide = float(np.dot(slide_emb.flatten(), entry_slide.flatten()))
            sim_q = float(np.dot(question_emb.flatten(), entry_q.flatten()))

            combined_score = lambda_slide * sim_slide + lambda_question * sim_q
            scores.append((entry, combined_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def retrieve_prototype_bundle(
        self,
        slide_emb: np.ndarray,
        question_emb: np.ndarray,
        k: int = 3,
        lambda_slide: float = 0.5,
        lambda_question: float = 0.5
    ) -> RetrievedPrototype:
        """
        Get prototype evidence embeddings and compact diagnostic cues from top-K memories.
        """
        retrieved = self.retrieve_top_k(
            slide_emb,
            question_emb,
            k=k,
            lambda_slide=lambda_slide,
            lambda_question=lambda_question,
        )

        evidence_embeddings: List[np.ndarray] = []
        diagnostic_cues: List[str] = []

        for entry, _ in retrieved:
            for vis_ev in entry.visual_evidence:
                evidence_embeddings.append(np.asarray(vis_ev["embedding"], dtype=np.float32))

            cues = entry.compact_diagnostic_cues or [entry.diagnostic_criteria]
            diagnostic_cues.extend([cue for cue in cues if cue])

        return RetrievedPrototype(
            entries=retrieved,
            evidence_embeddings=evidence_embeddings,
            diagnostic_cues=diagnostic_cues,
        )

    def get_prototype_evidence(
        self,
        slide_emb: np.ndarray,
        question_emb: np.ndarray,
        k: int = 3
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Backward-compatible wrapper returning evidence embeddings and cues.
        """
        bundle = self.retrieve_prototype_bundle(slide_emb, question_emb, k=k)
        return bundle.evidence_embeddings, bundle.diagnostic_cues

    def save(self, path: Optional[str] = None):
        """Save memory to disk."""
        save_path = path or self.memory_path
        if not save_path:
            raise ValueError("No save path specified")

        data = [entry.to_dict() for entry in self.entries]
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(self.entries)} memory entries to {save_path}")

    def load(self, path: str):
        """Load memory from disk."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.entries = [MemoryEntry.from_dict(d) for d in data]
        print(f"Loaded {len(self.entries)} memory entries from {path}")

    def __len__(self):
        return len(self.entries)


class EpisodicMemory:
    """
    Episodic memory for current case: M^epi
    Stores observed patches, their descriptions, embeddings, and support/oppose labels.
    Supports:
    - Adding new evidence
    - Retrieving supporting/opposing evidence for hypothesis
    - Computing redundancy for patch selection
    """

    def __init__(self):
        self.evidence: List[Dict] = []  # [{patch_name, embedding, description, label}, ...]
        self.hypotheses: List[Dict] = []  # [{hypothesis, confidence, step}, ...]

    def add_evidence(
        self,
        patch_name: str,
        embedding: np.ndarray,
        description: str,
        label: Optional[str] = None  # "support" / "oppose" / None
    ):
        """Add observed evidence to episodic memory."""
        self.evidence.append({
            "patch_name": patch_name,
            "embedding": embedding,
            "description": description,
            "label": label
        })

    def add_hypothesis(self, hypothesis: str, confidence: float, step: int):
        """Record hypothesis at each reasoning step."""
        self.hypotheses.append({
            "hypothesis": hypothesis,
            "confidence": confidence,
            "step": step
        })

    def get_evidence_embeddings(self) -> np.ndarray:
        """Get all evidence embeddings as matrix."""
        if not self.evidence:
            return np.array([])
        return np.stack([e["embedding"] for e in self.evidence], axis=0)

    def compute_redundancy(self, patch_emb: np.ndarray) -> float:
        """
        Compute redundancy score Red(p, M^epi) for a candidate patch.
        Higher score means more redundant with existing evidence.
        """
        embs = self.get_evidence_embeddings()
        if len(embs) == 0:
            return 0.0

        patch_emb = normalize_embedding(patch_emb)
        embs = normalize_embedding_matrix(embs)

        sims = np.dot(embs, patch_emb.flatten())
        return float(np.max(sims))  # Maximum similarity with existing evidence

    def retrieve_supporting(self, hypothesis_emb: np.ndarray, top_k: int = 3) -> List[Dict]:
        """
        Retrieve E_t^+: evidence supporting current hypothesis.
        """
        if not self.evidence:
            return []

        hypothesis_emb = normalize_embedding(hypothesis_emb)

        scored = []
        for ev in self.evidence:
            ev_emb = normalize_embedding(ev["embedding"])
            sim = float(np.dot(ev_emb.flatten(), hypothesis_emb.flatten()))
            scored.append((ev, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, s in scored[:top_k]]

    def retrieve_opposing(self, hypothesis_emb: np.ndarray, top_k: int = 3) -> List[Dict]:
        """
        Retrieve E_t^-: evidence opposing current hypothesis (least similar).
        """
        if not self.evidence:
            return []

        hypothesis_emb = normalize_embedding(hypothesis_emb)

        scored = []
        for ev in self.evidence:
            ev_emb = normalize_embedding(ev["embedding"])
            sim = float(np.dot(ev_emb.flatten(), hypothesis_emb.flatten()))
            scored.append((ev, sim))

        scored.sort(key=lambda x: x[1])  # Ascending: least similar first
        return [e for e, s in scored[:top_k]]

    def get_all_descriptions(self) -> str:
        """Get concatenated descriptions of all evidence."""
        return "\n".join([e["description"] for e in self.evidence])

    def get_latest_hypothesis(self) -> Optional[Dict]:
        """Get the most recent hypothesis."""
        if not self.hypotheses:
            return None
        return self.hypotheses[-1]

    def clear(self):
        """Clear episodic memory for new case."""
        self.evidence = []
        self.hypotheses = []

    def __len__(self):
        return len(self.evidence)


def max_similarity_to_memory(
    query_emb: np.ndarray,
    memory_embeddings: List[np.ndarray]
) -> float:
    """Maximum cosine similarity between a query and a list of memory embeddings."""
    if not memory_embeddings:
        return 0.0

    query_emb = normalize_embedding(query_emb)
    best_score = 0.0
    for memory_emb in memory_embeddings:
        score = float(np.dot(query_emb.flatten(), normalize_embedding(memory_emb).flatten()))
        best_score = max(best_score, score)
    return best_score


def score_candidate_patch(
    patch_emb: np.ndarray,
    question_emb: np.ndarray,
    proto_evidence: List[np.ndarray],
    episodic_memory: EpisodicMemory,
    alpha: float = 0.4,
    beta: float = 0.4,
    gamma: float = 0.2
) -> float:
    """
    Score a candidate patch for selection:
    s_patch(p) = α·sim(p, e^q) + β·max_{v∈E^proto} sim(p,v) - γ·Red(p, M^epi)

    Args:
        patch_emb: Embedding of candidate patch
        question_emb: Question embedding
        proto_evidence: Prototype evidence embeddings from retrieved memories
        episodic_memory: Current episodic memory
        alpha, beta, gamma: Weighting coefficients

    Returns:
        Score for patch selection (higher is better)
    """
    patch_emb = normalize_embedding(patch_emb)
    question_emb = normalize_embedding(question_emb)

    # Term 1: similarity with question
    sim_q = float(np.dot(patch_emb.flatten(), question_emb.flatten()))

    # Term 2: max similarity with prototype evidence
    sim_proto = max_similarity_to_memory(patch_emb, proto_evidence)

    # Term 3: redundancy with episodic memory
    redundancy = episodic_memory.compute_redundancy(patch_emb)

    score = alpha * sim_q + beta * sim_proto - gamma * redundancy
    return score


def score_expand_candidate(
    patch_emb: np.ndarray,
    missing_info_emb: np.ndarray,
    memory_evidence: List[np.ndarray],
    episodic_memory: EpisodicMemory,
    alpha: float = 0.6,
    beta: float = 0.3,
    gamma: float = 0.1
) -> float:
    """
    Memory-guided evidence expansion:
    Score_expand(p) =
      alpha * sim(p, missing_info)
      + beta * max sim(p, memory evidence)
      - gamma * redundancy(p, selected_patches)
    """
    patch_emb = normalize_embedding(patch_emb)
    missing_info_emb = normalize_embedding(missing_info_emb)

    sim_missing = float(np.dot(patch_emb.flatten(), missing_info_emb.flatten()))
    sim_memory = max_similarity_to_memory(patch_emb, memory_evidence)
    redundancy = episodic_memory.compute_redundancy(patch_emb)
    return alpha * sim_missing + beta * sim_memory - gamma * redundancy


def score_zoom_candidate(
    zoom_emb: np.ndarray,
    question_emb: np.ndarray,
    memory_cue_embeddings: List[np.ndarray],
    eta: float = 0.7,
    mu: float = 0.3
) -> float:
    """
    Memory-guided zoom scoring:
    Score_zoom(z) =
      eta * sim(z, question)
      + mu * max sim(z, memory cues)
    """
    zoom_emb = normalize_embedding(zoom_emb)
    question_emb = normalize_embedding(question_emb)

    sim_question = float(np.dot(zoom_emb.flatten(), question_emb.flatten()))
    sim_cues = max_similarity_to_memory(zoom_emb, memory_cue_embeddings)
    return eta * sim_question + mu * sim_cues


def decide_action(
    coverage: float,
    confidence: float,
    has_opposing_evidence: bool,
    support_incomplete: bool,
    discriminative_missing: bool,
    fine_grained_unresolved: bool,
    tau_c: float = 0.7,
    tau_s: float = 0.8
) -> str:
    """
    Memory-Governed Agentic Decision Making.
    Decide action a_t ∈ {answer, revisit, search, zoom}

    Args:
        coverage: Cov(h_t) - evidence coverage score
        confidence: Conf(h_t) - hypothesis confidence
        has_opposing_evidence: Whether E_t^- is non-empty
        support_incomplete: Whether supporting evidence is incomplete
        discriminative_missing: Whether discriminative evidence is missing
        fine_grained_unresolved: Whether only fine-grained morphology is unresolved
        tau_c: Coverage threshold
        tau_s: Confidence threshold

    Returns:
        Action string: "answer", "revisit", "search", or "zoom"
    """
    # Rule 1: If coverage and confidence both sufficient -> answer
    if coverage >= tau_c and confidence >= tau_s:
        return "answer"

    # Rule 2: If opposing evidence exists or support incomplete -> revisit
    if has_opposing_evidence or support_incomplete:
        return "revisit"

    # Rule 3: If discriminative evidence missing -> search
    if discriminative_missing:
        return "search"

    # Rule 4: If only fine-grained morphology unresolved -> zoom
    if fine_grained_unresolved:
        return "zoom"

    # Default: search for more evidence
    return "search"
