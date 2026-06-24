"""
Phonetic STT Correction Engine
Replaces mis-transcribed domain terms using sliding window + phonetic algorithms.
Handles: ontologies, acronyms, Telugu-accented English patterns.
"""

import json
import re
from pathlib import Path
from metaphone import doublemetaphone
from jellyfish import jaro_winkler_similarity
from rapidfuzz.fuzz import token_sort_ratio

DATA_DIR = Path(__file__).parent.parent / "data"

# --- Telugu accent normalization patterns ---
# Telugu speakers commonly swap/merge certain consonant sounds
TELUGU_ACCENT_PATTERNS = [
    # Only apply conservative, safe transformations
    # Do NOT touch "the", "they", "this", "that", "them", "there", "these", "those", "think", "thank"
    (r'(\w)ph(\w)', r'\1f\2'), # "ph" -> "f" sound in middle of words (e.g. "pharm" -> "farm")
]

STOPWORDS = frozenset([
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "this", "that", "these", "those", "i", "we", "you", "he", "she", "it",
    "they", "me", "us", "him", "her", "them", "my", "our", "your", "his",
    "their", "what", "which", "who", "when", "where", "how", "why", "not",
    "no", "so", "if", "then", "than", "too", "also", "just", "very",
    "run", "ran", "running", "get", "got", "set", "let", "put",
    "use", "used", "using", "make", "made", "take", "took", "come", "came",
    "see", "saw", "give", "gave", "go", "went", "know", "knew",
    "think", "thought", "say", "said", "look", "looked", "want", "wanted",
    "well", "good", "best", "new", "old", "big", "high", "low", "long",
    "now", "here", "there", "up", "down", "out", "off", "over", "all",
    "each", "more", "most", "some", "any", "few", "much", "many"
])


class PhoneticCorrector:
    def __init__(self):
        self.ontologies: list[str] = []
        self.acronym_map: dict[str, str] = {}
        self.phonetic_index: dict[str, list[str]] = {}  # metaphone_key -> [terms]
        self.ontology_lower_map: dict[str, str] = {}    # lowercase -> original
        self.known_transcription_map: dict[str, str] = {}  # wrong -> correct
        self._load_data()
        self._build_index()

    def _load_data(self):
        with open(DATA_DIR / "ontologies.json") as f:
            raw = json.load(f)
        for domain_terms in raw.values():
            self.ontologies.extend(domain_terms)

        with open(DATA_DIR / "acronyms.json") as f:
            self.acronym_map = json.load(f)

        # Load user-contributed known transcription corrections if exists
        kt_path = DATA_DIR / "known_transcriptions.json"
        if kt_path.exists():
            with open(kt_path) as f:
                self.known_transcription_map = json.load(f)

    def _build_index(self):
        """Precompute metaphone index for all ontology terms. O(n) at startup, O(1) at query time."""
        for term in self.ontologies:
            self.ontology_lower_map[term.lower()] = term
            key1, key2 = doublemetaphone(term)
            for key in [k for k in [key1, key2] if k]:
                self.phonetic_index.setdefault(key, []).append(term)

    def _metaphone_similarity(self, word: str, candidate: str) -> float:
        """Compare phonetic keys of two strings."""
        keys_a = set(k for k in doublemetaphone(word) if k)
        keys_b = set(k for k in doublemetaphone(candidate) if k)
        if not keys_a or not keys_b:
            return 0.0
        return len(keys_a & keys_b) / len(keys_a | keys_b)

    def _combined_score(self, window_text: str, candidate: str) -> float:
        """Weighted combination of phonetic + surface similarity metrics."""
        meta_score = self._metaphone_similarity(window_text, candidate)
        jw_score = jaro_winkler_similarity(window_text.lower(), candidate.lower())
        tsr_score = token_sort_ratio(window_text.lower(), candidate.lower()) / 100.0

        # Length penalty: heavily penalize large size mismatches
        len_ratio = min(len(window_text), len(candidate)) / max(len(window_text), len(candidate), 1)
        if len_ratio < 0.4:
            return 0.0

        return (0.45 * meta_score) + (0.35 * jw_score) + (0.20 * tsr_score)

    def _normalize_telugu_accent(self, text: str) -> str:
        """Apply heuristic normalization for common Telugu-accented English patterns."""
        result = text
        for pattern, replacement in TELUGU_ACCENT_PATTERNS:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result

    def correct(self, raw_transcript: str, threshold: float = 0.72) -> dict:
        """
        Main correction pipeline.
        Returns: { corrected: str, replacements: list[dict], stats: dict }
        """
        # Step 1: Normalize telugu accent patterns
        normalized = self._normalize_telugu_accent(raw_transcript)

        # Step 2: Tokenize
        tokens = normalized.split()
        n = len(tokens)
        result_tokens = []
        replacements = []
        i = 0

        # Step 3: Sliding window (max window = 8 words, covers multi-word terms)
        MAX_WINDOW = 8

        while i < n:
            matched = False

            # Try from largest window down to 1
            for window_size in range(min(MAX_WINDOW, n - i), 0, -1):
                window_tokens = tokens[i: i + window_size]
                window_text = " ".join(window_tokens)
                window_lower = window_text.lower()

                # Skip pure stopword windows
                if all(t.lower() in STOPWORDS for t in window_tokens):
                    break

                # --- Priority 1: Exact match in known transcription map ---
                if window_lower in self.known_transcription_map:
                    correct_term = self.known_transcription_map[window_lower]
                    replacements.append({
                        "original": window_text,
                        "corrected": correct_term,
                        "method": "known_transcription",
                        "confidence": 1.0
                    })
                    result_tokens.append(correct_term)
                    i += window_size
                    matched = True
                    break

                # --- Priority 2: Acronym hashmap ---
                if window_lower in self.acronym_map:
                    acronym = self.acronym_map[window_lower]
                    replacements.append({
                        "original": window_text,
                        "corrected": acronym,
                        "method": "acronym_map",
                        "confidence": 1.0
                    })
                    result_tokens.append(acronym)
                    i += window_size
                    matched = True
                    break

                # --- Priority 3: Direct ontology match (case-insensitive) ---
                if window_lower in self.ontology_lower_map:
                    correct_term = self.ontology_lower_map[window_lower]
                    result_tokens.append(correct_term)
                    i += window_size
                    matched = True
                    break

                # --- Priority 4: Phonetic similarity matching ---
                if window_size >= 1:
                    best_score = 0.0
                    best_match = None

                    # Get candidates from phonetic index (fast lookup)
                    key1, key2 = doublemetaphone(window_text)
                    candidate_set = set()
                    for key in [k for k in [key1, key2] if k]:
                        candidate_set.update(self.phonetic_index.get(key, []))

                    # Also check all ontologies for multi-word terms (slower but necessary)
                    if window_size > 1:
                        candidate_set.update(self.ontologies)

                    for candidate in candidate_set:
                        score = self._combined_score(window_text, candidate)
                        if score > best_score:
                            best_score = score
                            best_match = candidate

                    if best_score >= threshold and best_match:
                        replacements.append({
                            "original": window_text,
                            "corrected": best_match,
                            "method": "phonetic",
                            "confidence": round(best_score, 3)
                        })
                        result_tokens.append(best_match)
                        i += window_size
                        matched = True
                        break

            if not matched:
                result_tokens.append(tokens[i])
                i += 1

        corrected = " ".join(result_tokens)

        return {
            "original": raw_transcript,
            "corrected": corrected,
            "replacements": replacements,
            "stats": {
                "total_tokens": n,
                "replacements_made": len(replacements),
                "replacement_rate": round(len(replacements) / max(n, 1), 3)
            }
        }

    def add_ontology(self, term: str, domain: str = "custom") -> bool:
        """Dynamically add a new ontology term at runtime."""
        if term in self.ontologies:
            return False
        self.ontologies.append(term)
        self.ontology_lower_map[term.lower()] = term
        key1, key2 = doublemetaphone(term)
        for key in [k for k in [key1, key2] if k]:
            self.phonetic_index.setdefault(key, []).append(term)

        # Persist to ontologies file
        ont_path = DATA_DIR / "ontologies.json"
        with open(ont_path) as f:
            data = json.load(f)
        data.setdefault(domain, []).append(term)
        with open(ont_path, "w") as f:
            json.dump(data, f, indent=2)
        return True

    def add_acronym(self, enunciation: str, acronym: str) -> bool:
        """Add a new acronym mapping."""
        key = enunciation.lower().strip()
        if key in self.acronym_map:
            return False
        self.acronym_map[key] = acronym
        acr_path = DATA_DIR / "acronyms.json"
        with open(acr_path) as f:
            data = json.load(f)
        data[key] = acronym
        with open(acr_path, "w") as f:
            json.dump(data, f, indent=2)
        return True

    def log_failure(self, original: str, expected: str):
        """Log a known-bad transcription for the hashmap (self-updating)."""
        key = original.lower().strip()
        self.known_transcription_map[key] = expected
        kt_path = DATA_DIR / "known_transcriptions.json"
        existing = {}
        if kt_path.exists():
            with open(kt_path) as f:
                existing = json.load(f)
        existing[key] = expected
        with open(kt_path, "w") as f:
            json.dump(existing, f, indent=2)

    def get_stats(self) -> dict:
        return {
            "total_ontologies": len(self.ontologies),
            "total_acronyms": len(self.acronym_map),
            "known_transcriptions": len(self.known_transcription_map),
            "phonetic_index_keys": len(self.phonetic_index)
        }
