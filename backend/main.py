"""
Phonetic STT Correction API
FastAPI backend - serves correction, dictionary management, and analytics.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import time

from corrector import PhoneticCorrector

app = FastAPI(
    title="Phonetic STT Correction API",
    description="Corrects domain-specific speech-to-text transcriptions using phonetic algorithms",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton corrector - index built once at startup
corrector = PhoneticCorrector()


# --- Request/Response Models ---

class CorrectRequest(BaseModel):
    transcript: str
    threshold: Optional[float] = 0.72  # Confidence threshold for replacements


class CorrectResponse(BaseModel):
    original: str
    corrected: str
    replacements: list
    stats: dict
    processing_ms: float


class AddOntologyRequest(BaseModel):
    term: str
    domain: Optional[str] = "custom"


class AddAcronymRequest(BaseModel):
    enunciation: str   # e.g. "ay bee see"
    acronym: str       # e.g. "ABC"


class LogFailureRequest(BaseModel):
    original: str      # What STT returned (wrong)
    expected: str      # What it should have been (correct)


# --- Endpoints ---

@app.get("/")
def root():
    return {"status": "ok", "service": "Phonetic STT Correction API"}


@app.get("/health")
def health():
    return {"status": "healthy", **corrector.get_stats()}


@app.post("/correct", response_model=CorrectResponse)
def correct_transcript(req: CorrectRequest):
    if not req.transcript or not req.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript cannot be empty")

    start = time.perf_counter()
    result = corrector.correct(req.transcript, threshold=req.threshold)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    return CorrectResponse(
        original=result["original"],
        corrected=result["corrected"],
        replacements=result["replacements"],
        stats=result["stats"],
        processing_ms=elapsed_ms
    )


@app.post("/ontology/add")
def add_ontology(req: AddOntologyRequest):
    if not req.term.strip():
        raise HTTPException(status_code=400, detail="Term cannot be empty")
    added = corrector.add_ontology(req.term.strip(), req.domain)
    return {
        "success": added,
        "message": f"Term '{req.term}' added to domain '{req.domain}'" if added else f"Term '{req.term}' already exists"
    }


@app.post("/acronym/add")
def add_acronym(req: AddAcronymRequest):
    if not req.enunciation.strip() or not req.acronym.strip():
        raise HTTPException(status_code=400, detail="Both enunciation and acronym required")
    added = corrector.add_acronym(req.enunciation.strip(), req.acronym.strip())
    return {
        "success": added,
        "message": f"Mapped '{req.enunciation}' -> '{req.acronym}'" if added else "Mapping already exists"
    }


@app.post("/failure/log")
def log_failure(req: LogFailureRequest):
    """Log a bad transcription for the self-updating hashmap."""
    corrector.log_failure(req.original, req.expected)
    return {"success": True, "message": f"Logged: '{req.original}' -> '{req.expected}'"}


@app.get("/stats")
def get_stats():
    return corrector.get_stats()


@app.get("/ontologies")
def list_ontologies(domain: Optional[str] = None):
    import json
    from pathlib import Path
    with open(Path(__file__).parent.parent / "data" / "ontologies.json") as f:
        data = json.load(f)
    if domain:
        return {domain: data.get(domain, [])}
    return data


@app.get("/acronyms")
def list_acronyms():
    return corrector.acronym_map
