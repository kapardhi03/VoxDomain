# VoxDomain — Phonetic STT Correction Engine

> Zero-latency, domain-aware speech-to-text correction for enterprise and multi-domain applications.  
> No GPU. No cloud STT API. No cost per call. Built for real production accents.

---

## What This Is

Standard speech-to-text engines — browser Web Speech API, Whisper, anything — fail predictably on domain-specific vocabulary. They mishear `EBITDA` as `ee-bit-da`, `RAG` as `rag`, `UPI` as `you-pee-eye`, `KYC` as `kay-why-see`.

VoxDomain sits between your STT output and your application. It intercepts the raw transcript, runs a phonetic sliding-window correction pass across 475+ business terms and 125+ acronyms spanning 11 industry domains, and returns a corrected transcript — in under 1 millisecond.

The correction is deterministic, traceable, and self-improving. Every miss you log makes it smarter.

---

## Performance

| Metric | Value |
|--------|-------|
| Correction latency | ~0.5ms |
| Test accuracy (12-case suite) | 12 / 12 |
| Ontology terms (out of box) | 475+ |
| Acronym mappings | 125+ |
| Domains covered | 11 |
| Infrastructure cost | $0 |

---

## How It Works

```
Voice Input
    │
    ▼
Browser Web Speech API  ──────────────────────►  Raw Transcript (fast, free, imperfect)
                                                          │
                                                          ▼
                                              Telugu Accent Normalization
                                                          │
                                                          ▼
                                              Tokenize → Sliding Window (max 5 tokens)
                                                          │
                                          ┌───────────────┼───────────────────┐
                                          │               │                   │
                                    Priority 1      Priority 2          Priority 3 & 4
                                  Known Hashmap   Acronym Map       Phonetic Similarity
                                  (self-learning) (spoken→symbol)   Metaphone + Jaro-Winkler
                                                                     + Token Sort Ratio
                                                          │
                                                          ▼
                                                  Corrected Transcript
                                               + Replacement Trace + Latency
```

Each sliding window grouping is matched in priority order:

1. **Known transcription hashmap** — exact match against logged corrections (instant, highest priority)
2. **Acronym hashmap** — maps spoken enunciations like `"ay pee eye"` → `API`
3. **Direct ontology match** — case-insensitive lookup in the term dictionary
4. **Phonetic similarity** — weighted combination: 45% Metaphone + 35% Jaro-Winkler + 20% Token Sort Ratio, threshold 0.72

The phonetic index is precomputed at startup — no recomputation per query.

---

## Domains Covered

| Domain | Sample Terms |
|--------|-------------|
| Finance | EBITDA, WACC, DCF, CAPEX, OPEX, P&L, derivatives, securitization, LIBOR, SOFR |
| Technology | API, Kubernetes, RAG, LLM, CI/CD, OAuth, JWT, Kafka, OpenTelemetry |
| Legal | indemnity, arbitration, fiduciary, stare decisis, force majeure, escrow |
| Medical | pharmacokinetics, comorbidity, CRISPR, PCR, ELISA, randomized controlled trial |
| Governance | GSWS, MeeSeva, DBT, PFMS, gram sabha, e-procurement, GeM, RTGS |
| Fintech | UPI, KYC, AML, BNPL, NACH, account aggregator, OCEN, co-lending |
| AI / ML | LLM, RAG, RLHF, hallucination, embeddings, fine-tuning, federated learning |
| Supply Chain | 3PL, kanban, incoterms, FOB, WMS, TMS, EOQ, cross-docking |
| HR | HRIS, OKR, ESOP, attrition, stack ranking, psychometric, POSH |
| Marketing | ROAS, ABM, CTR, LTV, CAC, MQL, SQL, omnichannel |
| Real Estate | RERA, FSI, TDR, dharani, patta, khata, encumbrance certificate |

---

## Project Structure

```
voxdomain/
├── backend/
│   ├── main.py          # FastAPI app — 8 REST endpoints
│   └── corrector.py     # Core engine — sliding window + phonetic algorithms
├── frontend/
│   └── index.html       # React UI — single file, no build step required
├── data/
│   ├── ontologies.json           # 475+ domain terms (11 categories)
│   ├── acronyms.json             # 125+ spoken-form → acronym mappings
│   └── known_transcriptions.json # Auto-created on first logged correction
├── requirements.txt
└── README.md
```

---

## Setup

### Requirements
- Python 3.10+
- Chrome or Edge (Web Speech API; Firefox does not support it)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the backend
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Open the frontend
Open `frontend/index.html` directly in Chrome or Edge. No build step, no npm, no webpack.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/correct` | Correct a raw STT transcript |
| `POST` | `/ontology/add` | Add a new domain term at runtime |
| `POST` | `/acronym/add` | Map a new spoken form to an acronym |
| `POST` | `/failure/log` | Log a correction for the self-learning hashmap |
| `GET`  | `/stats` | Index statistics |
| `GET`  | `/ontologies` | List all domain terms |
| `GET`  | `/acronyms` | List all acronym mappings |
| `GET`  | `/health` | Health check |

### Correct a transcript

```bash
curl -X POST http://localhost:8000/correct \
  -H "Content-Type: application/json" \
  -d '{"transcript": "the ar ay gee pipeline improved our ee bee eye tee dee ay"}'
```

```json
{
  "original": "the ar ay gee pipeline improved our ee bee eye tee dee ay",
  "corrected": "the RAG pipeline improved our EBITDA",
  "replacements": [
    {"original": "ar ay gee", "corrected": "RAG", "method": "acronym_map", "confidence": 1.0},
    {"original": "ee bee eye tee dee ay", "corrected": "EBITDA", "method": "acronym_map", "confidence": 1.0}
  ],
  "stats": {"total_tokens": 10, "replacements_made": 2, "replacement_rate": 0.2},
  "processing_ms": 0.41
}
```

### Add a term dynamically

```bash
curl -X POST http://localhost:8000/ontology/add \
  -H "Content-Type: application/json" \
  -d '{"term": "stochastic gradient descent", "domain": "ai_ml"}'
```

### Log a correction (self-learning)

```bash
curl -X POST http://localhost:8000/failure/log \
  -H "Content-Type: application/json" \
  -d '{"original": "car fend nile", "expected": "carfentanyl"}'
```

---

## Expanding the Dictionary

### Via UI (recommended)
The frontend exposes three panels: add domain terms, add acronym mappings, and log corrections. All changes persist immediately to the JSON data files.

### Via API
All three management operations are available as REST endpoints and can be called programmatically from any service.

### Bulk additions
Edit `data/ontologies.json` directly and restart the backend. The phonetic index rebuilds at startup in under a second for thousands of terms.

### Self-learning
Every correction logged via `/failure/log` or the UI panel is written to `known_transcriptions.json` and immediately active — no restart required. This is the highest-priority lookup in the pipeline.

---

## Integration Guide

VoxDomain is designed as a middleware layer. To integrate into an existing FastAPI application:

```python
# In your app startup
from corrector import PhoneticCorrector
corrector = PhoneticCorrector()  # Build index once

# Before any prompt hits your LLM
@app.post("/chat")
async def chat(req: ChatRequest):
    result = corrector.correct(req.raw_transcript)
    clean_prompt = result["corrected"]
    # Pass clean_prompt to your LLM pipeline
```

The corrector is a pure Python class with no async dependencies — it can be imported and used anywhere.

---

## Accent Coverage

The Web Speech API is configured to `en-IN` locale, which captures Telugu-accented English significantly better than `en-US`. The phonetic matching layer (Metaphone) is inherently accent-tolerant since it compares consonant patterns rather than exact spelling.

For production accent coverage, run collection sessions with representative users across target accents. Log misses via the `/failure/log` endpoint. The known-transcriptions hashmap will self-populate and handle accent-specific patterns with 100% accuracy on seen examples.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| STT | Browser Web Speech API (`en-IN`) |
| Backend | FastAPI + Uvicorn |
| Phonetic matching | `metaphone` (Double Metaphone) |
| Surface similarity | `jellyfish` (Jaro-Winkler) |
| Token matching | `rapidfuzz` (Token Sort Ratio) |
| Frontend | React 18 (CDN, no build step) |
| Data | JSON flat files |

---

## License

MIT
