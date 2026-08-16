# GMC Policy Extraction → QMS Mapping

Reads Group Medical Cover (GMC) policy PDFs from **any insurer**, identifies the insurer and
the TPA, extracts the policy terms, and maps everything into a fixed QMS schema as JSON and
CSV — with page-level evidence and a confidence grade on every field.

Built for the Plancover technical assignment (*AI/LLM Engineering Intern — Document
Intelligence*).

```
                  ┌──────────────────────────────────────────────┐
   policy.pdf ───▶│ 1. Ingest      layout-sorted text, tables,   │
                  │                noise scrub, OCR fallback     │
                  ├──────────────────────────────────────────────┤
                  │ 2. Detect      insurer · TPA · product type  │
                  ├──────────────────────────────────────────────┤
                  │ 3. Extract     rule engine ──┐               │
                  │                LLM layer  ───┴─▶ reconcile   │
                  ├──────────────────────────────────────────────┤
                  │ 4. Map         Pydantic QMS schema           │
                  └───────────────────┬──────────────────────────┘
                                      ▼
                     JSON · flat CSV · run summary · JSON Schema
```

---

## Results on the provided samples

| Document | Insurer detected | Product | TPA | Fields filled | Flagged |
| --- | --- | --- | --- | --- | --- |
| `1.Policy Copy.pdf` | Care Health Insurance Ltd. (IRDAI 148) | GMC | in-house | 35/54 (64.8%) | 7 |
| `GHI Policy.pdf` | Care Health Insurance Ltd. (IRDAI 148) | GMC | in-house | 35/54 (64.8%) | 5 |
| `olj4KTUo9B…GMC Renewal Policy 00.pdf` | Niva Bupa Health Insurance (IRDAI 145) | GMC | in-house | 34/54 (63.0%) | 2 |
| `Net Catalyst - GPA…2022-23.pdf` | Liberty General Insurance Ltd. (IRDAI 150) | **GPA** | none | 8/17 (47.1%) | 1 |
| `Policy liberty 2022-2023.pdf` | Liberty General Insurance Ltd. (IRDAI 150) | **GPA** | none | 8/17 (47.1%) | 1 |

**Accuracy: 136/136 (100%) on a hand-verified ground-truth table** — every insurer, date,
premium, sum insured tier, headcount, room-rent basis, maternity limit, waiting-period status
and benefit verdict I could confirm by reading the PDFs myself. The table lives in
`tests/test_accuracy.py` and is never consulted by the pipeline; run `pytest` to regenerate
the score.

Unfilled fields are *genuinely absent from the documents* (AYUSH, LGBTQ+, live-in partner,
organ donor, air ambulance, surrogacy, vaccination, spouse/child/parent headcount breakdowns).
They are reported as `not_found` rather than guessed — see [Honest gaps](#honest-gaps).

> **Two of the five samples are not GMC.** `Net Catalyst` and `Policy liberty` are Group
> *Personal Accident* schedules (and are byte-identical to each other). The pipeline classifies
> product type and marks medical-benefit fields `not_applicable` instead of inventing maternity
> limits for an accident policy.

---

## Quick start

Requires Python 3.9+ (tested on 3.9.6). No API key, no database, no network access needed.

```bash
git clone <repo-url> && cd plancover
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# process every PDF in data/input, write to data/output
python run.py run

# or a single file, to a directory of your choice
python run.py run --input "data/input/GHI Policy.pdf" --output /tmp/out
```

`run.py` is a small shim that puts `src/` on the path, so no install step is needed. If you
prefer a real install, `pip install -e .` gives you a `gmc-extract` console script and
`python -m gmc_extract` (requires pip ≥ 21.3 for editable `src/` layouts).

Outputs land in `data/output/` (the committed copies were produced by exactly this command):

| File | What it is |
| --- | --- |
| `<document>.json` | Full QMS record per document: value, unit, evidence, page, source, confidence |
| `qms_flat.csv` | One row per document, one column per QMS field — the spreadsheet view |
| `run_summary.json` | Coverage, confidence histogram, review flags, warnings per document |
| `qms_schema.json` | JSON Schema generated from the Pydantic models |

Other commands:

```bash
python run.py fields                  # list every declared field spec and its cue count
python run.py schema                  # write the JSON Schema only
python -m pytest -q                   # 96 tests, prints the ground-truth accuracy report
```

### Optional extras

```bash
# OCR for scanned policies (needs the tesseract binary: brew install tesseract)
pip install pytesseract==0.3.13 pillow==11.3.0

# REST endpoint
pip install "fastapi==0.116.1" "uvicorn==0.35.0" "python-multipart==0.0.20"
uvicorn gmc_extract.api.app:app --reload
curl -F "file=@data/input/GHI Policy.pdf" http://127.0.0.1:8000/extract
```

### Enabling the LLM layer (optional)

The pipeline is **fully functional without an LLM** and runs in `rule_only` mode by default.
Adding a key switches on a second, independent extractor and enables cross-validation.

```bash
cp .env.example .env
# set GMC_LLM_PROVIDER=openai  and  OPENAI_API_KEY=sk-...
python run.py run                              # now reports mode: "hybrid"
python run.py run --no-llm                     # force deterministic mode
```

Supported: `openai` (or any OpenAI-compatible endpoint — Groq, Together, OpenRouter, vLLM),
`gemini`, `anthropic`, `ollama` (local, free, no key). **Paid-API disclosure:** the first three
are paid services if you enable them; nothing in the committed output required them.

---

## Methodology

### The central decision: hybrid, not pure-LLM and not pure-regex

The brief grades on accuracy first and adaptability second, and explicitly rules out
insurer-specific templates. I evaluated four approaches:

| Approach | Why not / why yes |
| --- | --- |
| Per-insurer regex templates | Directly violates the brief. Dies on insurer #4. **Rejected as the sole method.** |
| Pure LLM over the whole document | Generalises well, but non-deterministic, gives no evidence trail, and *invents plausible limits* — the worst failure mode when accuracy is the primary metric. Also makes the committed output unreproducible without someone's paid key. **Rejected as the sole method.** |
| RAG with embeddings + vector DB | Over-engineered for 3–6 page documents. Insurance terminology is a closed jargon vocabulary ("LSCS", "PED", "sub-limit") where exact-term matching is *more* precise than semantic similarity — which happily rates "pre-hospitalisation" and "post-hospitalisation" as near-identical when the entire task is telling them apart. **Rejected.** |
| **Hybrid: rule engine + LLM, reconciled** | ✅ Two independent methods cross-check each other; agreement means high confidence, disagreement raises a review flag instead of silently shipping a wrong number. Runs with no key and improves with one. |

### 1. Ingestion — where most of the accuracy actually comes from

**Layout-sorted extraction is the single highest-leverage decision in the project.** With
PyMuPDF's default reading order, the Niva Bupa schedule emits *every label first* and every
value ~40 lines later — `Policy number` and `00600900202301` end up nowhere near each other,
and no proximity-based extraction can work. `page.get_text("text", sort=True)` reorders spans
geometrically and restores clean `label ....... value` rows:

```
Policy number                                           00600900202301
Date and time of Policy commencement                    01-August-2023
Aggregate Sum Insured                                   4500000.00
Claims Administrator     Niva Bupa Health Insurance Company Limited …
```

Also in this stage: `pdfplumber` tables rendered as pipe-delimited rows (which keeps a
label cell adjacent to its value cell in the character stream), removal of raw PostScript
operators that the Liberty PDFs leak into their text layer (`313 292 m`, `0.1 0 0 0.1 9 0 cm`),
and a page-level OCR fallback that only fires when a page has no usable text layer.

### 2. Insurer & TPA detection — weighted multi-signal evidence

Five independent fingerprints per insurer: legal-name aliases (including former names),
brand tokens, website domain, **IRDAI registration number** and **CIN**. Each hit contributes
a weight; the highest total wins and the winning evidence is returned in the output.

A single name regex is not enough, because policy documents:

- **rename themselves** — "Care Health Insurance Ltd. *(formerly known as Religare Health
  Insurance Company Limited)*", "Niva Bupa *(formerly Max Bupa)*";
- **name other insurers** — the Liberty schedule credits "Liberty Mutual" as trade-logo owner,
  which is not the issuing entity;
- **omit the legal name** from the schedule page, leaving only a domain or an IRDAI number in
  the footer.

All five samples score 11.8 with 5/5 signals firing. IRDAI number and CIN are near-unique
fingerprints that break ties decisively; a test asserts they are unique across the registry so
a typo cannot silently misattribute a document.

**TPA detection is three tiers plus a resolution step.** A known-TPA lexicon (Medi Assist,
Paramount, Vidal, FHPL, Ericson, Raksha, MDIndia, …), then label proximity on *Existing TPA* /
*Claims Administrator* / *Third Party Administrator*, then a generic `<Name> … TPA … Ltd` shape.
Two details matter here:

- **None of these three insurers uses an external TPA** — Care Health and Niva Bupa administer
  claims in-house. Emitting `"tpa": null` would look like a broken detector, so when the claims
  administrator resolves to the insurer itself the answer is
  `"in_house_insurer_administered"` **with evidence**.
- **A broker is not a TPA.** Both the Niva Bupa and Liberty schedules name "Hii Insurance
  Broking Services Private Limited" as agent/intermediary, sitting right where a naive
  proximity search would read. Broker-shaped names are explicitly rejected, and the rejection
  is recorded in the evidence trail.

### 3. Extraction — declarative cues, not positions

All 59 field specs live in `src/gmc_extract/extraction/field_specs.py` (51 map straight to a
QMS cell; 8 are `scratch.*` intermediates the mapper post-processes, such as the flat maternity
limit that feeds the metro/non-metro columns). A spec looks like this:

```python
FieldSpec(
    path="benefits.room_and_hospitalisation.icu_charges",
    label="ICU Charges",
    kind=ValueKind.PERCENT_OR_MONEY,
    anchor_cues=("hospital accommodation", "room rent", "room and boarding", ...),
    cues=("maximum eligibility for icu hospitalization", "icu hospitalization",
          "icu charges", "intensive care unit", "icu/day", "icu"),
    percent_defaults_to_sum_insured=True,
)
```

A spec never says *"insurer X puts ICU on page 2 line 14"*. It says *"ICU charges are expressed
with these synonyms, and the value is a percentage of sum insured, a rupee cap, or a textual
limit"*. That is knowledge about **insurance language**, which transfers to unseen insurers;
positional knowledge would not. **This distinction is the whole adaptability argument.**

Extraction then resolves the value by enumerating where it could be — a generic
"read the cell next to the label" pass — and scoring the candidates:

| Where the value sits | Real example |
| --- | --- |
| right of the label, same row | Niva Bupa: `ICU  2%` |
| **the column beneath the label** | Care Health: header `Sum Insured │ …Normal Hospitalization │ …ICU Hospitalization` over `Rs. 300,000 │ 2 % of Sum Insured per day │ 4 % of Sum Insured per day` |
| left of the label, in prose | Care Health: `Rs. 75,000 for Normal` |
| the next row | vertical label/value stacks |

Candidates are ranked by cue specificity, **anchor quality** (a cue near a label-shaped
anchor beats the same words buried in a paragraph), **label shape** (a cue preceded only by an
enumeration marker is a label; one preceded by a word is a prose mention and is penalised),
strategy likelihood, and distance. Weights are deliberately coarse — tuning them precisely
against five documents would be overfitting.

Three worked examples of why this is not as simple as it looks:

- **`Normal 1%` vs `Normal 25,000`.** In the Niva Bupa schedule the word "Normal" labels both
  the room-rent percentage and the maternity limit, ~950 characters apart. Anchoring room rent
  to `Hospital Accommodation` and maternity to `Maternity Expenses`, with bounded windows,
  separates them.
- **"Rs. 75,000 for Normal and Rs. 75,000 for LSCS".** When the label sits to the *right* of
  its value, the nearest amount is the *last* one, not the first. Taking the first silently
  swaps Normal and C-Section whenever they differ.
- **Polarity.** "Pre-existing diseases **are covered**" and "Pre-Existing Disease (PED):
  **Waived Off**" mean the same thing — no PED waiting period — using opposite words. Status
  parsing is therefore *context-mode aware*: in `WAITING_PERIOD` mode "covered"/"waived" both
  map to `waived_off`, while in `BENEFIT` mode "covered" maps to `covered`. Negative phrases
  are tested before positive ones, because "not covered" contains "covered".

Normalisation is what makes the output QMS-ready. `Rs. 5 LAKH`, `INR 5 Lakhs`, `5,00,000`
(Indian grouping), `51,900,000` (international grouping), `4500000.00` and `₹75,000/-` all
become one integer — while `raw_text` keeps the original words for a human auditor. Shipping
only the number would be unverifiable; shipping only the text would be unusable.

### 4. LLM layer — retrieval + one strict-JSON call per group

Fields are grouped (room rent, maternity, waiting periods, other benefits,
infertility/ambulance, buffer, policy meta, demographics). For each group, lexical retrieval
selects the most relevant page chunks by cue overlap — reusing the *same* cue vocabulary as the
rule layer, so terminology added for one improves the other — and the model is asked for JSON
matching that group's fields only.

Eight focused prompts beat one 51-field mega-prompt: the model keeps the whole instruction in
view, context stays on-topic, the prompt fits any context window regardless of document length
(the brief mentions 50–80 page policies), and a malformed response degrades one section instead
of the whole document.

**Anti-hallucination guard.** Every populated field must come with a verbatim `evidence` quote,
which is checked against the document. Any value whose evidence cannot be located is discarded.
The check requires that **every numeric token** in the quote appears in the document, because a
fabricated limit always carries a number that is not there.

### 5. Reconciliation and confidence

| Situation | Result | Reasoning |
| --- | --- | --- |
| Rule and LLM agree | `rule+llm`, **high** | Two independent methods agreeing is the strongest available signal |
| Both found it, values differ | LLM wins, rule kept as `alternate`, **`needs_review`** | LLM handles nested conditions better; nothing is discarded |
| Both found it, *statuses* differ | LLM wins, **low**, `needs_review` | A polarity conflict is the most consequential kind |
| Rule only | `rule`, high if the match was strong and unambiguous, else medium | Deterministic and evidenced, but unconfirmed |
| LLM only | `llm`, medium | Evidence-verified, but phrasing the rule layer did not know |
| Neither | `not_found` | Reported explicitly, never dropped |

Flagging disagreement is the point. A confidently wrong number is a bug; an honest
`needs_review: true` is a working quality gate.

---

## Output schema

Every field is an object carrying both a machine value and its provenance:

```json
"room_rent": {
  "status": "covered",
  "value": 2.0,
  "unit": "percent_of_sum_insured",
  "basis": "per day",
  "display": "2% of sum insured per day",
  "raw_text": "Rs. 300,000  2 % of Sum Insured per day  4 % of Sum Insured per day",
  "page": 2,
  "source": "rule",
  "confidence": "high",
  "needs_review": false,
  "alternate": null,
  "notes": null
}
```

`status` ∈ `covered` · `not_covered` · `waived_off` · `applied` · `present` · `not_specified` ·
`not_found` · `not_applicable`.
`present` is for informational fields (policy number, premium, headcount) where "covered" would
be meaningless; `not_specified` means the *document itself* says NA/nil; `not_applicable` means
the field cannot apply to this product.

Top-level structure — **identical keys for every document**, whatever the insurer or product
(asserted by a test), so a QMS never has to handle a missing key:

```
document      file name, SHA-256, page count, OCR pages, text-layer flag
insurer       name, canonical key, IRDAI no., CIN, confidence, score, evidence[], runner_up
tpa           name, mode (external / in_house / unknown), confidence, evidence[]
policy        policy no., policyholder, product name, product type,
              previous_year_policy_period { inception, expiry, tenure, first inception },
              previous_year_premium { net, gross, tax, payment mode }
structure     family_structure { employee, spouse, children, parents, parents_in_law,
              max_children, cover_type }, sum_insured_tiers[], basis, aggregate_sum_insured
demographics  employees, spouses, children, parents, parents_in_law, dependents, total_lives
benefits      room_and_hospitalisation · maternity · waiting_periods · other_benefits ·
              infertility_and_ambulance · buffer_and_waivers
extraction    mode, provider, duration, coverage %, confidence histogram,
              fields_needing_review[], warnings[]
```

The full contract is published as JSON Schema in `data/output/qms_schema.json`.

### Mapping to the brief's field list

| Brief section | Where it lands |
| --- | --- |
| A. Insurer & TPA detection | `insurer`, `tpa` |
| B. Previous year's details, premium, structure, demographics | `policy.previous_year_policy_period`, `policy.previous_year_premium`, `structure`, `demographics` |
| Room rent, ICU, pre/post hospitalisation | `benefits.room_and_hospitalisation` |
| Maternity: 9-month wait, baby day one, vaccination, normal & C-section (metro/non-metro) | `benefits.maternity` |
| Waiting periods: 30-day, 1st/2nd year, PED | `benefits.waiting_periods` |
| Day care, OPD, teleconsultation, pharmacy, domiciliary, health check-up, modern, bariatric, psychiatric, AYUSH, LGBTQ+, live-in partner, organ donor | `benefits.other_benefits` |
| Infertility, surrogacy, ambulance, air ambulance | `benefits.infertility_and_ambulance` |
| Corporate buffer, disease-wise capping, waivers | `benefits.buffer_and_waivers` |

---

## Project structure

```
src/gmc_extract/
├── cli.py                  run / schema / fields commands
├── config.py               LLM settings from env (.env supported)
├── pipeline.py             orchestration; LLM stage is best-effort
├── ingestion/
│   ├── layout.py           layout-sorted text, PostScript scrub, table rendering
│   ├── loader.py           PDF → PolicyDocument with an offset→page index
│   └── ocr.py              optional Tesseract fallback
├── detection/
│   ├── registry.py         ← EDIT THIS to support a new insurer or TPA (data only)
│   ├── insurer.py          weighted multi-signal scoring + product classification
│   └── tpa.py              three-tier TPA resolution with broker guard
├── extraction/
│   ├── field_specs.py      ← declarative catalogue of every field (59 specs)
│   ├── parsers.py          money / percent / duration / date / status polarity
│   ├── tabular.py          generic label→cell resolution, logical-line assembly
│   ├── rule_extractor.py   cue-window scoring engine
│   ├── structural.py       family structure, sum insured tiers, product name
│   ├── retrieval.py        lexical chunk selection for the LLM
│   ├── llm_extractor.py    per-group strict-JSON extraction + evidence verification
│   ├── merge.py            reconciliation and confidence grading
│   └── mapping.py          record assembly, derivations, not_applicable marking
├── llm/providers.py        OpenAI / Gemini / Anthropic / Ollama over plain requests
├── outputs/writers.py      JSON, flat CSV, run summary, JSON Schema
├── schema/qms.py           the Pydantic QMS contract
└── api/app.py              optional POST /extract
```

**Adding a new insurer is a data edit, not a code change** — append a signature to
`detection/registry.py`. Adding a new QMS field is one `FieldSpec` plus one schema attribute.

### Technology choices

| Need | Chosen | Why not the alternative |
| --- | --- | --- |
| PDF text | **PyMuPDF** | `sort=True` is exactly the layout fix required; pdfminer is slower with no upside |
| PDF tables | **pdfplumber** | camelot needs Ghostscript — a system dependency I won't force on a reviewer |
| Schema | **Pydantic v2** | Validation *and* a free published JSON Schema |
| LLM access | **plain `requests`** | LangChain is an enormous dependency surface to wrap one HTTP POST |
| CLI | **argparse** | Stdlib; three subcommands don't justify Typer |
| Retrieval | **lexical cue overlap** | Embeddings are less precise on closed insurance jargon, and add a model + vector store |

Deliberately **not** used: LangChain/LlamaIndex, a vector database, a fine-tuned model, a
frontend, Docker.

---

## Assumptions

1. **"Previous year's details" = the period printed on these documents.** The samples are
   expiring/renewing policies (2022-23, 2023-24), so the period on the face of the document *is*
   the prior-year period the QMS asks for. Field names say `previous_year_*` to make this
   explicit.
2. **Dates are day-first** (`02/06/2022` = 2 June 2022), per Indian convention.
3. **Per-member sum insured is a multiple of ₹5,000.** This is what separates a real tier from
   the per-life premium values printed in the same rate table (`25,505.46`).
4. **A per-member sum insured lies between ₹25,000 and ₹1 crore**; anything larger near a
   sum-insured label is the group aggregate.
5. **Metro / non-metro maternity:** none of the samples differentiate, so the single stated
   limit is reported in both columns with a note saying the document made no distinction.
   Filling both is more useful to an integrator than two blanks and more honest than inventing
   a split.
6. **In-house claims administration is inferred** when the claims administrator resolves to the
   insurer itself — reported with evidence, not as a null.
7. **Total lives is derived** (`employees + dependents`) when the document leaves the total in an
   unlabelled row, as Care Health does. Marked `source: "derived"`.
8. **The document is the sole source of truth.** No market defaults, no "typical" values.

## Honest gaps

- **No OCR was exercised.** All five samples have text layers. The fallback is implemented and
  wired in, but untested against a real scanned policy, and `tesseract` is not installed in the
  environment I built this in.
- **The committed output is `rule_only`.** I had no LLM API key available. The hybrid path is
  fully implemented and tested against a stub provider (response parsing, evidence verification,
  the whole reconciliation matrix, and provider-failure degradation), but the numbers in
  `data/output/` were produced by the deterministic extractor alone. Set a key and re-run to
  regenerate them in `hybrid` mode.
- **Fields absent from the samples are untested against real text.** AYUSH, LGBTQ+, live-in
  partner, organ donor, air ambulance, surrogacy, vaccination, pharmacy discount and annual
  health check-up have cue lists but no sample document mentions them, so they correctly report
  `not_found` — and their cues are unverified.
- **Spouse / child / parent headcount breakdowns are not in any sample.** Care Health gives only
  "Primary Insured Members / Dependents"; Niva Bupa gives only employees and total insured
  members. These come back `not_found` rather than being split by guesswork.
- **Only three insurers were available to test against.** The registry ships 28, but 25 are
  unverified against a real document.
- **Documents are 3–6 pages, not the 50–80 the brief describes.** Chunked retrieval and
  page-scoped extraction are designed for length, but that has not been proven on a long policy.
- **Weights are hand-set, not learned.** With five documents, fitting them would be overfitting.
- **The REST endpoint is unauthenticated** and intended for local use only; see the note in
  `api/app.py`.

## Possible next steps

- Ground-truth set across more insurers to turn the accuracy table into a real regression suite.
- Per-insurer accuracy tracking so a registry change that helps one insurer and hurts another is
  visible.
- A review UI over `needs_review` fields — the flags are already there, so the human-in-the-loop
  step is cheap to add.
- Endorsement handling: policies get amended mid-term, and an endorsement should override the
  base schedule.

---

## Development log

`step.md` records the build in order: what was inspected, every design decision with the
options rejected and why, and a table of the 11 accuracy defects found during development with
their root causes. Worth reading alongside this file if you want the reasoning rather than the
result.

## License

MIT.
