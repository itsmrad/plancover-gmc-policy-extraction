# GMC Policy Extraction → QMS Mapping

Reads Group Medical Cover (GMC) policy PDFs from **any insurer**, identifies the insurer and
the TPA, extracts the policy terms, and maps everything into a fixed QMS schema as JSON and
CSV — with page-level evidence and a confidence grade on **every** field.

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

## Results

### Accuracy: 136/136 (100%) on a hand-verified ground-truth table

Every insurer, date, premium, sum insured tier, head count, room-rent basis, maternity limit,
waiting-period status and benefit verdict I could confirm by reading the PDFs myself. The
table lives in `tests/test_accuracy.py`, is **never consulted by the pipeline**, and the score
regenerates on every `pytest` run.

| Document | Insurer detected | Product | TPA | Fields filled | Flagged |
| --- | --- | --- | --- | --- | --- |
| `1.Policy Copy.pdf` | Care Health Insurance Ltd. (IRDAI 148) | GMC | in-house | 35/54 | 7 |
| `GHI Policy.pdf` | Care Health Insurance Ltd. (IRDAI 148) | GMC | in-house | 36/54 | 5 |
| `olj4KTUo9B…GMC Renewal Policy 00.pdf` | Niva Bupa Health Insurance (IRDAI 145) | GMC | in-house | 34/54 | 3 |
| `Net Catalyst - GPA…2022-23.pdf` | Liberty General Insurance Ltd. (IRDAI 150) | **GPA** | none | 8/17 | 1 |
| `Policy liberty 2022-2023.pdf` | Liberty General Insurance Ltd. (IRDAI 150) | **GPA** | none | 8/17 | 1 |

All three insurers identified at **high confidence with 5/5 independent signals firing**
(legal name, CIN, IRDAI number, website domain, brand token).

### Tested live against three LLMs via OpenRouter

The hybrid layer was run end-to-end against three different models. **All three reach 136/136**,
which is the point: the pipeline is not tuned to one provider.

| Model | Ground truth | High confidence | Medium | Low | Coverage |
| --- | --- | --- | --- | --- | --- |
| `google/gemini-2.5-flash` *(committed output)* | **136/136** | 105 | 14 | 2 | 61.7% |
| `deepseek/deepseek-chat` | **136/136** | 95 | 25 | 2 | 62.2% |
| `openai/gpt-4o-mini` | **136/136** | 93 | 26 | 5 | 63.3% |
| *rule-only, no API key* | **136/136** | 69 | 51 | 0 | 61.2% |

Score any output directory yourself: `python tests/verify_output_dir.py data/output`.

### What the second extractor actually bought

Of **121 populated fields** in the committed run:

- **106 (87.6%) were independently confirmed by both extractors** → `source: "rule+llm"`,
  `confidence: "high"`. That cross-validation is the whole reason for the hybrid design.
- **12** were rule-only (phrasing the LLM did not return).
- **3** were found only by the LLM — including `total_lives = 152` from a bare `Total  152`
  row that carries no label for a cue-based extractor to latch onto.
- **17** are flagged `needs_review: true` with both candidate values retained.
- High-confidence fields went from **69 → 105** versus rule-only mode.

**The LLM found a real error in my hand-written ground truth.** I had recorded the Niva Bupa
gross premium as ₹1,04,635. The two extractors disagreed; reviewing the conflict showed page 4
of the schedule states `83,454 net + 15,022 IGST = 98,476`, while ₹1,04,635 appears on page 6
in a separate **Premium Receipt**. **98,476 is the policy's gross premium and my ground truth
was wrong.** I fixed the ground truth, taught the rule layer to ignore receipt pages, and left
the reasoning in the test file — catching my mistake is exactly what a second independent
extractor is for.

### Unfilled fields are honest, not broken

Coverage sits at ~62%. That is **not a 38% failure rate.** The unfilled fields are *genuinely
absent from these documents*: AYUSH, LGBTQ+, live-in partner, organ donor, air ambulance,
surrogacy, vaccination, pharmacy discount, annual health check-up, and the spouse/child/parent
head-count breakdown. They report `not_found`. Inventing plausible values would raise coverage
and destroy accuracy — the metric the brief weights first.

> **Two of the five samples are not GMC.** `Net Catalyst` and `Policy liberty` are Group
> *Personal Accident* schedules (and are byte-identical to each other). The pipeline classifies
> product type and marks medical-benefit fields `not_applicable` instead of inventing maternity
> limits for an accident policy. Those fields are excluded from the coverage denominator.

---

## Quick start

Python 3.9+ (tested on 3.9.6). **No API key, database or network access required.**

```bash
git clone <repo-url> && cd plancover
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run.py run                                              # every PDF in data/input
python run.py run --input "data/input/GHI Policy.pdf" -o /tmp/x # one file
```

`run.py` puts `src/` on the path so no install step is needed. `pip install -e .` also gives you
a `gmc-extract` console script.

Outputs in `data/output/` (the committed copies came from exactly this command):

| File | What it is |
| --- | --- |
| `<document>.json` | Full QMS record: value, unit, evidence, page, source, confidence |
| `qms_flat.csv` | One row per document, one column per QMS field — the spreadsheet view |
| `run_summary.json` | Coverage, confidence histogram, review flags, warnings |
| `qms_schema.json` | JSON Schema generated from the Pydantic models |

```bash
python run.py fields                       # every declared field spec and its cue count
python -m pytest -q                        # 104 tests + the ground-truth report
python tests/verify_output_dir.py <dir>     # score any output dir against ground truth
```

### Enabling the LLM layer (optional)

```bash
cp .env.example .env      # set GMC_LLM_PROVIDER=openai + OPENAI_API_KEY
python run.py run         # reports mode: "hybrid"
python run.py run --no-llm    # force deterministic mode

GMC_TEST_HYBRID=1 python -m pytest tests/test_accuracy.py -q   # score hybrid vs ground truth
```

Providers: `openai` (**and any OpenAI-compatible endpoint — OpenRouter, Groq, Together,
vLLM**), `gemini`, `anthropic`, `ollama` (local, free). **Paid-API disclosure:** testing used
OpenRouter (paid, ~$0.05 total for all runs above). Nothing in the committed rule-only path
requires it.

Optional extras:

```bash
pip install pytesseract==0.3.13 pillow==11.3.0                    # OCR (needs tesseract binary)
pip install "fastapi==0.116.1" "uvicorn==0.35.0" "python-multipart==0.0.20"
uvicorn gmc_extract.api.app:app --reload      # then POST a PDF to /extract
```

---

## Methodology

### The central decision: hybrid, not pure-LLM and not pure-regex

| Approach | Verdict |
| --- | --- |
| Per-insurer regex templates | Violates the brief; dies on insurer #4. **Rejected as sole method.** |
| Pure LLM over the whole document | Generalises, but non-deterministic, no evidence trail, and *invents plausible limits* — the worst failure mode when accuracy is graded first. Also makes the committed output unreproducible without a paid key. **Rejected as sole method.** |
| RAG with embeddings + vector DB | Over-engineered for these documents. Insurance terminology is closed jargon ("LSCS", "PED", "sub-limit") where exact-term matching beats semantic similarity — which rates "pre-hospitalisation" and "post-hospitalisation" as near-identical when the entire task is telling them apart. **Rejected.** |
| **Hybrid: rule engine + LLM, reconciled** | ✅ Two independent methods cross-check each other. Agreement → high confidence; disagreement → review flag, not a silently wrong number. Runs with no key, improves with one. |

### 1. Ingestion — where most of the accuracy comes from

**Layout-sorted extraction is the single highest-leverage decision in the project.** With
PyMuPDF's default reading order the Niva Bupa schedule emits *every label first* and every value
~40 lines later — `Policy number` and `00600900202301` end up nowhere near each other and no
proximity-based extraction can work. `page.get_text("text", sort=True)` reorders spans
geometrically:

```
Policy number                                           00600900202301
Date and time of Policy commencement                    01-August-2023
Aggregate Sum Insured                                   4500000.00
Claims Administrator     Niva Bupa Health Insurance Company Limited …
```

Also here: `pdfplumber` tables rendered as pipe rows (keeping a label cell adjacent to its value
cell in the character stream), removal of raw PostScript operators the Liberty PDFs leak into
their text layer (`313 292 m`, `0.1 0 0 0.1 9 0 cm`), and a page-level OCR fallback that fires
only when a page has no usable text layer.

### 2. Insurer & TPA detection — weighted multi-signal evidence

Five independent fingerprints per insurer: legal-name aliases (**including former names**),
brand tokens, website domain, **IRDAI registration number** and **CIN**. Each hit adds weight;
the winning evidence is returned in the output.

One regex is not enough, because policy documents:
- **rename themselves** — "Care Health *(formerly Religare)*", "Niva Bupa *(formerly Max Bupa)*";
- **name other insurers** — the Liberty schedule credits "Liberty Mutual" as trade-logo owner,
  which is not the issuer;
- **omit the legal name**, leaving only a domain or IRDAI number in the footer.

A test asserts IRDAI numbers and CINs are unique across the registry, so a typo cannot silently
misattribute a document.

**TPA detection is three tiers plus a resolution step** — a known-TPA lexicon (Medi Assist,
Paramount, Vidal, FHPL, Ericson, Raksha, MDIndia, …), then label proximity on *Existing TPA* /
*Claims Administrator* / *Third Party Administrator*, then a generic `<Name> … TPA … Ltd` shape.
Two details matter:

- **None of these three insurers uses an external TPA** — they administer claims in-house.
  `"tpa": null` would look like a broken detector, so when the claims administrator resolves to
  the insurer itself the answer is `"in_house_insurer_administered"` **with evidence**.
- **A broker is not a TPA.** Both the Niva Bupa and Liberty schedules name "Hii Insurance Broking
  Services Private Limited" as intermediary, sitting exactly where a naive proximity search would
  read. Broker-shaped names are rejected, and the rejection is recorded in the evidence trail.

### 3. Extraction — declarative cues, not positions

Every field is a small declarative spec in `extraction/field_specs.py`:

```python
FieldSpec(
    path="benefits.room_and_hospitalisation.icu_charges",
    kind=ValueKind.PERCENT_OR_MONEY,
    anchor_cues=("hospital accommodation", "room rent", "room and boarding", ...),
    cues=("maximum eligibility for icu hospitalization", "icu hospitalization",
          "icu charges", "intensive care unit", "icu/day", "icu"),
    percent_defaults_to_sum_insured=True,
)
```

A spec never says *"insurer X puts ICU on page 2 line 14"*. It says *"ICU is expressed with
these synonyms and its value is a % of sum insured, a rupee cap, or a textual limit"* —
knowledge about **insurance language**, which transfers to unseen insurers. Positional knowledge
would not. **This is the adaptability argument.**

Values are then resolved by a generic "read the cell next to the label" pass:

| Where the value sits | Real example |
| --- | --- |
| right of the label, same row | Niva Bupa: `ICU  2%` |
| **the column beneath the label** | Care Health: header `Sum Insured │ …Normal Hospitalization │ …ICU Hospitalization` over `Rs. 300,000 │ 2 % of Sum Insured per day │ 4 % of Sum Insured per day` |
| left of the label, in prose | Care Health: `Rs. 75,000 for Normal` |
| the next row | vertical label/value stacks |

Candidates are ranked by cue specificity, **anchor quality**, **label shape** (a cue preceded
only by an enumeration marker is a label; one preceded by a word is a prose mention and is
penalised), strategy likelihood and distance. Weights are deliberately coarse — tuning them
against five documents would be overfitting.

Three cases showing why this is harder than it looks:

- **`Normal 1%` vs `Normal 25,000`.** In the Niva Bupa schedule "Normal" labels both the
  room-rent percentage and the maternity limit, ~950 characters apart. Anchoring room rent to
  `Hospital Accommodation` and maternity to `Maternity Expenses` separates them.
- **"Rs. 75,000 for Normal and Rs. 75,000 for LSCS".** When the label sits to the *right* of its
  value, the nearest amount is the *last* one. Taking the first silently swaps Normal and
  C-Section whenever they differ.
- **Polarity.** "Pre-existing diseases **are covered**" and "Pre-Existing Disease (PED):
  **Waived Off**" mean the same thing using opposite words. Status parsing is *context-mode
  aware*: in `WAITING_PERIOD` mode both map to `waived_off`; in `BENEFIT` mode "covered" maps to
  `covered`. Negatives are tested before positives, because "not covered" contains "covered".

Normalisation makes the output QMS-ready: `Rs. 5 LAKH`, `INR 5 Lakhs`, `5,00,000` (Indian
grouping), `51,900,000` (international grouping), `4500000.00` and `₹75,000/-` all become one
integer, while `raw_text` keeps the original words for a human auditor.

### 4. LLM layer — retrieval + one strict-JSON call per group

Fields are grouped into eight sets. Lexical retrieval selects the most relevant page chunks by
cue overlap — reusing the *same* vocabulary as the rule layer, so terminology added for one
improves the other — and the model is asked for JSON covering that group only. Eight focused
prompts beat one 51-field mega-prompt: the model keeps the whole instruction in view, context
stays on-topic, the prompt fits any context window regardless of document length (the brief
mentions 50–80 page policies), and a bad response degrades one section, not the document.

Three guards, all of which caught real model errors during testing:

1. **Evidence verification.** Every populated field needs a verbatim quote, checked against the
   document. Any value whose evidence cannot be located is discarded. The check requires that
   **every numeric token** in the quote appears in the document — a fabricated limit always
   carries a number that is not there.
2. **Per-field status vocabulary.** Each field declares the statuses it may return, and an
   out-of-vocabulary label is coerced rather than published. Without this, models labelled
   *every* extracted figure "applied", turning 30+ correct values into spurious disagreements.
3. **Evidence repair.** If the model returns a good quote but a mangled value or no unit, the
   quote is re-parsed with the rule layer's own parsers. This fixed `gpt-4o-mini` returning
   `0.5` for "upto 50% of the Sum Insured".

### 5. Reconciliation and confidence

| Situation | Result | Reasoning |
| --- | --- | --- |
| Rule and LLM agree | `rule+llm`, **high** | Two independent methods agreeing is the strongest available signal |
| Same status, only one gives a limit | keep the limit, medium | Not a conflict — take the more informative answer |
| Conflict, rule read a **labelled field** | rule wins, `needs_review` | Niva Bupa's labelled `Co-payment  NA` beats a 50% co-pay mentioned in a Special Conditions paragraph |
| Conflict, rule read prose | LLM wins, `needs_review` | The LLM handles nested conditions better |
| Status conflict | **low** + `needs_review` | The most consequential kind of disagreement |
| One extractor only | `rule` / `llm`, medium | Evidenced but unconfirmed |
| Neither | `not_found` | Reported explicitly, never dropped |

Flagging disagreement is the point. A confidently wrong number is a bug; an honest
`needs_review: true` is a working quality gate.

---

## Output schema

Every field carries both a machine value and its provenance:

```json
"room_rent": {
  "status": "covered",
  "value": 2.0,
  "unit": "percent_of_sum_insured",
  "basis": "per day",
  "display": "2% of sum insured per day",
  "raw_text": "Rs. 300,000  2 % of Sum Insured per day  4 % of Sum Insured per day",
  "page": 2,
  "source": "rule+llm",
  "confidence": "high",
  "needs_review": false,
  "alternate": null,
  "notes": null
}
```

`status` ∈ `covered` · `not_covered` · `waived_off` · `applied` · `present` · `not_specified` ·
`not_found` · `not_applicable`. `present` is for informational fields (policy number, premium,
head count) where "covered" would be meaningless; `not_specified` means the *document* says
NA/nil; `not_applicable` means the field cannot apply to this product.

**Identical keys for every document**, whatever the insurer or product (asserted by a test), so
a QMS never handles a missing key:

```
document      file name, SHA-256, page count, OCR pages, text-layer flag
insurer       name, key, IRDAI no., CIN, confidence, score, evidence[], runner_up
tpa           name, mode (external / in_house / unknown), confidence, evidence[]
policy        policy no., policyholder, product name & type,
              previous_year_policy_period { inception, expiry, tenure, first inception },
              previous_year_premium { net, gross, tax, payment mode }
structure     family_structure { employee, spouse, children, parents, parents_in_law,
              max_children, cover_type }, sum_insured_tiers[], basis, aggregate_sum_insured
demographics  employees, spouses, children, parents, parents_in_law, dependents, total_lives
benefits      room_and_hospitalisation · maternity · waiting_periods · other_benefits ·
              infertility_and_ambulance · buffer_and_waivers
extraction    mode, provider, model, duration, coverage %, confidence histogram,
              fields_needing_review[], warnings[]
```

Full contract: `data/output/qms_schema.json`.

### Mapping to the brief

| Brief section | Where it lands |
| --- | --- |
| A. Insurer & TPA detection | `insurer`, `tpa` |
| B. Previous year, premium, structure, demographics | `policy.previous_year_*`, `structure`, `demographics` |
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
├── cli.py · config.py · pipeline.py
├── ingestion/    layout.py (sorted text, scrub) · loader.py · ocr.py
├── detection/    registry.py ← EDIT to add an insurer · insurer.py · tpa.py
├── extraction/   field_specs.py ← the field catalogue · parsers.py · tabular.py
│                 rule_extractor.py · structural.py · retrieval.py
│                 llm_extractor.py · merge.py · mapping.py
├── llm/          providers.py (OpenAI / Gemini / Anthropic / Ollama)
├── outputs/      writers.py · schema/qms.py · api/app.py
tests/            parsers · detection · accuracy (ground truth) · llm_and_merge
```

**Adding an insurer is a data edit, not a code change.** Adding a QMS field is one `FieldSpec`
plus one schema attribute.

| Need | Chosen | Why not the alternative |
| --- | --- | --- |
| PDF text | **PyMuPDF** | `sort=True` is exactly the layout fix required |
| PDF tables | **pdfplumber** | camelot needs Ghostscript — a system dep I won't force on a reviewer |
| Schema | **Pydantic v2** | Validation *and* a free published JSON Schema |
| LLM access | **plain `requests`** | LangChain is an enormous dependency to wrap one HTTP POST |
| CLI | **argparse** | Stdlib; three subcommands don't justify Typer |
| Retrieval | **lexical cue overlap** | Embeddings are less precise on closed jargon, and add a model + vector store |

Deliberately **not** used: LangChain/LlamaIndex, a vector DB, a fine-tuned model, a frontend.

---

## Assumptions

1. **"Previous year's details" = the period printed on these documents.** The samples are
   expiring/renewing policies, so the period on the document *is* the prior-year period.
2. **Dates are day-first** (`02/06/2022` = 2 June 2022), per Indian convention.
3. **Per-member sum insured is a multiple of ₹5,000** — this is what separates a real tier from
   the per-life premium values in the same rate table (`25,505.46`).
4. **A per-member sum insured is between ₹25,000 and ₹1 crore**; anything larger near a
   sum-insured label is the group aggregate.
5. **Metro / non-metro maternity:** no sample differentiates, so the single stated limit is
   reported in both columns with a note. More useful than two blanks, more honest than inventing
   a split.
6. **Gross premium comes from the policy schedule, not a payment receipt** (see the Niva Bupa
   case above).
7. **In-house claims administration is inferred** when the claims administrator resolves to the
   insurer itself — reported with evidence, not as null.
8. **Total lives is derived** (`employees + dependents`) when the document leaves the total in an
   unlabelled row. Marked `source: "derived"`.
9. **The document is the sole source of truth.** No market defaults, no "typical" values.

## Known limitations

- **OCR is implemented but never exercised** — all five samples have text layers, and `tesseract`
  was not installed in the build environment.
- **Fields absent from the samples have unverified cue lists** — AYUSH, LGBTQ+, live-in partner,
  organ donor, air ambulance, surrogacy, vaccination, pharmacy discount, annual health check-up.
- **Spouse / child / parent head-count breakdowns are not in any sample**, so they return
  `not_found` rather than being split by guesswork.
- **Only three insurers were available to test against.** The registry ships 28; 25 are
  unverified against a real document.
- **Documents are 3–6 pages, not the 50–80 the brief describes.** Chunked retrieval is designed
  for length but unproven on a long policy.
- **Scoring weights are hand-set, not learned.** Five documents is not enough to fit them.
- **LLM output is non-deterministic** — field-level `source`/`confidence` can shift slightly
  between hybrid runs. All three tested models still reach 136/136; the rule-only path is fully
  deterministic.
- **The REST endpoint is unauthenticated** and intended for local use only.

## Next steps

- Ground truth across more insurers, to turn the accuracy table into a real regression suite.
- A review UI over `needs_review` fields — the flags already exist, so the human-in-the-loop step
  is cheap.
- Endorsement handling: policies get amended mid-term and an endorsement should override the base
  schedule.

---

## License

MIT.
