# MedX-ES — Explainable Hybrid Expert System

**Assignment No. 6 — Mini Project (Expert System)**
Course: **2311304 — Artificial Intelligence and Data Science Lab**, T.Y. B.Tech AI & DS, Semester VI (Pattern 2023)
Course Outcome: **CO5** — *Make use of suitable AI algorithms to solve problems*
Domain chosen: **Hospitals and medical facilities**

---

## 1. What the project does (one line)

MedX-ES is a small medical **expert system**: you tell it your symptoms (by questionnaire
or by free text in English / Hindi / Marathi) and it reasons with doctor-written
IF–THEN rules to suggest the **most likely disease with a confidence score, a full
explanation of which rules fired, and advice** — and it **refuses to guess** when the
evidence is too weak.

## 2. Why it is a hybrid ("neuro-symbolic") expert system

| Component | Module | Role |
|---|---|---|
| Knowledge base | `knowledge_base.py` | 9 diseases, 29 IF–THEN rules with **certainty factors** |
| Inference engine | `inference_engine.py` | **Forward chaining** + MYCIN certainty-factor algebra + **abstention gate** |
| Explanation facility | `inference_engine.py` (`fnExplainResult`) | shows exactly which rules fired and how the CF was combined |
| ML prior | `ml_prior.py` | **Naive Bayes** trained on the **original Kaggle dataset** (4,920 records, 41 diseases; 92.9 % test accuracy); blended with the rule CF (75 % rules / 25 % ML) |
| NLP front-end | `symptom_lexicon.py` | free text → structured symptoms (English + Hindi + Marathi keywords) |
| User interface | `main.py` | menu-driven CLI with input validation |

**Design rule (mention this in the viva):** the rule base **gates** the diagnosis — a
disease can only be reported if at least one rule fired for it; the ML prior only
**re-ranks** the rule-supported diseases. So every answer stays explainable, and the ML
model alone can never produce a diagnosis.

## 3. Dataset used (original, open source)

**"Disease Prediction Using Machine Learning"** — Kaggle (user `kaushil268`)
<https://www.kaggle.com/datasets/kaushil268/diseaseprediction-using-machine-learning>

* 4,920 patient records · 41 diseases · 132 binary symptom columns
* Local copies (already inside this folder):
  * `data/Training.csv` — 4,920 rows, used for **training**
  * `data/Testing.csv` — 42 rows, used for **evaluation**
* Measured result: **Naive Bayes test accuracy = 92.9 %** (39/42) on the held-out test
  file (printed automatically when the program starts). The model scores only the reported
  symptoms, because in this dataset the absence of a symptom is not informative.

The knowledge base's 23 symptoms are mapped to the dataset's 132 symptom columns in
`ml_prior.py` (`g_dicSymptomToDatasetColumn`), e.g. `severe_headache → headache`,
`retro_orbital_pain → pain_behind_the_eyes`, `sneezing → continuous_sneezing`.

## 4. How to run (needs only Python 3 — no installation)

**Option A — the web app (premium Neubrutalism UI, animations):**

```bash
cd MedX-ES
python3 server.py          # then open http://localhost:8000
```

**Option B — the command-line program (used for the journal outputs):**

```bash
cd MedX-ES
python3 main.py            # menu: demo / questionnaire / free-text
python3 test_cases.py      # runs the 12 journal test cases
```

Web-UI features: grouped symptom chips + live search, free-text entry in
English/Hindi/Marathi, one-click demo cases, ranked diagnosis cards with
animated confidence bars, a "Why this diagnosis?" explanation accordion
(fired rules + certainty-factor arithmetic), count-up confidence numbers,
confetti on a confident match, and a styled abstention card that also shows
what the ML prior alone would have guessed (and why that is *not* a diagnosis).

## 5. Project structure

```
MedX-ES/
├── server.py             # web server (stdlib only) -> serves the UI + API
├── main.py               # command-line interface (menu, input validation)
├── knowledge_base.py     # the RULES written by the "domain expert"
├── inference_engine.py   # forward chaining + certainty factors + explanations
├── ml_prior.py           # Naive Bayes ML module trained on the original dataset
├── symptom_lexicon.py    # free-text -> symptoms (EN / Hindi / Marathi)
├── test_cases.py         # 12 test cases with expected results
├── frontend/
│   └── index.html        # the Neubrutalism web UI (HTML + CSS + JS, one file)
├── data/
│   ├── Training.csv      # original Kaggle dataset - training split (4,920 rows)
│   └── Testing.csv       # original Kaggle dataset - test split (42 rows)
├── sample_output.txt     # captured output for the journal
├── JOURNAL_WRITEUP.md    # ready-to-copy journal content (theory, tests, viva)
├── START_HERE.txt        # 2-step quick-start card
└── README.md             # this file
```

## 6. Algorithm (for the journal)

```
1.  Collect patient findings -> dicPatient {symptom: certainty 0..1}
2.  Forward chaining:
      for each disease, for each rule of that disease
          strength = min(CF of premises) * CF(rule)
          if strength > 0 : rule fires, remember it
      CF(disease) = combine all fired strengths with the CF algebra
3.  ML prior: P(disease | symptoms) from the Naive Bayes model trained on the
      original Kaggle dataset (4,920 records, 41 diseases)
4.  Final(disease) = 0.75 * CF(disease) + 0.25 * P_ML(disease)
      (only diseases with CF > 0 get a final score - the rule base gates)
5.  Rank the rule-supported diseases by Final score
6.  Abstention gate: if no rule fired, or best Final < 0.25
      -> answer "insufficient evidence, consult a doctor"
7.  Otherwise print the top disease, its confidence, the fired rules
    (explanation) and the advice + disclaimer
```

## 7. Architecture / flowchart (ASCII)

```
   symptoms (questionnaire / free text)
                 |
                 v
      +--------------------+        +-------------------+
      |  NLP front-end     | -----> |  WORKING MEMORY   |
      | (symptom_lexicon)  |        | dicPatient        |
      +--------------------+        +---------+---------+
                                               |
                 +-----------------------------+
                 v                             v
      +----------------------+      +------------------------+
      | INFERENCE ENGINE     |      | ML PRIOR               |
      | forward chaining     |      | Naive Bayes trained on |
      | + certainty factors  |      | the Kaggle dataset     |
      +----------+-----------+      +-----------+------------+
                 |                              |
                 +--------------+---------------+
                                v
            Final = 0.75*CF + 0.25*P_ML   (rule base gates)
                                |
                    +-----------+-----------+
                    v                       v
            confidence >= 0.25       confidence < 0.25
                    |                       |
                    v                       v
        diagnosis + explanation      ABSTAIN -> see a doctor
        + advice + disclaimer
```

## 8. Test results (see `sample_output.txt`)

**12 / 12 test cases pass (100 %)** — 9 classic presentations are diagnosed correctly,
and 3 under-specified cases correctly trigger the **abstention gate** (the system says
"not enough evidence" instead of guessing, even though the ML prior alone would have
guessed with high confidence — this is shown on screen and is a good viva talking point).

## 9. Lab-journal mapping (as required by the syllabus)

| Journal section | Where it comes from |
|---|---|
| Title / problem statement | Section 1 above + `JOURNAL_WRITEUP.md` |
| Theory in brief | expert systems, production rules, certainty factors, forward chaining, hybrid AI (MYCIN, neuro-symbolic surveys), Naive Bayes |
| Algorithm | Section 6 above |
| Flowchart | Section 7 above |
| Test cases | `test_cases.py` → 12 cases with expected vs actual results |
| Conclusions | see `JOURNAL_WRITEUP.md` |
| Program codes + sample outputs | this folder + `sample_output.txt` |
| Coding standards | Hungarian notation (`dicPatient`, `lstRules`, `dCfFinal`, `bAbstain`), comments, PEP8 indentation, input validation |

## 10. Honest scope (say this in the viva)

- Educational **decision-support** prototype — not for real clinical use.
- The rule base (9 diseases) is a teaching subset of the dataset's 41 diseases; the ML
  module is trained on the real 4,920-record dataset.
- The multilingual lexicon is keyword-based; a full version would use spaCy NER.
- The Naive Bayes model is written from scratch (standard library) to keep the project
  dependency-free and easy to explain; scikit-learn can replace it without changing any
  other module.
