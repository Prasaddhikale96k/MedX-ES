# MedX-ES — Journal Write-up (ready to copy into the lab journal)

**Title:** MedX-ES: An Explainable Hybrid Expert System for Multi-Disease Clinical Decision Support

**Assignment:** No. 6 — Mini Project: Expert System (Hospitals and medical facilities)
**Course:** 2311304 — Artificial Intelligence and Data Science Lab · **CO:** CO5

---

## 1. Problem statement

Build an expert system that accepts a patient's symptoms (through a questionnaire or free
text in English/Hindi/Marathi), reasons over a clinical knowledge base of IF–THEN rules
with certainty factors, and produces a ranked list of likely diseases with a confidence
score, an explanation of the rules that fired, and basic advice — while refusing to
diagnose when the evidence is insufficient.

## 2. Theory in brief (points to write)

- **Expert system** = knowledge base + inference engine + explanation facility + UI;
  it mimics a human domain expert (classic example: **MYCIN** for medical diagnosis,
  using IF–THEN rules with **certainty factors** and **backward chaining**).
- **Certainty factor (CF)** in [−1, +1]: +1 definitely true, 0 no evidence, −1 definitely
  false. Combination: `CF1 + CF2(1−CF1)` for agreeing evidence;
  `(CF1+CF2)/(1−min(|CF1|,|CF2|))` for conflicting evidence. A rule's strength =
  `min(CF of premises) × CF of rule`.
- **Forward chaining** = data-driven reasoning: start from facts, fire every matching
  rule, collect conclusions. (Backward chaining, goal-driven, is the alternative.)
- **Hybrid (neuro-symbolic) AI**: symbolic rules give transparency and auditability;
  machine learning gives statistical priors from data. Modern surveys (IEEE TNNLS 2024;
  ACM TKDD 2024) and medical KG work (J. Web Semantics 2025) recommend exactly this blend.
- **Abstention / human-in-the-loop**: state-of-the-art clinical decision-support systems
  (e.g. CLIN-LLM, 2025) flag low-confidence cases for expert review instead of forcing an
  answer — MedX-ES implements the same safety idea with a confidence threshold.
- **Naive Bayes**: simple probabilistic classifier; P(disease|symptoms) computed from
  symptom counts with Laplace smoothing. Implemented from scratch (standard library).
- **Dataset**: "Disease Prediction Using Machine Learning" (Kaggle, kaushil268) —
  4,920 records, 41 diseases, 132 binary symptom columns; 4,920 used for training and
  42 held-out records for testing. Measured Naive Bayes test accuracy: **92.9 %**
  (39/42). The model scores only the reported symptoms, because in this dataset the
  absence of a symptom is not informative.

## 3. Algorithm

(see README section 6 — the 7-step algorithm; reproduce it in the journal)

## 4. Flowchart

(see README section 7 — redraw in draw.io for the journal)

## 5. Test cases and results

| # | Input (findings) | Expected | Actual | Result |
|---|---|---|---|---|
| T01 | fever, severe headache, pain behind eyes, joint pain, rash, low platelets | Dengue | Dengue | PASS |
| T02 | fever, chills, sweating, headache, fatigue | Malaria | Malaria | PASS |
| T03 | fever, abdominal pain, loss of appetite, fatigue | Typhoid | Typhoid | PASS |
| T04 | fever, cough, sore throat, body ache, chills | Influenza | Influenza | PASS |
| T05 | runny nose, sneezing, sore throat, cough | Common Cold | Common Cold | PASS |
| T06 | cough, chest pain, breathlessness, fever | Pneumonia | Pneumonia | PASS |
| T07 | diarrhoea, vomiting, abdominal pain | Gastroenteritis | Gastroenteritis | PASS |
| T08 | cough, chest pain, fatigue, sore throat | Bronchitis | Bronchitis | PASS |
| T09 | burning urination, frequent urination, abdominal pain | Urinary Tract Infection | Urinary Tract Infection | PASS |
| T10 | fever, sore throat only (partial info) | Influenza (low CF) | Influenza | PASS |
| T11 | fever only (too little info) | ABSTAIN | ABSTAIN | PASS |
| T12 | fatigue, loss of appetite (non-specific) | ABSTAIN | ABSTAIN | PASS |

**Summary: 12/12 (100 %) test cases passed.** Output text: `sample_output.txt`.

## 6. Conclusions

- The expert system correctly diagnoses 9 classic disease presentations and, importantly,
  **abstains** (refuses to guess) on 3 under-specified cases — demonstrating safe,
  human-in-the-loop behaviour rather than over-confident answers. (In those cases the
  ML prior alone would have guessed with high confidence — shown on screen — which is
  exactly why the rule base gates the final answer.)
- The explanation facility makes every answer auditable: each conclusion lists the exact
  rules that fired and the certainty-factor arithmetic — the key advantage of symbolic
  expert systems over black-box ML.
- Blending rule-based CFs (75 %) with an ML prior trained on the real 4,920-record
  Kaggle dataset (25 %) keeps the system explainable while adding data-driven priors —
  the neuro-symbolic design recommended by current research.
- **Limitations:** teaching-sized rule base (9 of the dataset's 41 diseases); keyword-based
  multilingual lexicon (full version: spaCy NER); Naive Bayes written from scratch for a
  dependency-free, easy-to-explain implementation (full version: scikit-learn).
- **Future work:** extend the rule base to all 41 diseases, add a disease–symptom
  knowledge graph for differential-diagnosis explanations, replace the lexicon with
  proper NER, and run an expert-agreement evaluation with doctors.

## 7. Viva one-liners (memorise these)

1. *Why certainty factors and not plain probabilities?* — CFs model belief change under
   partial evidence with simple, explainable algebra (MYCIN tradition).
2. *Forward vs backward chaining?* — Forward is used here (facts → conclusions); backward
   would be used for "why do you ask this question?" goal-driven queries.
3. *What does the ML module add, and how is it combined?* — statistical priors learned
   from the real dataset; `Final = 0.75·CF_rule + 0.25·P_ML`.
4. *Why can the ML module never produce a diagnosis on its own?* — the rule base gates:
   only diseases with at least one fired rule receive a final score; ML only re-ranks them.
5. *Why does the system sometimes refuse to answer?* — abstention gate: confidence below
   0.25 (or no rule fired) → escalate to a human; safer than a wrong guess.
6. *What accuracy did the ML part achieve?* — Naive Bayes, trained on 4,920 records and
   tested on 42 held-out records: 92.9 % (39/42) test accuracy.
7. *Why hybrid and not a neural network alone?* — medicine needs traceable justification;
   pure ML is a black box, pure rules miss statistical patterns.

## 8. References

1. Huang, X. et al. (2021). *A Generic Knowledge Based Medical Diagnosis Expert System.* iiWAS2021, ACM.
2. Deng, Y. (2024). *An Expert System for Health Diagnosis Based on NLP and Reasoning Engine.* Advances in Computer, Signals and Systems 8:126–133.
3. DeLong, L.N., Mir, R.F., Fleuriot, J.D. (2024). *Neurosymbolic AI for Reasoning over Knowledge Graphs: A Survey.* IEEE TNNLS 36(5):7822–7842.
4. Cheng, K. et al. (2024). *Neural-Symbolic Methods for Knowledge Graph Reasoning: A Survey.* ACM TKDD 18(9):225.
5. Vidal, M.-E. et al. (2025). *Integrating knowledge graphs with symbolic AI: the path to interpretable hybrid AI systems in medicine.* J. Web Semantics 84:100856.
6. *CLIN-LLM: A Safety-Constrained Hybrid Framework for Clinical Diagnosis and Treatment Generation.* arXiv:2510.22609 (2025).
7. Dataset: Kaggle — *Disease Prediction Using Machine Learning* (kaushil268).
8. Columbia University — *DiseaseSymptomKB*.
