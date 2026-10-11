"""
=============================================================================
 MedX-ES  ::  ml_prior.py
-----------------------------------------------------------------------------
 The MACHINE-LEARNING module of the hybrid expert system.

 It learns a statistical prior  P(disease | symptoms)  from a REAL dataset
 using a Bernoulli Naive Bayes classifier written from scratch with the
 Python standard library only (no external packages needed).

 DATASET (original, open source):
    "Disease Prediction Using Machine Learning"  (Kaggle, user kaushil268)
    https://www.kaggle.com/datasets/kaushil268/diseaseprediction-using-machine-learning
    4,920 patient records - 41 diseases - 132 binary symptom columns.
    Local copies used by this program:
        data/Training.csv   (4,920 rows - used for training)
        data/Testing.csv    (  42 rows - used for evaluation)

 Why a hybrid?  The rule base is transparent but hand-written; the ML model
 learns statistical patterns from real data but is a black box. Blending
 both (see inference_engine.fnHybridScore) gives explainability + learning -
 the "neuro-symbolic" idea from current research.

 DESIGN RULE (important, explain in the viva):
    The rule base GATES the diagnosis - a disease can only be reported if at
    least one rule fired for it. The ML prior REFINES the ranking among the
    rule-supported diseases. This keeps every answer explainable and makes
    the abstention gate meaningful (ML alone can never produce a diagnosis).
=============================================================================
"""

import csv
import math
import os
import random
from collections import defaultdict

from knowledge_base import g_dicDiseaseRules, g_lstAllSymptoms

random.seed(42)   # fixed seed -> reproducible results for the journal

# ---------------------------------------------------------------------------
# paths of the local dataset copies
# ---------------------------------------------------------------------------
g_strTrainingCsv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "data", "Training.csv")
g_strTestingCsv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "data", "Testing.csv")

# ---------------------------------------------------------------------------
# mapping:  our knowledge-base symptom  ->  dataset column name
# (symptoms without a matching dataset column are simply ignored by the ML
#  module - the rule engine still uses them)
# ---------------------------------------------------------------------------
g_dicSymptomToDatasetColumn = {
    "high_fever":         "high_fever",
    "chills":             "chills",
    "sweating":           "sweating",
    "severe_headache":    "headache",
    "retro_orbital_pain": "pain_behind_the_eyes",
    "severe_joint_pain":  "joint_pain",
    "body_ache":          "muscle_pain",
    "skin_rash":          "skin_rash",
    "nausea":             "nausea",
    "vomiting":           "vomiting",
    "abdominal_pain":     "abdominal_pain",
    "diarrhoea":          "diarrhoea",
    "cough":              "cough",
    "sore_throat":        "throat_irritation",
    "runny_nose":         "runny_nose",
    "sneezing":           "continuous_sneezing",
    "chest_pain":         "chest_pain",
    "breathlessness":     "breathlessness",
    "fatigue":            "fatigue",
    "loss_of_appetite":   "loss_of_appetite",
    "platelet_drop":      None,                     # not in the dataset
    "burning_urination":  "burning_micturition",
    "frequent_urination": "continuous_feel_of_urine",
}


def fnNormaliseDisease(strDisease):
    """Disease names are matched case-insensitively across modules."""
    return strDisease.strip().lower()


# ---------------------------------------------------------------------------
# 1. LOADING THE ORIGINAL DATASET
# ---------------------------------------------------------------------------

def fnLoadCsvDataset(strPath):
    """Read one dataset CSV.

    Returns (lstSymptomColumns, lstSamples) where every sample is
    (set_of_present_symptoms, normalised_disease_name).
    """
    lstSamples = []
    with open(strPath, newline="", encoding="utf-8") as fileCsv:
        dicReader = csv.DictReader(fileCsv)
        lstColumns = [c for c in dicReader.fieldnames if c and c != "prognosis"]
        for dicRow in dicReader:
            setPresent = {c for c in lstColumns if dicRow.get(c) == "1"}
            lstSamples.append((setPresent, fnNormaliseDisease(dicRow["prognosis"])))
    return lstColumns, lstSamples


# ---------------------------------------------------------------------------
# 2. TRAINING  (Bernoulli Naive Bayes, Laplace smoothing)
# ---------------------------------------------------------------------------

def fnTrainNaiveBayes(lstSamples=None):
    """Train on the ORIGINAL dataset and return the model dictionary.

    If the dataset files are missing, fall back to synthetic data generated
    from the knowledge base, so the program always runs.
    """
    strSource = "synthetic (generated from the knowledge base)"
    if lstSamples is None:
        if os.path.exists(g_strTrainingCsv):
            _, lstSamples = fnLoadCsvDataset(g_strTrainingCsv)
            strSource = f"original Kaggle dataset ({g_strTrainingCsv})"
        else:
            lstSamples = fnGenerateSyntheticData()
            print(f"  [note: {g_strTrainingCsv} not found - using synthetic data]")

    dicSymptomCount = defaultdict(lambda: defaultdict(int))  # disease -> symptom -> count
    dicDiseaseCount = defaultdict(int)                       # disease -> cases

    for (setSymptoms, strDisease) in lstSamples:
        dicDiseaseCount[strDisease] += 1
        for strSymptom in setSymptoms:
            dicSymptomCount[strDisease][strSymptom] += 1

    return {
        "dicSymptomCount": {d: dict(s) for d, s in dicSymptomCount.items()},
        "dicDiseaseCount": dict(dicDiseaseCount),
        "iTotalCases": len(lstSamples),
        "strSource": strSource,
    }


# ---------------------------------------------------------------------------
# 3. PREDICTION
# ---------------------------------------------------------------------------

def fnPredictMlProbabilities(dicModel, dicPatient):
    """Return {normalised_disease: probability} for one patient.

    Naive Bayes scoring over the REPORTED symptoms only:

        score(d) = log P(d) + sum over reported symptoms s of log P(s|d)

    Only reported symptoms are scored because in this dataset the ABSENCE of
    a symptom is not informative (most diseases list only a handful of the
    132 possible symptoms), so absent-symptom penalties would distort the
    result - e.g. a single matching symptom would otherwise make a disease
    that never shows any other symptom look very likely.

    Scores are converted to probabilities with a softmax so they sum to 1.
    A knowledge-base symptom counts as reported when the patient's CF for it
    is above 0.5; it is translated to its dataset column first.
    """
    dicSymptomCount = dicModel["dicSymptomCount"]
    dicDiseaseCount = dicModel["dicDiseaseCount"]
    iTotalCases = dicModel["iTotalCases"]

    # translate our symptoms into the dataset's symptom vocabulary
    # (keys that are already dataset columns are used as they are, so the
    #  same function works both for consultations and for test-set evaluation)
    lstAllColumns = set()
    for dicCounts in dicSymptomCount.values():
        lstAllColumns.update(dicCounts.keys())

    setPresentColumns = set()
    for strSymptom, dCf in dicPatient.items():
        if dCf > 0.5:
            if strSymptom in lstAllColumns:                 # already a dataset column
                setPresentColumns.add(strSymptom)
            else:                                           # knowledge-base symptom
                strColumn = g_dicSymptomToDatasetColumn.get(strSymptom)
                if strColumn:
                    setPresentColumns.add(strColumn)

    dicScores = {}
    for strDisease, iCases in dicDiseaseCount.items():
        dLogScore = math.log(iCases / iTotalCases)          # P(disease)
        for strColumn in setPresentColumns:                 # reported symptoms only
            dProb = ((dicSymptomCount[strDisease].get(strColumn, 0) + 1)
                     / (iCases + 2))                        # Laplace smoothing
            dLogScore += math.log(dProb)
        dicScores[strDisease] = dLogScore

    dMax = max(dicScores.values())
    dicExp = {d: math.exp(s - dMax) for d, s in dicScores.items()}
    dSum = sum(dicExp.values())
    return {d: round(v / dSum, 4) for d, v in dicExp.items()}


# ---------------------------------------------------------------------------
# 4. EVALUATION on the dataset's own test file (real accuracy number)
# ---------------------------------------------------------------------------

def fnEvaluateOnTestSet(dicModel):
    """Train-on-Training / test-on-Testing accuracy of the Naive Bayes model."""
    if not os.path.exists(g_strTestingCsv):
        return None
    _, lstTestSamples = fnLoadCsvDataset(g_strTestingCsv)
    iCorrect = 0
    for (setPresent, strDisease) in lstTestSamples:
        dicPatient = {s: 1.0 for s in setPresent}
        dicProbs = fnPredictMlProbabilities(dicModel, dicPatient)
        if dicProbs and max(dicProbs, key=dicProbs.get) == strDisease:
            iCorrect += 1
    return iCorrect / len(lstTestSamples)


# ---------------------------------------------------------------------------
# 5. FALLBACK: synthetic data (only used if the CSVs are missing)
# ---------------------------------------------------------------------------

def fnGenerateSyntheticData(iSamplesPerDisease=80):
    """Create (symptom_set, disease) samples from the rule base."""
    lstData = []
    for strDisease, lstRules in g_dicDiseaseRules.items():
        for _ in range(iSamplesPerDisease):
            setSymptoms = set()
            for (lstPremises, _) in lstRules:
                if random.random() < 0.75:
                    for strSymptom in lstPremises:
                        if random.random() > 0.10:
                            setSymptoms.add(strSymptom)
            for strSymptom in g_lstAllSymptoms:
                if random.random() < 0.03:
                    setSymptoms.add(strSymptom)
            lstData.append((setSymptoms, fnNormaliseDisease(strDisease)))
    return lstData
