"""
=============================================================================
 MedX-ES  ::  knowledge_base.py
-----------------------------------------------------------------------------
 Course    : 2311304 - Artificial Intelligence and Data Science Lab (Sem VI)
 Assignment: No. 6 - Mini Project (Expert System - Hospitals & Medical domain)
 CO        : CO5
-----------------------------------------------------------------------------
 This file is the KNOWLEDGE BASE of the expert system.
 A knowledge base = the domain expert's knowledge, written as IF-THEN rules.

 Rule format :  ( [list of premise symptoms] , certainty_factor )
 Certainty factor (CF) lies in [-1, +1]  (MYCIN convention):
     +1.0  -> the conclusion is definitely true
      0.0  -> no evidence either way
     -1.0  -> the conclusion is definitely false
 A rule fires only when ALL its premise symptoms are present; the strength
 of the fired rule = min(CF of each premise) * CF of the rule.

 NOTE FOR THE LAB JOURNAL: in a real project these rules are written by a
 doctor / taken from standard clinical textbooks, and each rule carries a
 `source:` field citing the reference. The values below are illustrative
 teaching values.
=============================================================================
"""

# ---------------------------------------------------------------------------
# 1. SYMPTOMS
#    Every symptom has a plain-language question, used by the questionnaire.
# ---------------------------------------------------------------------------

g_dicSymptomQuestions = {
    "high_fever":        "Fever (temperature above 100 F / 38 C)?",
    "chills":            "Chills or shivering?",
    "sweating":          "Unusual sweating (especially at night)?",
    "severe_headache":   "Severe headache?",
    "retro_orbital_pain":"Pain behind the eyes?",
    "severe_joint_pain": "Severe joint or muscle pain?",
    "body_ache":         "Generalised body ache?",
    "skin_rash":         "Skin rash?",
    "nausea":            "Feeling nauseated?",
    "vomiting":          "Vomiting?",
    "abdominal_pain":    "Abdominal (stomach) pain?",
    "diarrhoea":         "Loose motions / diarrhoea?",
    "cough":             "Cough?",
    "sore_throat":       "Sore throat?",
    "runny_nose":        "Runny or blocked nose?",
    "sneezing":          "Sneezing?",
    "chest_pain":        "Chest pain?",
    "breathlessness":    "Shortness of breath?",
    "fatigue":           "Unusual tiredness / weakness?",
    "loss_of_appetite":  "Loss of appetite?",
    "platelet_drop":     "Has a blood test shown a low platelet count?",
    "burning_urination": "Burning sensation while passing urine?",
    "frequent_urination":"Frequent urination?",
}

# ordered list of every symptom the engine can reason about
g_lstAllSymptoms = list(g_dicSymptomQuestions.keys())

# ---------------------------------------------------------------------------
# 2. PRODUCTION RULES  (disease  ->  list of rules)
#    The first rules of each disease use 3-4 premises (strong evidence);
#    the last rule of each disease uses 2 premises (weak / partial evidence),
#    so that the certainty factor correctly drops when information is missing.
# ---------------------------------------------------------------------------

g_dicDiseaseRules = {

    "Dengue": [
        (["high_fever", "severe_headache", "retro_orbital_pain"], 0.80),
        (["high_fever", "severe_joint_pain", "skin_rash"],        0.70),
        (["high_fever", "platelet_drop"],                         0.85),
        (["nausea", "vomiting", "abdominal_pain"],                0.40),
        (["high_fever", "severe_headache"],                       0.45),  # weak rule
    ],

    "Malaria": [
        (["high_fever", "chills", "sweating"],                    0.90),
        (["high_fever", "severe_headache", "fatigue"],            0.50),
        (["high_fever", "chills"],                                0.45),  # weak rule
    ],

    "Typhoid": [
        (["high_fever", "abdominal_pain", "loss_of_appetite"],    0.75),
        (["high_fever", "fatigue", "loss_of_appetite"],           0.40),
        (["high_fever", "loss_of_appetite"],                      0.40),  # weak rule
    ],

    "Influenza": [
        (["high_fever", "cough", "sore_throat", "body_ache"],     0.80),
        (["high_fever", "chills", "fatigue"],                     0.60),
        (["high_fever", "sore_throat"],                           0.45),  # weak rule
    ],

    "Common Cold": [
        (["runny_nose", "sneezing", "sore_throat"],               0.85),
        (["cough", "runny_nose"],                                 0.45),
        (["runny_nose", "sneezing"],                              0.55),  # weak rule
    ],

    "Pneumonia": [
        (["cough", "chest_pain", "breathlessness"],               0.90),
        (["high_fever", "cough", "breathlessness"],               0.75),
        (["cough", "breathlessness"],                             0.60),  # weak rule
    ],

    "Gastroenteritis": [
        (["diarrhoea", "vomiting", "abdominal_pain"],             0.85),
        (["nausea", "diarrhoea", "fatigue"],                      0.50),
        (["diarrhoea", "vomiting"],                               0.70),  # weak rule
    ],

    "Bronchitis": [
        (["cough", "chest_pain", "fatigue"],                      0.70),
        (["cough", "sore_throat", "body_ache"],                   0.45),
        (["cough", "chest_pain"],                                 0.45),  # weak rule
    ],

    "Urinary Tract Infection": [
        (["burning_urination", "frequent_urination"],             0.85),
        (["burning_urination", "abdominal_pain"],                 0.55),
    ],
}

# ---------------------------------------------------------------------------
# 3. ADVICE shown with each diagnosis (the "recommendation" part of the system)
# ---------------------------------------------------------------------------

g_dicDiseaseAdvice = {
    "Dengue":                "Rest and fluids; a platelet count / NS1 test is usually advised.",
    "Malaria":               "A blood smear or rapid antigen test is usually advised.",
    "Typhoid":               "Blood culture / Widal test is usually advised; complete the course of medicine.",
    "Influenza":             "Rest, fluids and paracetamol; see a doctor if breathlessness appears.",
    "Common Cold":           "Usually self-limiting - rest, warm fluids and steam inhalation.",
    "Pneumonia":             "Please see a doctor soon - a chest X-ray may be needed.",
    "Gastroenteritis":       "Oral rehydration solution (ORS) is important; see a doctor if dehydrated.",
    "Bronchitis":            "Avoid smoke/dust; see a doctor if cough lasts more than 2-3 weeks.",
    "Urinary Tract Infection":"Drink plenty of water; a urine test and doctor's consultation are advised.",
}

# ---------------------------------------------------------------------------
# 4. DISCLAIMER printed with every result
# ---------------------------------------------------------------------------

g_strDisclaimer = ("DISCLAIMER: MedX-ES is an educational decision-SUPPORT prototype. "
                   "It is NOT a doctor and must never be used for real treatment decisions.")
