"""
=============================================================================
 MedX-ES  ::  symptom_lexicon.py
-----------------------------------------------------------------------------
 The NLP FRONT-END of the expert system.

 Real users do not know symptom IDs - they type free text like
 "two din se bukhar aur sir dard hai" or "fever with chills since morning".
 This module converts such free text into the structured symptom set the
 inference engine understands, using a simple keyword lexicon
(English + Hindi + Marathi transliteration).

 In the full project this is upgraded to proper NER (spaCy / transformer
 models); the interface - text in, symptom set out - stays the same.
=============================================================================
"""

from knowledge_base import g_lstAllSymptoms

# ---------------------------------------------------------------------------
# keyword lexicon : symptom -> list of trigger words (lowercase, substring match)
# ---------------------------------------------------------------------------

g_dicSymptomKeywords = {
    "high_fever":        ["fever", "bukhar", "bukhaar", "taap", "jwar", "temperature",
                          "बुखार", "ताप", "ज्वर"],
    "chills":            ["chill", "thand", "thandi", "rigor", "shiver",
                          "ठंडी", "कंपकंपी", "कंप"],
    "sweating":          ["sweat", "paseena", "pasina", "पसीना"],
    "severe_headache":   ["headache", "sir dard", "sir mein dard", "head pain",
                          "सिर दर्द", "डोके दुखणे", "डोकेदुखी"],
    "retro_orbital_pain":["behind the eye", "eye pain", "aankh ke peeche",
                          "डोळ्यांमागे"],
    "severe_joint_pain": ["joint pain", "joint", "muscle pain", "body pain",
                          "jodon mein dard", "सांधेदुखी"],
    "body_ache":         ["body ache", "body pain", " badan dard", "शरीर दुखणे"],
    "skin_rash":         ["rash", "daane", "dane", "acne-like", "पुरळ", "दाग"],
    "nausea":            ["nausea", "feel like vomiting", "matli", "मळमळ"],
    "vomiting":          ["vomit", "vomiting", "ulti", "ultiyaan", "उलटी"],
    "abdominal_pain":    ["stomach", "abdominal", "pet mein dard", "pet dard",
                          "पोटदुखी", "छाती"],
    "diarrhoea":         ["loose motion", "loose stool", "diarrhoea", "diarrhea",
                          "dast", "जुलाब", "अतिसार"],
    "cough":             ["cough", "khasi", "khansi", "खोकला", "खोक"],
    "sore_throat":       ["sore throat", "throat pain", "gala dard", "gale mein dard",
                          "घसा", "गळदुखी"],
    "runny_nose":        ["runny nose", "blocked nose", "cold", "naak", "zukam",
                          "जुकाम", "नाक वाहणे"],
    "sneezing":          ["sneeze", "sneezing", "chheenk", "छींक"],
    "chest_pain":        ["chest pain", "chest", "seene mein dard", "छातीत दुखी"],
    "breathlessness":    ["breathless", "shortness of breath", "saans", "breathing problem",
                          "स्वास", "ध्वनी"],
    "fatigue":           ["fatigue", "tired", "weakness", "kamzori", "thakan",
                          "थकवा", "कमजोरी"],
    "loss_of_appetite":  ["loss of appetite", "no appetite", "bhookh nahi", "bhook kam",
                          "भूक नाही", "क्षुधा"],
    "platelet_drop":     ["platelet", "low platelet", "प्लेटलेट"],
    "burning_urination": ["burning urination", "burning while urinating", "peshab mein jalan",
                          "मूत्र वेदना"],
    "frequent_urination": ["frequent urination", "urinating often", "baar baar peshab",
                           "वारंवार मूत्र"],
}


def fnExtractSymptomsFromText(strText):
    """Convert free text into a set of recognised symptom IDs.

    Simple, transparent matching: lowercase the text and look for each
    keyword as a substring. Returns a set of symptom IDs found.
    """
    strLower = strText.lower()
    setFound = set()
    for strSymptom in g_lstAllSymptoms:
        for strKeyword in g_dicSymptomKeywords.get(strSymptom, []):
            if strKeyword in strLower:
                setFound.add(strSymptom)
                break
    return setFound
