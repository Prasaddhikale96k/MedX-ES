"""
=============================================================================
 MedX-ES  ::  main.py      (command-line interface)
-----------------------------------------------------------------------------
 Run:   python3 main.py
 Menu:  1. Demo cases             - three scripted patients, full explanation
        2. Interactive consultation - questionnaire (yes / no / unknown)
        3. Free-text entry        - type symptoms in English / Hindi / Marathi
        4. Exit

 All input is validated, and every result carries a disclaimer, as required
 by the laboratory guidelines (input validation, coding standards).
=============================================================================
"""

from knowledge_base import (g_dicSymptomQuestions, g_strDisclaimer)
from inference_engine import fnConsult, fnExplainResult
from ml_prior import fnTrainNaiveBayes, fnPredictMlProbabilities, fnEvaluateOnTestSet
from symptom_lexicon import fnExtractSymptomsFromText


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def fnPrintReport(dicResult):
    """Print the ranked diagnosis table, the explanation and the disclaimer."""
    if not dicResult["lstRanking"]:
        # no rule fired -> nothing to diagnose; show ML-only info for transparency
        print("\n  No rule in the knowledge base fired for these findings.")
        lstMlTop = sorted(dicResult["dicMlProbs"].items(),
                          key=lambda kv: kv[1], reverse=True)[:3]
        if lstMlTop:
            print("  (For transparency, the ML prior alone would have guessed:")
            for strDisease, dProb in lstMlTop:
                print(f"     {strDisease:<28}{dProb:>8}")
            print("   These are NOT diagnoses - the rule base gates the answer.)")
        print("\n  " + "-" * 62)
        print("  DIAGNOSIS TABLE: empty (no rule-supported hypothesis)")
        print("  " + "-" * 62)
    else:
        print("\n  " + "-" * 62)
        print("  DIAGNOSIS TABLE (ranked by final confidence)")
        print("  " + "-" * 62)
        print(f"  {'Disease':<26}{'Rule CF':>10}{'ML prob':>10}{'Final':>10}")
        for strDisease, dFinal in dicResult["lstRanking"][:5]:     # top 5 only
            dCf = dicResult["dicCfScores"].get(strDisease, 0.0)
            dMl = dicResult["dicMlProbs"].get(strDisease.strip().lower(), 0.0)
            print(f"  {strDisease:<26}{dCf:>10}{dMl:>10}{dFinal:>10}")
        print("  (Only rule-supported diseases are listed: the rule base gates")
        print("   the diagnosis, the ML prior re-ranks the supported diseases.)")

    if dicResult["bAbstain"]:
        fnExplainResult(dicResult)
    else:
        strTop, dTop = dicResult["lstRanking"][0]
        if dTop >= 0.60:
            print(f"\n  RESULT: LIKELY {strTop}  (confidence {dTop})")
        else:
            print(f"\n  RESULT: POSSIBLE {strTop}  (confidence {dTop}) - "
                  f"please see a doctor to confirm.")
        fnExplainResult(dicResult)

    print(f"\n  {g_strDisclaimer}")


def fnAskYesNoUnknown(strQuestion):
    """Ask one validated yes/no/unknown question."""
    while True:
        try:
            strAnswer = input(f"  {strQuestion} [y/n/u]: ").strip().lower()
        except EOFError:                       # piped input ended
            return "u"
        if strAnswer in ("y", "n", "u"):
            return strAnswer
        print("    Invalid input - please enter y (yes), n (no) or u (unknown).")


def fnConsultWithEngine(dicPatient, dicModel):
    """Run one consultation through the hybrid engine."""
    dicMl = fnPredictMlProbabilities(dicModel, dicPatient)
    return fnConsult(dicPatient, dicMl)


# ---------------------------------------------------------------------------
# mode 1 - demo cases
# ---------------------------------------------------------------------------

def fnRunDemoCases(dicModel):
    print("\n### DEMO CASES ###")
    lstDemos = [
        ("Demo 1 - classic dengue picture",
         {"high_fever": 1.0, "severe_headache": 1.0, "retro_orbital_pain": 1.0,
          "severe_joint_pain": 1.0, "skin_rash": 1.0, "platelet_drop": 1.0,
          "fatigue": 0.8}),
        ("Demo 2 - classic malaria picture",
         {"high_fever": 1.0, "chills": 1.0, "sweating": 1.0,
          "severe_headache": 1.0, "fatigue": 1.0}),
        ("Demo 3 - weak evidence (abstention gate)",
         {"fatigue": 0.6, "loss_of_appetite": 0.6}),
    ]
    for strName, dicPatient in lstDemos:
        print("\n" + "=" * 70)
        print(f"CASE: {strName}")
        print("=" * 70)
        print(f"  Reported findings: "
              f"{ {k: v for k, v in dicPatient.items() if v > 0} }")
        fnPrintReport(fnConsultWithEngine(dicPatient, dicModel))


# ---------------------------------------------------------------------------
# mode 2 - interactive questionnaire
# ---------------------------------------------------------------------------

def fnInteractiveConsultation(dicModel):
    print("\n### INTERACTIVE CONSULTATION ###")
    print("Answer each question: y = yes, n = no, u = unknown.\n")
    dicPatient = {}
    for strSymptom, strQuestion in g_dicSymptomQuestions.items():
        strAnswer = fnAskYesNoUnknown(strQuestion)
        dicPatient[strSymptom] = {"y": 1.0, "n": 0.0, "u": 0.0}[strAnswer]
    fnPrintReport(fnConsultWithEngine(dicPatient, dicModel))


# ---------------------------------------------------------------------------
# mode 3 - free-text entry (NLP front-end)
# ---------------------------------------------------------------------------

def fnFreeTextConsultation(dicModel):
    print("\n### FREE-TEXT SYMPTOM ENTRY ###")
    print("Describe your symptoms in English, Hindi or Marathi, e.g.")
    print("  'two din se bukhar aur sir dard'  |  'fever with chills and sweating'\n")
    try:
        strText = input("  Your symptoms: ")
    except EOFError:
        return
    setSymptoms = fnExtractSymptomsFromText(strText)
    if not setSymptoms:
        print("  Sorry, no known symptoms were recognised in that text. "
              "Please try the questionnaire (option 2).")
        return
    print(f"  Recognised symptoms: {sorted(setSymptoms)}")
    dicPatient = {s: 1.0 for s in setSymptoms}
    fnPrintReport(fnConsultWithEngine(dicPatient, dicModel))


# ---------------------------------------------------------------------------
# main menu
# ---------------------------------------------------------------------------

def fnMain():
    print("=" * 70)
    print(" MedX-ES :: Explainable Hybrid Expert System (educational prototype)")
    print(" Assignment 6 - 2311304 AI & Data Science Lab - CO5")
    print("=" * 70)

    dicModel = fnTrainNaiveBayes()          # train the ML prior on the dataset
    print(f"\n  [ML prior trained on {dicModel['iTotalCases']} patient records "
          f"covering {len(dicModel['dicDiseaseCount'])} diseases]")
    print(f"  [training source: {dicModel['strSource']}]")
    dAccuracy = fnEvaluateOnTestSet(dicModel)
    if dAccuracy is not None:
        print(f"  [Naive Bayes test accuracy on the held-out test file: "
              f"{100.0 * dAccuracy:.1f}%]")

    while True:
        print("\n  MENU")
        print("   1. Demo cases")
        print("   2. Interactive consultation (questionnaire)")
        print("   3. Free-text symptom entry (English / Hindi / Marathi)")
        print("   4. Exit")
        try:
            strChoice = input("  Enter choice [1-4]: ").strip()
        except EOFError:
            break
        if strChoice == "1":
            fnRunDemoCases(dicModel)
        elif strChoice == "2":
            fnInteractiveConsultation(dicModel)
        elif strChoice == "3":
            fnFreeTextConsultation(dicModel)
        elif strChoice == "4":
            print("  Goodbye! " + g_strDisclaimer)
            break
        else:
            print("  Invalid choice - please enter 1, 2, 3 or 4.")


if __name__ == "__main__":
    fnMain()
