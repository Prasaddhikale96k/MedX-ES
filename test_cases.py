"""
=============================================================================
 MedX-ES  ::  test_cases.py
-----------------------------------------------------------------------------
 Laboratory assignments require TEST CASES. This file runs 12 standard
 test patients through the expert system and checks the result:

   - cases 1-9  : classic presentations -> the correct disease must rank #1
   - cases 10-12: incomplete evidence  -> the system must ABSTAIN
                  (refuse to guess) or correctly lower its confidence

 Run:  python3 test_cases.py
=============================================================================
"""

from inference_engine import fnConsult
from ml_prior import fnTrainNaiveBayes, fnPredictMlProbabilities

# (test-case name, patient findings, expected top-1 disease or "ABSTAIN")
g_lstTestCases = [
    ("T01  Dengue, classic",              {"high_fever": 1.0, "severe_headache": 1.0,
                                           "retro_orbital_pain": 1.0, "severe_joint_pain": 1.0,
                                           "skin_rash": 1.0, "platelet_drop": 1.0}, "Dengue"),
    ("T02  Malaria, classic",             {"high_fever": 1.0, "chills": 1.0,
                                           "sweating": 1.0, "severe_headache": 1.0,
                                           "fatigue": 1.0}, "Malaria"),
    ("T03  Typhoid, classic",             {"high_fever": 1.0, "abdominal_pain": 1.0,
                                           "loss_of_appetite": 1.0, "fatigue": 1.0}, "Typhoid"),
    ("T04  Influenza, classic",           {"high_fever": 1.0, "cough": 1.0,
                                           "sore_throat": 1.0, "body_ache": 1.0,
                                           "chills": 1.0}, "Influenza"),
    ("T05  Common cold",                  {"runny_nose": 1.0, "sneezing": 1.0,
                                           "sore_throat": 1.0, "cough": 1.0}, "Common Cold"),
    ("T06  Pneumonia",                    {"cough": 1.0, "chest_pain": 1.0,
                                           "breathlessness": 1.0, "high_fever": 1.0}, "Pneumonia"),
    ("T07  Gastroenteritis",              {"diarrhoea": 1.0, "vomiting": 1.0,
                                           "abdominal_pain": 1.0}, "Gastroenteritis"),
    ("T08  Bronchitis",                   {"cough": 1.0, "chest_pain": 1.0,
                                           "fatigue": 1.0, "sore_throat": 1.0}, "Bronchitis"),
    ("T09  Urinary tract infection",      {"burning_urination": 1.0,
                                           "frequent_urination": 1.0,
                                           "abdominal_pain": 1.0}, "Urinary Tract Infection"),
    ("T10  Partial info (weak evidence)", {"high_fever": 1.0, "sore_throat": 1.0}, "Influenza"),
    ("T11  Fever only (too little info)", {"high_fever": 1.0}, "ABSTAIN"),
    ("T12  Non-specific complaints",      {"fatigue": 0.6, "loss_of_appetite": 0.6}, "ABSTAIN"),
]


def fnRunTests():
    print("=" * 78)
    print(" MedX-ES :: TEST CASE RESULTS")
    print("=" * 78)
    dicModel = fnTrainNaiveBayes()

    iPassed = 0
    print(f"\n  {'Test case':<34}{'Expected':<24}{'Actual':<24}{'Result'}")
    print("  " + "-" * 74)

    for (strName, dicPatient, strExpected) in g_lstTestCases:
        dicMl = fnPredictMlProbabilities(dicModel, dicPatient)
        dicResult = fnConsult(dicPatient, dicMl)

        if dicResult["bAbstain"]:
            strActual = "ABSTAIN"
        else:
            strActual = dicResult["lstRanking"][0][0]

        bPass = (strActual == strExpected)
        iPassed += 1 if bPass else 0
        print(f"  {strName:<34}{strExpected:<24}{strActual:<24}"
              f"{'PASS' if bPass else 'FAIL'}")

    print("  " + "-" * 74)
    print(f"  SUMMARY: {iPassed}/{len(g_lstTestCases)} test cases passed "
          f"({100.0 * iPassed / len(g_lstTestCases):.0f}%).")
    print("\n  Note: 'ABSTAIN' is the CORRECT behaviour for T11 and T12 - an expert")
    print("  system that has too little evidence must refuse to guess.")


if __name__ == "__main__":
    fnRunTests()
