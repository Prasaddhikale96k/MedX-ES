"""
=============================================================================
 MedX-ES  ::  inference_engine.py
-----------------------------------------------------------------------------
 This file is the INFERENCE ENGINE - the "brain" of the expert system.
 It links the FACTS (patient's symptoms) with the RULES (knowledge base)
 and derives conclusions, exactly like a human expert reasoning step by step.

 Techniques used:
   1. FORWARD CHAINING  - start from the facts, fire every rule that matches,
                          and collect the conclusions (data-driven).
   2. CERTAINTY FACTORS - how strongly the evidence supports each conclusion
                          (MYCIN-style combination algebra).
   3. HYBRID SCORING    - rule-based CF blended with the machine-learning
                          prior probability (neuro-symbolic design).
   4. ABSTENTION GATE   - if confidence is too low the system refuses to
                          guess and asks the user to consult a doctor
                          (human-in-the-loop safety, as in modern CDSS work).
=============================================================================
"""

from knowledge_base import (g_dicDiseaseRules, g_lstAllSymptoms,
                            g_dicDiseaseAdvice, g_strDisclaimer)

# ---------------------------------------------------------------------------
# tunable constants
# ---------------------------------------------------------------------------
dSymbolicWeight   = 0.75  # weight of the rule-based CF  in the hybrid score
dMlWeight         = 0.25  # weight of the ML probability in the hybrid score
dAbstainThreshold = 0.25  # below this final score the system must abstain


# ---------------------------------------------------------------------------
# 1. CERTAINTY-FACTOR ALGEBRA
# ---------------------------------------------------------------------------

def fnCfCombineTwo(dCfOne, dCfTwo):
    """Combine two certainty factors of the SAME conclusion (MYCIN rules).

    agreeing evidence (same sign) : CF1 + CF2 * (1 - CF1)
    conflicting evidence           : (CF1 + CF2) / (1 - min(|CF1|,|CF2|))
    """
    if dCfOne * dCfTwo >= 0:                       # both positive or both negative
        return dCfOne + dCfTwo * (1.0 - dCfOne)
    dDenominator = 1.0 - min(abs(dCfOne), abs(dCfTwo))
    if dDenominator == 0.0:                        # total conflict -> cancel out
        return 0.0
    return (dCfOne + dCfTwo) / dDenominator


def fnCfCombineList(lstCf):
    """Combine a whole list of certainty factors into a single CF in [-1,1]."""
    dResult = 0.0
    for dCf in lstCf:
        dResult = fnCfCombineTwo(dResult, dCf)
    return round(max(-1.0, min(1.0, dResult)), 4)


def fnRuleStrength(dicPatient, lstPremises, dRuleCf):
    """Strength of one rule for one patient.

    A rule fires only if EVERY premise is present, and its strength is
    min(CF of each premise) * rule CF   (classic MYCIN conjunction rule).
    """
    lstPremiseCf = [dicPatient.get(strSymptom, 0.0) for strSymptom in lstPremises]
    return round(min(lstPremiseCf) * dRuleCf, 4)


# ---------------------------------------------------------------------------
# 2. FORWARD CHAINING
# ---------------------------------------------------------------------------

def fnForwardChain(dicPatient):
    """Run forward chaining over the whole rule base.

    Returns
        dicCfScores  : {disease: combined certainty factor}
        lstFiredRules: [(disease, premises, strength), ...]  (for explanations)
    """
    dicCfScores = {}
    lstFiredRules = []

    for strDisease, lstRules in g_dicDiseaseRules.items():
        lstDiseaseCf = []
        for (lstPremises, dRuleCf) in lstRules:
            dStrength = fnRuleStrength(dicPatient, lstPremises, dRuleCf)
            if dStrength > 0.0:                    # the rule fired
                lstDiseaseCf.append(dStrength)
                lstFiredRules.append((strDisease, tuple(lstPremises), dStrength))
        if lstDiseaseCf:
            dicCfScores[strDisease] = fnCfCombineList(lstDiseaseCf)

    return dicCfScores, lstFiredRules


# ---------------------------------------------------------------------------
# 3. HYBRID SCORE  (symbolic CF  +  ML prior)
# ---------------------------------------------------------------------------

def fnHybridScore(dicCfScores, dicMlProbs):
    """FinalScore(d) = 0.75 * CF_rule(d)  +  0.25 * P_ML(d)

    GATING RULE: a disease with certainty factor 0 (no rule fired) gets a
    final score of 0 - the rule base decides WHAT can be diagnosed, and the
    ML prior only RE-RANKS the diseases the rules already support. This
    keeps every answer explainable and makes abstention meaningful.
    """
    setAllDiseases = set(dicCfScores) | set(dicMlProbs)
    dicFinal = {}
    for strDisease in setAllDiseases:
        dCf = dicCfScores.get(strDisease, 0.0)
        if dCf <= 0.0:                       # gated: no symbolic support
            dicFinal[strDisease] = 0.0
            continue
        dMl = dicMlProbs.get(strDisease.strip().lower(), 0.0)
        dicFinal[strDisease] = round(dSymbolicWeight * dCf + dMlWeight * dMl, 4)
    return dicFinal


# ---------------------------------------------------------------------------
# 4. MAIN ENTRY POINT OF THE ENGINE
# ---------------------------------------------------------------------------

def fnConsult(dicPatient, dicMlProbs=None):
    """Consult the expert system with one patient's findings.

    dicPatient  : {symptom_id: certainty 0..1}   (1 = definitely present)
    dicMlProbs  : optional {disease: probability} from the ML module

    Returns a result dictionary with the ranking, top hypothesis and the
    decision of the abstention gate.
    """
    if dicMlProbs is None:
        dicMlProbs = {}

    dicCfScores, lstFiredRules = fnForwardChain(dicPatient)
    dicFinal = fnHybridScore(dicCfScores, dicMlProbs)
    # the ranking only contains diseases with symbolic support (Final > 0);
    # the ML prior can re-rank them but never introduce a new disease
    lstRanking = sorted(((d, s) for d, s in dicFinal.items() if s > 0.0),
                        key=lambda kv: kv[1], reverse=True)

    dicResult = {
        "lstRanking":    lstRanking,                       # [(disease, final), ...]
        "dicCfScores":   dicCfScores,
        "dicMlProbs":    dicMlProbs,
        "lstFiredRules": lstFiredRules,
        "bAbstain":      False,
        "strMessage":    "",
    }

    if not lstFiredRules:
        # No rule in the knowledge base matched - the safe answer is to abstain.
        dicResult["bAbstain"] = True
        dicResult["strMessage"] = ("No rule in the knowledge base matches these "
                                   "findings - not enough evidence to conclude.")
        return dicResult

    strTopDisease, dTopScore = lstRanking[0]
    if dTopScore < dAbstainThreshold:
        dicResult["bAbstain"] = True
        dicResult["strMessage"] = (f"Confidence too low (best score {dTopScore} "
                                   f"< threshold {dAbstainThreshold}).")
        return dicResult

    dicResult["strMessage"] = f"Most likely: {strTopDisease} (confidence {dTopScore})."
    return dicResult


# ---------------------------------------------------------------------------
# 5. EXPLANATION FACILITY  (the "why" behind every answer)
# ---------------------------------------------------------------------------

def fnExplainResult(dicResult):
    """Print a human-readable trace of the reasoning (rule firings + CF math)."""
    if dicResult["bAbstain"]:
        print(f"  {dicResult['strMessage']}")
        print("  -> The system refuses to guess. Please consult a qualified doctor.")
        return

    strTop = dicResult["lstRanking"][0][0]
    print(f"\n  EXPLANATION for '{strTop}':")
    print("  Rules that fired (IF premises THEN conclusion):")
    for (strDisease, lstPremises, dStrength) in dicResult["lstFiredRules"]:
        if strDisease == strTop:
            print(f"    IF {list(lstPremises)}  ->  contributes CF = {dStrength}")
    print("  Certainty-factor combination used:")
    print("    CF_combine = CF1 + CF2*(1-CF1)               [agreeing evidence]")
    print("    CF_combine = (CF1+CF2)/(1-min(|CF1|,|CF2|))) [conflicting evidence]")
    print(f"  Combined rule CF for {strTop} = {dicResult['dicCfScores'].get(strTop, 0.0)}")
    if strTop.strip().lower() in dicResult["dicMlProbs"]:
        print(f"  ML prior probability for {strTop} = "
              f"{dicResult['dicMlProbs'][strTop.strip().lower()]}")
    print(f"  Hybrid score: Final = {dSymbolicWeight}*CF + {dMlWeight}*P_ML "
          f"(rule base gates, ML prior re-ranks)")
    print(f"  Advice: {g_dicDiseaseAdvice.get(strTop, 'Consult a doctor.')}")
