#!/usr/bin/env python3
"""
=============================================================================
 MedX-ES  ::  medx_wsgi.py      (WSGI entry point for PythonAnywhere)
-----------------------------------------------------------------------------
 This file lets the SAME expert system run on PythonAnywhere (and any other
 WSGI host). It reuses the exact logic modules used by server.py:

     knowledge_base.py   inference_engine.py   ml_prior.py   symptom_lexicon.py

 The API contract is IDENTICAL to server.py:

     GET  /                -> the Neubrutalism web UI (frontend/index.html)
     GET  /api/symptoms    -> {dicQuestions, dicCategories}
     GET  /api/demos       -> {lstDemoCases}
     GET  /api/model-info  -> {iRecords, iDiseases, strSource, dNbAccuracy}
     POST /api/consult     body {dicSymptoms:{id:1.0}}
     POST /api/consult-text body {strText}

 The ML prior is trained ONCE, when this module is first imported (the WSGI
 process stays warm between requests, so training cost is paid only once).

 HOW TO DEPLOY ON PYTHONANYWHERE:
     1. Upload this whole project folder to your PythonAnywhere account
        (Dashboard -> Files, or clone the GitHub repo via a Bash console).
     2. Dashboard -> Web tab -> "Add a new web app"
        -> "Manual configuration" -> pick your Python version (3.13).
     3. Open the generated WSGI file (link on the Web tab) and replace its
        contents with:

            import sys
            sys.path.insert(0, "/home/YOURUSERNAME/MedX-ES")
            from medx_wsgi import application

        (use the real path to this folder on PythonAnywhere).
     4. Click "Reload" on the Web tab. Done - open the given URL.

 NOTE: educational decision-SUPPORT prototype. Not a medical device.
=============================================================================
"""

import json
import os
import sys

# make sure the project directory is importable no matter where the WSGI
# file is invoked from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from knowledge_base import g_dicSymptomQuestions, g_strDisclaimer
from inference_engine import fnConsult
from ml_prior import (fnTrainNaiveBayes, fnPredictMlProbabilities,
                      fnEvaluateOnTestSet)
from symptom_lexicon import fnExtractSymptomsFromText

INDEX_HTML = os.path.join(BASE_DIR, "frontend", "index.html")

# train the ML prior once, at import time (the WSGI process is long-lived)
dicModel = fnTrainNaiveBayes()
dNbAccuracy = fnEvaluateOnTestSet(dicModel)

# symptom categories - used by the UI to group the chips nicely
g_dicSymptomCategories = {
    "General":        ["high_fever", "chills", "sweating", "fatigue", "loss_of_appetite"],
    "Head & Nervous": ["severe_headache", "retro_orbital_pain"],
    "Respiratory":    ["cough", "sore_throat", "runny_nose", "sneezing",
                       "chest_pain", "breathlessness"],
    "Digestive":      ["nausea", "vomiting", "abdominal_pain", "diarrhoea"],
    "Skin & Joints":  ["skin_rash", "severe_joint_pain", "body_ache"],
    "Urinary":        ["burning_urination", "frequent_urination"],
    "Laboratory":     ["platelet_drop"],
}

# ready-made demo patients (same ones as main.py demo mode)
g_lstDemoCases = [
    {"strName": "Dengue",   "strEmoji": "🦟",
     "dicPatient": {"high_fever": 1.0, "severe_headache": 1.0,
                    "retro_orbital_pain": 1.0, "severe_joint_pain": 1.0,
                    "skin_rash": 1.0, "platelet_drop": 1.0, "fatigue": 0.8}},
    {"strName": "Malaria",  "strEmoji": "🌡️",
     "dicPatient": {"high_fever": 1.0, "chills": 1.0, "sweating": 1.0,
                    "severe_headache": 1.0, "fatigue": 1.0}},
    {"strName": "Typhoid",  "strEmoji": "🤒",
     "dicPatient": {"high_fever": 1.0, "abdominal_pain": 1.0,
                    "loss_of_appetite": 1.0, "fatigue": 1.0}},
    {"strName": "Pneumonia", "strEmoji": "🫁",
     "dicPatient": {"cough": 1.0, "chest_pain": 1.0, "breathlessness": 1.0,
                    "high_fever": 1.0}},
    {"strName": "Gastroenteritis", "strEmoji": "🤢",
     "dicPatient": {"diarrhoea": 1.0, "vomiting": 1.0, "abdominal_pain": 1.0}},
]


def fnConsultWithModel(dicPatient):
    """Run one consultation through the hybrid expert-system engine."""
    dicMl = fnPredictMlProbabilities(dicModel, dicPatient)
    return fnConsult(dicPatient, dicMl)


def fnBuildResultPayload(dicResult):
    """Convert the engine's result dictionary into JSON for the web UI."""
    dicPayload = {
        "bAbstain": dicResult["bAbstain"],
        "strMessage": dicResult["strMessage"],
        "strDisclaimer": g_strDisclaimer,
        "lstRanking": [
            {
                "strDisease": strDisease,
                "dCf": dicResult["dicCfScores"].get(strDisease, 0.0),
                "dMl": dicResult["dicMlProbs"].get(strDisease.strip().lower(), 0.0),
                "dFinal": dFinal,
            }
            for (strDisease, dFinal) in dicResult["lstRanking"][:5]
        ],
        "lstMlOnlyTop": [],
        "dicExplanation": None,
    }

    # what the ML prior alone would have guessed (shown when we abstain)
    if dicResult["bAbstain"] and not dicResult["lstFiredRules"]:
        dicPayload["lstMlOnlyTop"] = [
            {"strDisease": d, "dProb": p}
            for d, p in sorted(dicResult["dicMlProbs"].items(),
                               key=lambda kv: kv[1], reverse=True)[:3]
            if p > 0.01
        ]

    # explanation trace for the top disease
    if not dicResult["bAbstain"] and dicResult["lstRanking"]:
        strTop = dicResult["lstRanking"][0][0]
        dicPayload["dicExplanation"] = {
            "strDisease": strTop,
            "lstFiredRules": [
                {"lstPremises": list(lstPremises), "dStrength": dStrength}
                for (strDisease, lstPremises, dStrength) in dicResult["lstFiredRules"]
                if strDisease == strTop
            ],
            "dCombinedCf": dicResult["dicCfScores"].get(strTop, 0.0),
            "dMlProb": dicResult["dicMlProbs"].get(strTop.strip().lower(), 0.0),
            "dFinal": dicResult["lstRanking"][0][1],
            "strAdvice": __import__("knowledge_base").g_dicDiseaseAdvice.get(
                strTop, "Consult a doctor."),
        }
    return dicPayload


# ---------------------------------------------------------------------------
# WSGI application
# ---------------------------------------------------------------------------

def _send_json(start_response, dicPayload, iStatus=200):
    byBody = json.dumps(dicPayload).encode("utf-8")
    start_response(str(iStatus), [
        ("Content-Type", "application/json"),
        ("Content-Length", str(len(byBody))),
        ("Access-Control-Allow-Origin", "*"),
    ])
    return [byBody]


def _send_html(start_response):
    try:
        with open(INDEX_HTML, "rb") as fileHtml:
            byBody = fileHtml.read()
    except FileNotFoundError:
        return _send_json(start_response, {"strError": "frontend/index.html not found"}, 404)
    start_response("200", [
        ("Content-Type", "text/html; charset=utf-8"),
        ("Content-Length", str(len(byBody))),
    ])
    return [byBody]


def application(environ, start_response):
    """WSGI entry point - routes every request exactly like server.py."""
    strPath = environ.get("PATH_INFO", "/")
    strMethod = environ.get("REQUEST_METHOD", "GET").upper()

    # ---- GET routes ------------------------------------------------------
    if strMethod == "GET":
        if strPath in ("/", "/index.html"):
            return _send_html(start_response)
        if strPath == "/api/symptoms":
            return _send_json(start_response, {
                "dicQuestions": g_dicSymptomQuestions,
                "dicCategories": g_dicSymptomCategories,
            })
        if strPath == "/api/demos":
            return _send_json(start_response, {"lstDemoCases": g_lstDemoCases})
        if strPath == "/api/model-info":
            return _send_json(start_response, {
                "iRecords": dicModel["iTotalCases"],
                "iDiseases": len(dicModel["dicDiseaseCount"]),
                "strSource": dicModel["strSource"],
                "dNbAccuracy": dNbAccuracy,
            })
        return _send_json(start_response, {"strError": "not found"}, 404)

    # ---- POST routes -----------------------------------------------------
    if strMethod == "POST":
        try:
            iLength = int(environ.get("CONTENT_LENGTH", 0) or 0)
        except ValueError:
            iLength = 0
        try:
            dicBody = json.loads(environ["wsgi.input"].read(iLength) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _send_json(start_response, {"strError": "invalid JSON"}, 400)

        if strPath == "/api/consult":
            dicPatient = {strK: float(dV)
                          for strK, dV in dicBody.get("dicSymptoms", {}).items()}
            return _send_json(start_response,
                              fnBuildResultPayload(fnConsultWithModel(dicPatient)))

        if strPath == "/api/consult-text":
            strText = dicBody.get("strText", "")
            setSymptoms = fnExtractSymptomsFromText(strText)
            if not setSymptoms:
                return _send_json(start_response, {
                    "bAbstain": True,
                    "strMessage": "No known symptom was recognised in that text.",
                    "strDisclaimer": g_strDisclaimer,
                    "lstRanking": [], "lstMlOnlyTop": [], "dicExplanation": None,
                    "setRecognised": [],
                })
            dicPatient = {s: 1.0 for s in setSymptoms}
            dicPayload = fnBuildResultPayload(fnConsultWithModel(dicPatient))
            dicPayload["setRecognised"] = sorted(setSymptoms)
            return _send_json(start_response, dicPayload)

        return _send_json(start_response, {"strError": "not found"}, 404)

    return _send_json(start_response, {"strError": "method not allowed"}, 405)
