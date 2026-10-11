#!/usr/bin/env python3
"""
=============================================================================
 MedX-ES  ::  server.py      (web interface - no external packages needed)
-----------------------------------------------------------------------------
 Course    : 2311303 -> 2311304 - AI & Data Science Lab (Sem VI)
 Assignment: No. 6 - Mini Project (Expert System - Hospitals & Medical)

 This server connects the Neubrutalism web UI (frontend/index.html) to the
 Python expert-system modules (knowledge base, inference engine, ML prior).

 It uses ONLY the Python standard library, so it runs on any machine with
 Python 3 installed - nothing to pip install.

 Usage:
     python3 server.py            -> http://localhost:8000
     python3 server.py 8080       -> http://localhost:8080
=============================================================================
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from knowledge_base import g_dicSymptomQuestions, g_strDisclaimer
from inference_engine import fnConsult
from ml_prior import (fnTrainNaiveBayes, fnPredictMlProbabilities,
                      fnEvaluateOnTestSet)
from symptom_lexicon import fnExtractSymptomsFromText

# ---------------------------------------------------------------------------
# paths and the trained ML model (trained once, when the server starts)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML = os.path.join(BASE_DIR, "frontend", "index.html")

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
# HTTP handler
# ---------------------------------------------------------------------------

class MedXESHandler(BaseHTTPRequestHandler):

    def log_message(self, *args):        # keep the console quiet
        pass

    # ---- helpers ---------------------------------------------------------
    def _send_json(self, dicPayload, iStatus=200):
        byBody = json.dumps(dicPayload).encode("utf-8")
        self.send_response(iStatus)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(byBody)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(byBody)

    def _send_html(self):
        try:
            with open(INDEX_HTML, "rb") as fileHtml:
                byBody = fileHtml.read()
        except FileNotFoundError:
            self._send_json({"strError": "frontend/index.html not found"}, 404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(byBody)))
        self.end_headers()
        self.wfile.write(byBody)

    # ---- GET routes ------------------------------------------------------
    def do_GET(self):
        strPath = urlparse(self.path).path
        if strPath in ("/", "/index.html"):
            self._send_html()
        elif strPath == "/api/symptoms":
            self._send_json({
                "dicQuestions": g_dicSymptomQuestions,
                "dicCategories": g_dicSymptomCategories,
            })
        elif strPath == "/api/demos":
            self._send_json({"lstDemoCases": g_lstDemoCases})
        elif strPath == "/api/model-info":
            self._send_json({
                "iRecords": dicModel["iTotalCases"],
                "iDiseases": len(dicModel["dicDiseaseCount"]),
                "strSource": dicModel["strSource"],
                "dNbAccuracy": dNbAccuracy,
            })
        else:
            self._send_json({"strError": "not found"}, 404)

    # ---- POST routes -----------------------------------------------------
    def do_POST(self):
        strPath = urlparse(self.path).path
        iLength = int(self.headers.get("Content-Length", 0))
        try:
            dicBody = json.loads(self.rfile.read(iLength) or b"{}")
        except json.JSONDecodeError:
            self._send_json({"strError": "invalid JSON"}, 400)
            return

        if strPath == "/api/consult":
            dicPatient = {strK: float(dV)
                          for strK, dV in dicBody.get("dicSymptoms", {}).items()}
            self._send_json(fnBuildResultPayload(fnConsultWithModel(dicPatient)))

        elif strPath == "/api/consult-text":
            strText = dicBody.get("strText", "")
            setSymptoms = fnExtractSymptomsFromText(strText)
            if not setSymptoms:
                self._send_json({
                    "bAbstain": True,
                    "strMessage": "No known symptom was recognised in that text.",
                    "strDisclaimer": g_strDisclaimer,
                    "lstRanking": [], "lstMlOnlyTop": [], "dicExplanation": None,
                    "setRecognised": [],
                })
                return
            dicPatient = {s: 1.0 for s in setSymptoms}
            dicPayload = fnBuildResultPayload(fnConsultWithModel(dicPatient))
            dicPayload["setRecognised"] = sorted(setSymptoms)
            self._send_json(dicPayload)
        else:
            self._send_json({"strError": "not found"}, 404)


# ---------------------------------------------------------------------------
# start the server
# ---------------------------------------------------------------------------

def fnMain():
    iPort = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print("=" * 64)
    print(" MedX-ES :: Explainable Hybrid Expert System - web interface")
    print("=" * 64)
    print(f"  ML prior trained on {dicModel['iTotalCases']} records, "
          f"{len(dicModel['dicDiseaseCount'])} diseases ({dicModel['strSource']})")
    if dNbAccuracy is not None:
        print(f"  Naive Bayes test accuracy on held-out data: "
              f"{100.0 * dNbAccuracy:.1f}%")
    print(f"\n  >>> Open  http://localhost:{iPort}  in your browser <<<\n")
    print("  (press Ctrl+C to stop)")
    try:
        ThreadingHTTPServer(("0.0.0.0", iPort), MedXESHandler).serve_forever()
    except OSError:
        print(f"  Port {iPort} is busy - try:  python3 server.py 8080")
    except KeyboardInterrupt:
        print("\n  Server stopped.")


if __name__ == "__main__":
    fnMain()
