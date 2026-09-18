"""
graph/stub_llm.py
-----------------
A deterministic fake LLM for testing NodeExpander without needing
a real API key or internet connection.

WHAT IT DOES:
  Given a prompt containing a parent claim, it picks from a bank of
  pre-written medical hypotheses. Uses simple keyword matching to pick
  contextually relevant responses.

WHY DETERMINISTIC?
  Tests should be repeatable. A real LLM gives different outputs each run.
  This stub always returns the same children for the same input.

SWAP POINT:
  Replace StubLLMClient with a real client that implements:
      def chat(self, prompt: str) -> str

  Example Ollama client:
      class OllamaLLMClient:
          def chat(self, prompt: str) -> str:
              import requests
              resp = requests.post("http://localhost:11434/api/generate",
                  json={"model": "llama3.1:8b", "prompt": prompt, "stream": False})
              return resp.json()["response"]
"""

# Pre-written hypothesis banks keyed by domain/keyword
HYPOTHESIS_BANK = {
    "stemi": [
        "Right coronary artery occlusion is the most likely cause of inferior STEMI.",
        "Immediate primary PCI is indicated within 90 minutes of symptom onset.",
        "Antiplatelet therapy with aspirin and P2Y12 inhibitor should be initiated.",
        "Risk of cardiogenic shock requires haemodynamic monitoring in ICU.",
        "Reperfusion injury may cause transient arrhythmias post-PCI.",
        "Left ventricular wall motion abnormality expected on echocardiogram.",
    ],
    "pe": [
        "Bilateral lower limb DVT should be ruled out with Doppler ultrasound.",
        "CT pulmonary angiography is the gold standard for PE confirmation.",
        "Anticoagulation with LMWH or DOAC should begin immediately.",
        "Massive PE with haemodynamic instability may require thrombolysis.",
        "Paradoxical embolism through patent foramen ovale is a rare complication.",
        "Right heart strain on echo suggests massive or submassive PE.",
    ],
    "aspirin": [
        "Proton pump inhibitor co-prescribing reduces GI bleeding risk with aspirin.",
        "Clopidogrel monotherapy may be considered when aspirin is contraindicated.",
        "GI bleeding risk must be weighed against thrombotic risk in ACS management.",
        "Haemoglobin and haematocrit monitoring required during dual antiplatelet therapy.",
        "Endoscopic haemostasis should precede anticoagulation where possible.",
        "The risk-benefit ratio of aspirin must be reassessed given active haemorrhage.",
    ],
    "default": [
        "Further diagnostic workup is required to narrow the differential.",
        "Comorbid conditions may be contributing to the clinical presentation.",
        "Specialist consultation should be considered for this presentation.",
        "Laboratory investigations should be repeated in 6 hours for trend analysis.",
        "Clinical deterioration warrants escalation to intensive care monitoring.",
        "Patient history of prior similar episodes should be explored.",
    ],
}


class StubLLMClient:
    """
    Deterministic fake LLM. Picks 3 hypotheses from the bank based on
    keywords in the prompt. Always returns the same output for the same input.
    """

    def chat(self, prompt: str) -> str:
        """Returns 3 hypotheses as a newline-separated string."""
        prompt_lower = prompt.lower()

        if "stemi" in prompt_lower or "troponin" in prompt_lower or "ecg" in prompt_lower:
            bank = HYPOTHESIS_BANK["stemi"]
        elif "pe" in prompt_lower or "pulmonary" in prompt_lower or "dvt" in prompt_lower:
            bank = HYPOTHESIS_BANK["pe"]
        elif "aspirin" in prompt_lower or "gi bleed" in prompt_lower or "haemorrhage" in prompt_lower:
            bank = HYPOTHESIS_BANK["aspirin"]
        else:
            bank = HYPOTHESIS_BANK["default"]

        return "\n".join(bank[:3])


class CyclingStubLLMClient:
    """
    A variant that CYCLES through hypothesis banks, causing entropy to rise
    after initial decline — specifically for triggering RabbitHoleDetector
    in synthetic_case_2 tests.

    HOW: After 2 expansions of good relevant hypotheses, it starts returning
    generic/vague ones (higher entropy). This mimics a reasoning path that
    drifts into speculation.
    """

    def __init__(self):
        self._call_count = 0

    def chat(self, prompt: str) -> str:
        self._call_count += 1
        prompt_lower = prompt.lower()

        # First 2 calls: relevant, low-entropy hypotheses
        if self._call_count <= 2:
            if "pe" in prompt_lower or "pulmonary" in prompt_lower:
                bank = HYPOTHESIS_BANK["pe"]
            else:
                bank = HYPOTHESIS_BANK["stemi"]
            return "\n".join(bank[:3])

        # After that: drift into vague, high-entropy territory
        vague = [
            "The aetiology of this presentation remains unclear and requires broader investigation.",
            "Multiple overlapping systemic conditions may be contributing to this clinical picture.",
            "Rare or atypical presentations of common conditions cannot be excluded at this stage.",
        ]
        return "\n".join(vague)
