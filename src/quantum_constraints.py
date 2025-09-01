# src/quantum_constraints.py
"""
Turn a single quantum bitstring (and known register layout) into narrative constraints
that a random-number generator wouldn't naturally produce.

Fingerprints used:
- Time-chain parity ΔT[i] = T[i] XOR T[i+1]  -> scene shift / continuity
- Triadic twist τ[i] = T[i-1] XOR T[i] XOR T[i+1] -> panel "twist" requirement
- Style parity σS = XOR over style qubits -> global rhetorical device bias
- Anchor pairs (T[i], S[j]) -> recurring phrase / motif placement
- Backend fingerprint (name, depth) -> global tone, line-length bias

These constraints drive Gemini-Flash (text) to produce JSON that honors: dialogue acts, devices,
motifs, line counts, and “must-have” features per panel.
"""

from dataclasses import dataclass
from typing import Dict, List, Any
import hashlib

@dataclass
class RegisterShape:
    Tn: int
    An: int
    En: int
    Cn: int
    Sn: int

RDEV_MAP = ["anaphora", "antimetabole", "call-and-response", "chiasmus"]

def bits_to_int(bits: str) -> int:
    return int(bits[::-1], 2) if bits else 0  # reverse for readability

def parity(bits: str) -> int:
    return sum(int(b) for b in bits) & 1

def hash_phrase(seed: str, vocab: List[str]) -> str:
    h = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    idx = int(h[:8], 16) % len(vocab)
    return vocab[idx]

def build_constraints(
    bitstring: str,
    shapes: RegisterShape,
    backend_name: str = "unknown",
    transpiled_depth: int = 0,
    panels_data: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    Tn, An, En, Cn, Sn = shapes.Tn, shapes.An, shapes.En, shapes.Cn, shapes.Sn
    T_bits = bitstring[:Tn]
    A_bits = bitstring[Tn:Tn+An]
    E_bits = bitstring[Tn+An:Tn+An+En]
    C_bits = bitstring[Tn+An+En:Tn+An+En+Cn]
    S_bits = bitstring[Tn+An+En+Cn:Tn+An+En+Cn+Sn]

    # Time-chain differentials: continuity (0) vs shift (1)
    deltaT = [ (int(T_bits[i]) ^ int(T_bits[i+1])) for i in range(Tn-1) ]
    # Triadic twist τ
    tri = []
    for i in range(Tn):
        left = int(T_bits[i-1]) if i-1 >= 0 else 0
        mid  = int(T_bits[i])
        right= int(T_bits[i+1]) if i+1 < Tn else 0
        tri.append((left ^ mid ^ right))

    # Global style parity -> rhetorical device bias
    sigmaS = parity(S_bits)
    rdev_bias = RDEV_MAP[(bits_to_int(S_bits) % len(RDEV_MAP))]

    # Backend fingerprint → tone + line length bias
    # (Shorter lines on noisier/ deeper circuits, just as a poetic nod to decoherence.)
    depth_bias = max(0, min(8, transpiled_depth // 25))  # 0..8 scale
    base_line_limit = 95 - 5 * depth_bias
    tone = "noir" if "brisbane" in backend_name.lower() else (
           "surreal" if "torino" in backend_name.lower() else "contemporary")

    # Recurring phrase seed (quantum-anchored)
    vocab = [
      "Every choice forks.", "Look where the light bends.", "We were already here.",
      "Say it twice to make it true.", "Paths remember us.", "Keep the bridge honest.",
      "Silence isn’t neutral.", "Nothing collapses alone."
    ]
    recurring = hash_phrase(T_bits + S_bits + backend_name, vocab)

    # Panel-level constraints
    panels_constraints = []
    for i, p in enumerate(panels_data or [{"panel_index": j+1} for j in range(Tn)]):
        # Dialogue act: continuity vs shift decides whether panel prefers Q/A vs assertion
        if i < Tn-1 and deltaT[i] == 1:
            dialog_act = "question-or-challenge"
        else:
            dialog_act = "statement-or-reassurance"

        # Rhetorical device selection nudged by triadic twist and style bias
        # Weights: when tri[i]==1, prefer “inversion/call-and-response”; else “anaphora/chiasmus”
        if tri[i] == 1:
            device = "call-and-response" if sigmaS == 0 else "antimetabole"
        else:
            device = "anaphora" if sigmaS == 0 else "chiasmus"

        # Enforce recurring phrase in a few panels using anchors (T[i], S[j])
        must_use_recurring = ( (i % 2) == (sigmaS % 2) )

        # Line counts driven by style bits + depth
        lines = 1 + ((i + bits_to_int(S_bits)) % 2)  # 1–2 lines
        max_chars = max(60, base_line_limit)

        panels_constraints.append({
            "index": p["panel_index"],
            "dialogue_act": dialog_act,               # "question-or-challenge" | "statement-or-reassurance"
            "rhetorical_device": device,              # anaphora | antimetabole | call-and-response | chiasmus
            "must_use_recurring": must_use_recurring, # advise to weave recurring phrase subtly
            "max_lines_per_speaker": lines,           # 1–2
            "max_chars_per_line": max_chars,          # tightened by depth
        })

    return {
        "backend": backend_name,
        "tone": tone,
        "style_parity": sigmaS,
        "rhetorical_bias": rdev_bias,
        "recurring_phrase": recurring,
        "panel_constraints": panels_constraints
    }
