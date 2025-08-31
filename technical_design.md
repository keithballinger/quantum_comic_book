# Concept & Architecture

**Goal:** One measurement run encodes an entire comic (e.g., 6–12 panels). Panels, characters, moods, actions, camera angles—everything—are derived from correlated quantum bits so the narrative feels like a single collapsed history, not a bucket of coin flips.

**Core idea:**

* Build a shallow circuit with:

  * A **time register** of `P` panels.
  * A **character register** of `C` characters per panel (or reuse/entangle across time).
  * Optional **style/mood registers** that influence prompts.
* Entangle across time (panel-to-panel) and across characters (intra-panel) to create correlated choices.
* Measure **once** (1 shot). That single bitstring becomes your whole story map.
* Translate bits → deterministic prompt “slots” (scene, action, emotion, camera, palette).
* Generate panel images with **Gemini 2.5 Flash Image Preview**, passing the **previous panel’s image** as conditioning to keep style/identity consistent.

---

# Quantum Design

**Registers (example for a 6-panel strip, 2 characters):**

* `T` (time chain): 6 qubits → control scene evolution.
* `A` (actions): 6×2 qubits → per panel: 2 bits choose among 4 canonical actions.
* `E` (emotion): 6×2 qubits → per panel: 2 bits choose among 4 emotions.
* `C` (camera): 6×2 qubits → per panel: 2 bits choose shot type/angle.
* `S` (style): 4 qubits global → palette/ink density/line weight.

That’s `6 + 6*2 + 6*2 + 6*2 + 4 = 6 + 12 + 12 + 12 + 4 = 46` qubits. Plenty of headroom below 127.

**Entanglement pattern (shallow):**

* **Time chain:** apply Hadamard on `T[0]`, then CZ along chain `T[i]—T[i+1]` to correlate adjacent panels.
* **Panel-coupling:** lightly entangle each panel’s `A/E/C` with its `T[i]` using controlled rotations (e.g., CRX/CRZ at small angles) to inherit time “mood.”
* **Style anchor:** put `S` into superposition and attach weak CZ/CRZ from a few `T` nodes so the global style coheres with the story arc.
* Keep depth low to respect noise; prefer 1–2 entangling layers and small-angle rotations.

**Measurement:** Measure all qubits **once**. That single shot is your strip.

**Execution mode:** Use `qiskit-ibm-runtime` `Sampler` with 1 shot (philosophically pure). If you want a “safety” option, take a handful of shots, but only **use the first returned**—log the rest as “worlds you didn’t get.”

---

# Prompt Mapping (bits → prompts)

Create deterministic lookup tables:

* **Action (2 bits):**
  `00`: walk/approach, `01`: point/reach, `10`: pause/reflect, `11`: turn/leave

* **Emotion (2 bits):**
  `00`: curious, `01`: determined, `10`: conflicted, `11`: relieved

* **Camera (2 bits):**
  `00`: medium shot, `01`: wide shot, `10`: close-up, `11`: over-the-shoulder

* **Style (4 bits → palette/ink):**
  Map 0–15 to a fixed palette dictionary: noir ink, muted watercolor, bright pop, newsprint dots, etc.

* **Time node `T[i]` (1 bit each, or combine neighbors):**
  Use `T[i]` plus `(T[i-1], T[i+1])` parity to nudge setting progression:
  indoor → alley → rooftop → street → bridge → dawn light, etc.

Finally, turn the decoded slots into a concise image prompt per panel.

---

# Image Generation Strategy

**Model:** `gemini-2.5-flash-image-preview`.

**Consistency:**

* **Panel 1:** no prior image; pass a crisp, style-anchored prompt.
* **Panel 2..N:** include the **previous panel’s image bytes** in `contents` plus text like:
  “Keep the same character appearance, composition logic, and palette. Evolve the scene with \[action/emotion changes]. Maintain comic style with panel borders.”

This “image-to-image” conditioning keeps character identity and vibe.

---

# Dependencies

```bash
pip install qiskit qiskit-ibm-runtime google-genai pillow numpy
# Or your existing Qiskit env; you already use Runtime Sampler per your script.
```

* **IBM auth:** via `QiskitRuntimeService()` (account saved or token in env).
* **Gemini auth:** `GEMINI_API_KEY` in env.

---

# Reference Implementation (single file)

Save as `many_worlds_comic.py`. It:

1. Builds the circuit,
2. Runs one shot on IBM Quantum (`Sampler`),
3. Decodes prompts,
4. Generates images panel-by-panel, conditioning on the previous image,
5. Writes an HTML strip.

```python
#!/usr/bin/env python3
"""
Many-Worlds Comic Strip (one-shot quantum narrative -> comic panels)

- Builds a shallow entangled circuit over registers for time, action, emotion, camera, and style.
- Takes ONE SHOT on IBM Quantum. That single bitstring defines the entire strip.
- Uses Gemini 2.5 Flash Image Preview to render each panel, conditioning on the previous panel image for style/character consistency.

Requirements:
  pip install qiskit qiskit-ibm-runtime google-genai pillow numpy
Env:
  GEMINI_API_KEY=<your key>
  (IBM credentials via saved account or env supported by QiskitRuntimeService)
"""

import os
import io
import base64
import mimetypes
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler, SamplerOptions

from google import genai
from google.genai import types

# -----------------------------
# CONFIG
# -----------------------------
PANELS = 6                  # comic panels
CHARACTERS = 2              # conceptually 2 protagonists
SEED = None                 # set to int for reproducible transpilation/random mapping
OUTPUT_DIR = Path("output") / f"comic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
MODEL = "gemini-2.5-flash-image-preview"

# Prompts boilerplate
BASE_STYLE_TEXT = (
    "Comic book, inked lines, halftone shading, clean panel borders, cinematic lighting. "
    "Keep characters consistent across panels (hair, clothing, colors, face). "
)

CHARACTER_BIO = (
    "Two protagonists: 1) a thoughtful engineer in a minimalist jacket, "
    "2) a bold law student with a light scarf. Contemporary urban setting."
)

# -----------------------------
# LOOKUP TABLES
# -----------------------------
ACTIONS = ["approaches", "reaches out", "pauses to reflect", "turns away"]
EMOTIONS = ["curious", "determined", "conflicted", "relieved"]
CAMERAS = ["medium shot", "wide shot", "close-up", "over-the-shoulder"]

PALETTES = {
    0:  "noir ink with deep shadows",
    1:  "muted watercolor wash",
    2:  "bright pop colors",
    3:  "newsprint halftone dots",
    4:  "sepia dusk tones",
    5:  "cool cyan + magenta",
    6:  "warm amber + teal",
    7:  "high-contrast black & white",
    8:  "pastel palette",
    9:  "gritty urban palette",
    10: "sunrise golds",
    11: "rain-slick night palette",
    12: "wintry bluish palette",
    13: "late-afternoon amber",
    14: "neon city palette",
    15: "soft neutral palette",
}

SETTINGS = [
    "quiet indoor corridor",
    "narrow alley",
    "city rooftop",
    "side street",
    "bridge overlook",
    "dawn-lit avenue",
]

# -----------------------------
# CIRCUIT CONSTRUCTION
# -----------------------------
def build_circuit(panels=PANELS):
    """
    Registers:
      T: time chain (panels) -> PANELS
      A/E/C: each panel: 2 bits -> 2*PANELS each
      S: global style -> 4
    Measurements: all into classical bits.
    """
    # Qubit counts
    Tn = panels
    An = 2 * panels
    En = 2 * panels
    Cn = 2 * panels
    Sn = 4

    total = Tn + An + En + Cn + Sn

    qc = QuantumCircuit(total, total, name="many_worlds_comic")

    # Index helpers
    def idx_T(i): return i
    def idx_A(k): return Tn + k
    def idx_E(k): return Tn + An + k
    def idx_C(k): return Tn + An + En + k
    def idx_S(k): return Tn + An + En + Cn + k

    # Put T[0] into superposition; lightly entangle time chain
    qc.h(idx_T(0))
    for i in range(panels - 1):
        # time chain entanglement (shallow)
        qc.cx(idx_T(i), idx_T(i + 1))
        # small-angle phase to create richer correlations
        qc.rz(0.17, idx_T(i + 1))

    # Style register superposition + couple to time endpoints
    for s in range(Sn):
        qc.h(idx_S(s))
    # couple first and last time nodes to style
    for s in range(min(2, Sn)):
        qc.cz(idx_T(0), idx_S(s))
    for s in range(2, Sn):
        qc.cz(idx_T(panels - 1), idx_S(s))

    # Panel-local superpositions + weak conditioning on time node
    # Actions, Emotions, Cameras are 2 bits each per panel.
    for p in range(panels):
        # prepare superposition
        for sub in range(2):
            qc.h(idx_A(2*p + sub))
            qc.h(idx_E(2*p + sub))
            qc.h(idx_C(2*p + sub))
        # weak conditional rotations from T[p]
        t = idx_T(p)
        for sub in range(2):
            qc.crx(0.11, t, idx_A(2*p + sub))
            qc.cry(0.09, t, idx_E(2*p + sub))
            qc.crz(0.07, t, idx_C(2*p + sub))

    # Final measure all
    for c in range(total):
        qc.measure(c, c)

    return qc, (Tn, An, En, Cn, Sn)

# -----------------------------
# IBM RUNTIME EXECUTION
# -----------------------------
def run_one_shot(qc):
    service = QiskitRuntimeService()  # uses saved account or env
    # Pick a device; you can replace with a specific backend name you prefer
    # e.g., "ibm_brisbane" / "ibm_torino" / any 127q device available to you
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=127) or service.backends(simulator=False)[0]

    options = SamplerOptions(default_shots=1)
    with Session(service=service, backend=backend) as session:
        sampler = Sampler(session=session, options=options)
        job = sampler.run([qc])
        result = job.result()

    # One circuit, shots=1 → first (and only) quasi-probabilities sample
    # Sampler returns distributions; sample a bitstring from quasi probs robustly:
    dist = result.quasi_dists[0]
    # Pick the highest-probability outcome deterministically (one collapse)
    bit_int = max(dist, key=dist.get)
    # Convert to bitstring of correct length
    nbits = qc.num_clbits
    bitstring = bin(bit_int)[2:].zfill(nbits)
    # Qiskit measures little-endian by classical bit index; we measured 1:1 mapping (q->c),
    # so the leftmost char is the most significant classical bit we measured last.
    return bitstring[::-1]  # reverse to align with our indices if desired

# -----------------------------
# DECODE BITSTRING -> PROMPTS
# -----------------------------
def decode(bitstring, shapes, panels=PANELS):
    Tn, An, En, Cn, Sn = shapes
    # Extract slices
    T_bits = bitstring[:Tn]
    A_bits = bitstring[Tn:Tn+An]
    E_bits = bitstring[Tn+An:Tn+An+En]
    C_bits = bitstring[Tn+An+En:Tn+An+En+Cn]
    S_bits = bitstring[Tn+An+En+Cn:Tn+An+En+Cn+Sn]

    # Global style id
    style_id = int(S_bits[::-1], 2)  # reverse for readability
    palette = PALETTES[style_id % len(PALETTES)]

    panels_data = []
    for p in range(panels):
        a = int(A_bits[2*p:2*p+2][::-1], 2)
        e = int(E_bits[2*p:2*p+2][::-1], 2)
        c = int(C_bits[2*p:2*p+2][::-1], 2)
        t = int(T_bits[p], 2)

        # Setting nudged by local time and neighbor parity
        left = int(T_bits[p-1], 2) if p-1 >= 0 else 0
        right = int(T_bits[p+1], 2) if p+1 < panels else 0
        parity = (t + left + right) % len(SETTINGS)
        setting = SETTINGS[parity]

        panels_data.append({
            "panel_index": p+1,
            "action": ACTIONS[a],
            "emotion": EMOTIONS[e],
            "camera": CAMERAS[c],
            "setting": setting
        })

    return palette, panels_data

def panel_prompt(palette, panel):
    # Compose a concise, controllable prompt
    return (
        f"{BASE_STYLE_TEXT} {CHARACTER_BIO} "
        f"Panel {panel['panel_index']}: {panel['camera']} in a {panel['setting']}. "
        f"Primary emotion: {panel['emotion']}. Action: one character {panel['action']}. "
        f"Use {palette}."
    )

# -----------------------------
# GEMINI IMAGE GENERATION
# -----------------------------
def save_bytes(path: Path, data: bytes, mime: str):
    ext = mimetypes.guess_extension(mime) or ".jpg"
    out = path.with_suffix(ext)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return out

def generate_panel_images(prompts):
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    files = []

    prev_image_bytes = None
    prev_mime = None

    for i, prompt in enumerate(prompts, start=1):
        parts = []
        if prev_image_bytes and prev_mime:
            parts.append(types.Part.from_bytes(mime_type=prev_mime, data=prev_image_bytes))
            parts.append(types.Part.from_text(
                "Create the next comic panel based on the attached previous panel. "
                "Preserve character identity, outfit, palette, and framing logic. "
                "Evolve the scene according to the new instruction."
            ))
        parts.append(types.Part.from_text(prompt))

        generate_content_config = types.GenerateContentConfig(
            response_modalities=["IMAGE","TEXT"],
        )

        file_index = 0
        img_path = None

        for chunk in client.models.generate_content_stream(
            model=MODEL,
            contents=[types.Content(role="user", parts=parts)],
            config=generate_content_config,
        ):
            if (
                chunk.candidates
                and chunk.candidates[0].content
                and chunk.candidates[0].content.parts
            ):
                part0 = chunk.candidates[0].content.parts[0]
                if getattr(part0, "inline_data", None) and part0.inline_data.data:
                    data_buffer = part0.inline_data.data
                    mime = part0.inline_data.mime_type or "image/jpeg"
                    img_path = save_bytes(OUTPUT_DIR / f"panel_{i:02d}", data_buffer, mime)
                    prev_image_bytes = data_buffer
                    prev_mime = mime
                elif getattr(chunk, "text", None):
                    # Optional: log text output
                    pass

        if img_path is None:
            raise RuntimeError(f"No image returned for panel {i}")

        files.append(img_path)

    return files

# -----------------------------
# HTML STRIP RENDER
# -----------------------------
def write_html_strip(files, palette, panels_data):
    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'/>",
        "<title>Many-Worlds Comic Strip</title>",
        "<style>body{font-family:sans-serif;background:#111;color:#eee;padding:20px} .row{display:flex;gap:12px;flex-wrap:wrap} img{max-width:100%;height:auto;border:3px solid #444;border-radius:6px} .meta{opacity:0.8;font-size:0.9em}</style>",
        "</head><body>",
        "<h1>Many-Worlds Comic Strip</h1>",
        f"<p class='meta'>Palette: {palette}</p>",
        "<div class='row'>"
    ]
    for i, f in enumerate(files):
        pd = panels_data[i]
        html.append("<div>")
        html.append(f"<img src='{f.as_posix()}' alt='Panel {i+1}'/>")
        html.append(f"<div class='meta'>Panel {i+1}: {pd['camera']} · {pd['setting']} · {pd['emotion']} · {pd['action']}</div>")
        html.append("</div>")
    html += ["</div></body></html>"]
    out = OUTPUT_DIR / "index.html"
    out.write_text("\n".join(html), encoding="utf-8")
    return out

# -----------------------------
# MAIN
# -----------------------------
def main():
    if SEED is not None:
        np.random.seed(SEED)

    qc, shapes = build_circuit(PANELS)
    bitstring = run_one_shot(qc)

    palette, panels_data = decode(bitstring, shapes, panels=PANELS)
    prompts = [panel_prompt(palette, p) for p in panels_data]

    print("\n=== QUANTUM COLLAPSE RESULTS ===")
    print("Bitstring (LSB->MSB):", bitstring)
    print("Palette:", palette)
    for p, pr in zip(panels_data, prompts):
        print(f"Panel {p['panel_index']}: {p} \nPrompt: {pr}\n")

    files = generate_panel_images(prompts)
    html = write_html_strip(files, palette, panels_data)

    print("\nSaved panels:")
    for f in files:
        print("  ", f)
    print("Open:")
    print("  ", html)

if __name__ == "__main__":
    main()
```

---

# How This Plays With Your Existing IBM Runtime Script

Your `/mnt/data/run_ibm_quantum.py` already uses `QiskitRuntimeService`, `Session`, and `Sampler`. The reference above follows the same pattern:

* Uses `Sampler` (low-latency, great for bitstrings).
* Keeps circuit **shallow** (a couple entangling layers).
* Requests **one shot** (purist one-world collapse). You can switch to a named backend or reuse your least-busy logic.

If you prefer, copy the **`run_one_shot`** body into your existing runtime wrapper to keep credentials/config unified.

---

# Tuning for Real Hardware

* **Depth control:** Favor `CX`, `CZ`, small-angle CR\* gates; avoid long chains.
* **Layout:** Let transpiler do its thing initially; if you see heavy routing, reduce panel count or collapse some registers.
* **Shots:** Philosophically one shot. Practically, you can run `shots=8` and pick the **highest-probability** distribution outcome (already implemented) to resist rare readout glitches.
* **Qubit count:** 46 in this template. Can scale to \~80 by adding registers (e.g., props/weather) as long as depth stays low.

---

# Variations You Might Enjoy

* **Speech balloons**: Generate short captions with the same bit-driven slots and overlay them locally with PIL; keep images “speechless” from Gemini to avoid text rendering artifacts.
* **Palette locking**: Hash the style bits to a fixed trio of hex colors and add them explicitly to the prompt for even stronger consistency.
* **Many-runs exhibit**: Run 5 times. Hang all 5 strips side-by-side as “parallel worlds.” Each is a different universe born from a different collapse.
* **Parallel universes**: Run 5 times. Hang all 5 strips side-by-side as “parallel worlds.” Each is a different universe born from a different collapse.
