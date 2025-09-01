# src/gemini_text.py
import os
import json
from google import genai
from google.genai import types
from typing import Dict, List, Any

from src.config import Config

MODEL_TEXT = "gemini-2.5-flash"

SYSTEM_RULES = (
  "You write compact JSON scripts for multi-panel comics. "
  "Obey all constraints exactly: dialogue acts, rhetorical devices, line limits, recurring phrases. "
  "Return ONLY valid JSON."
)

def build_script_prompt(title, palette, character_bio, panels_data, constraints):
    # Pack constraints explicitly; we’ll ask for strict adherence.
    schema_hint = {
      "title": "string",
      "palette": "string",
      "panels": [{
        "index": "int",
        "setting": "string",
        "camera": "string",
        "beats": ["string", "string?"],
        "dialogue": [{"speaker":"Engineer|Law Student","text":"<=110 chars"}],
        "caption": "string?",
        "sfx": ["string?"]
      }]
    }

    return (
      f"{SYSTEM_RULES}\n"
      f"Title seed: {title}\nPalette: {palette}\nTone: {constraints['tone']}\n"
      f"Characters: {character_bio}\n"
      f"Panel slots:\n{json.dumps([{'index':p['panel_index'],'setting':p['setting'],'camera':p['camera'],'emotion':p['emotion'],'action':p['action']} for p in panels_data])}\n"
      f"Quantum constraints:\n{json.dumps(constraints, ensure_ascii=False)}\n"
      f"Recurring phrase (weave subtly when 'must_use_recurring' is true): \"{constraints['recurring_phrase']}\"\n"
      f"Use rhetorical_device exactly as assigned per panel. "
      f"For 'anaphora', start the line with the same 1–3 words across appearances. "
      f"For 'antimetabole', invert a phrase later in the panel. "
      f"For 'call-and-response', ensure the two speakers mirror/rebut. "
      f"For 'chiasmus', AB-BA syntactic echo within or across the two lines. "
      f"Enforce max_lines_per_speaker and max_chars_per_line per panel.\n"
      f"Output JSON only in this schema:\n{json.dumps(schema_hint)}"
    )

import re

def generate_narrative(config: Config, title: str, palette: str, character_bio: str, panels_data: List[Dict[str, Any]], constraints: Dict[str, Any]):
    client = genai.Client(api_key=config.gemini_api_key)
    gen_config = types.GenerateContentConfig(
        temperature=0.85, top_p=0.9
    )
    contents = [types.Content(
        role="user",
        parts=[types.Part.from_text(build_script_prompt(title, palette, character_bio, panels_data, constraints))]
    )]
    resp = client.models.generate_content(model=MODEL_TEXT, contents=contents, generation_config=gen_config)
    text = resp.text
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # The model sometimes returns the JSON wrapped in markdown, so we extract it
        json_match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = text
        data = json.loads(json_str)
    return data
