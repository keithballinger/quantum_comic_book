"""
Prompt generation system for quantum comic book.

This module decodes quantum measurement results into narrative elements
and generates prompts for image generation.
"""

import json
import re
from typing import List, Dict, Any, Tuple

from src.quantum_circuit import CircuitRegisters
from src.gemini_client import GeminiClient
from src.config import Config
from src.narrative import ComicNarrative, PanelData


# Lookup tables for narrative elements
ACTIONS = [
    "approaches cautiously",
    "reaches out",
    "pauses to reflect",
    "turns away",
]

EMOTIONS = [
    "curious",
    "determined",
    "conflicted",
    "relieved",
]

CAMERA_ANGLES = [
    "medium shot",
    "wide shot",
    "close-up",
    "over-the-shoulder",
]

# Extended palette options for 4-bit style register (16 possibilities)
STYLE_PALETTES = {
    0: "noir ink with deep shadows",
    1: "muted watercolor wash",
    2: "bright pop colors",
    3: "newsprint halftone dots",
    4: "sepia dusk tones",
    5: "cool cyan and magenta",
    6: "warm amber and teal",
    7: "high-contrast black and white",
    8: "pastel palette",
    9: "gritty urban palette",
    10: "sunrise golds",
    11: "rain-slick night palette",
    12: "wintry bluish palette",
    13: "late-afternoon amber",
    14: "neon city palette",
    15: "soft neutral palette",
}

# Setting progression based on time bits
SETTINGS = [
    "quiet indoor corridor",
    "narrow alley",
    "city rooftop",
    "side street",
    "bridge overlook",
    "dawn-lit avenue",
    "underground passage",
    "abandoned warehouse",
]

# Character descriptions
CHARACTER_BIOS = {
    "default": (
        "Two protagonists: 1) a thoughtful engineer in a minimalist jacket, "
        "2) a bold law student with a light scarf. Contemporary urban setting."
    ),
    "scifi": (
        "Two explorers: 1) a quantum physicist with augmented vision gear, "
        "2) a reality debugger with temporal sensors. Near-future metropolis."
    ),
    "noir": (
        "Two investigators: 1) a weary detective in a worn trench coat, "
        "2) a sharp-eyed journalist with a press badge. 1940s city streets."
    ),
}

# Base style descriptions
BASE_STYLES = {
    "comic": (
        "Comic book art, inked lines, halftone shading, clean panel borders, "
        "cinematic lighting. Keep characters consistent across panels "
        "(hair, clothing, colors, face)."
    ),
    "graphic_novel": (
        "Graphic novel style, detailed linework, atmospheric shadows, "
        "sophisticated compositions. Maintain character identity and "
        "visual continuity throughout."
    ),
    "manga": (
        "Manga style artwork, expressive characters, dynamic angles, "
        "speed lines where appropriate. Consistent character designs "
        "and clothing details."
    ),
}





class PromptDecoder:
    """Decodes quantum bitstrings into narrative prompts."""

    def __init__(
        self,
        registers: CircuitRegisters,
        character_style: str = "default",
        art_style: str = "comic",
    ):
        """
        Initialize prompt decoder.

        Args:
            registers: Circuit register information
            character_style: Character bio style to use
            art_style: Art style to use
        """
        self.registers = registers
        self.character_bio = CHARACTER_BIOS.get(
            character_style, CHARACTER_BIOS["default"]
        )
        self.base_style = BASE_STYLES.get(art_style, BASE_STYLES["comic"])

    def decode_bitstring(self, bitstring: str) -> ComicNarrative:
        """
        Decode quantum bitstring into narrative structure.

        Args:
            bitstring: Measurement result bitstring

        Returns:
            Complete comic narrative
        """
        # Extract register slices
        time_bits = self._extract_time_bits(bitstring)
        action_bits = self._extract_action_bits(bitstring)
        emotion_bits = self._extract_emotion_bits(bitstring)
        camera_bits = self._extract_camera_bits(bitstring)
        style_bits = self._extract_style_bits(bitstring)

        # Decode global style
        style_palette = self._decode_style(style_bits)

        # Decode each panel
        panels = []
        num_panels = self.registers.time_qubits

        for panel_idx in range(num_panels):
            # Extract panel-specific bits
            action_val = self._get_panel_value(action_bits, panel_idx)
            emotion_val = self._get_panel_value(emotion_bits, panel_idx)
            camera_val = self._get_panel_value(camera_bits, panel_idx)

            # Determine setting based on time progression
            setting = self._determine_setting(time_bits, panel_idx)

            # Determine character focus (alternates based on panel parity)
            character_focus = panel_idx % 2

            panel = PanelData(
                panel_index=panel_idx + 1,
                action=ACTIONS[action_val % len(ACTIONS)],
                emotion=EMOTIONS[emotion_val % len(EMOTIONS)],
                camera=CAMERA_ANGLES[camera_val % len(CAMERA_ANGLES)],
                setting=setting,
                character_focus=character_focus,
            )
            panels.append(panel)

        return ComicNarrative(
            title="",  # Title will be generated by Gemini
            style_palette=style_palette,
            panels=panels,
            character_bio=self.character_bio,
            base_style=self.base_style,
            bitstring=bitstring,
        )

    def _extract_time_bits(self, bitstring: str) -> str:
        """Extract time register bits."""
        start = self.registers.time_start
        end = start + self.registers.time_qubits
        return bitstring[start:end]

    def _extract_action_bits(self, bitstring: str) -> str:
        """Extract action register bits."""
        start = self.registers.action_start
        end = start + self.registers.action_qubits
        return bitstring[start:end]

    def _extract_emotion_bits(self, bitstring: str) -> str:
        """Extract emotion register bits."""
        start = self.registers.emotion_start
        end = start + self.registers.emotion_qubits
        return bitstring[start:end]

    def _extract_camera_bits(self, bitstring: str) -> str:
        """Extract camera register bits."""
        start = self.registers.camera_start
        end = start + self.registers.camera_qubits
        return bitstring[start:end]

    def _extract_style_bits(self, bitstring: str) -> str:
        """Extract style register bits."""
        start = self.registers.style_start
        end = start + self.registers.style_qubits
        return bitstring[start:end]

    def _get_panel_value(self, bits: str, panel_idx: int) -> int:
        """Get 2-bit value for a specific panel."""
        start = panel_idx * 2
        if start + 2 <= len(bits):
            panel_bits = bits[start : start + 2]
            # Convert binary string to integer
            return int(panel_bits, 2) if panel_bits else 0
        return 0

    def _decode_style(self, style_bits: str) -> str:
        """Decode style bits into palette description."""
        if not style_bits:
            return STYLE_PALETTES[0]

        # Convert 4-bit style to integer
        style_id = int(style_bits, 2) if style_bits else 0
        return STYLE_PALETTES.get(style_id, STYLE_PALETTES[0])

    def _determine_setting(self, time_bits: str, panel_idx: int) -> str:
        """
        Determine setting based on time progression.

        Uses current time bit and neighbors to create progression.
        """
        if panel_idx >= len(time_bits):
            return SETTINGS[0]

        # Get current and neighboring time bits
        current = int(time_bits[panel_idx]) if time_bits[panel_idx] else 0

        # Get previous bit (or 0 if first panel)
        prev_bit = int(time_bits[panel_idx - 1]) if panel_idx > 0 else 0

        # Get next bit (or 0 if last panel)
        next_bit = (
            int(time_bits[panel_idx + 1]) if panel_idx < len(time_bits) - 1 else 0
        )

        # Combine bits to select setting (creates narrative progression)
        setting_idx = (prev_bit * 2 + current * 1 + next_bit) % len(SETTINGS)

        return SETTINGS[setting_idx]


class PromptGenerator:
    """Generates image generation prompts from narrative structure."""

    def __init__(self, narrative: ComicNarrative):
        """
        Initialize prompt generator.

        Args:
            narrative: Comic narrative structure
        """
        self.narrative = narrative

    def generate_narrative_prompt(self) -> str:
        """
        Generate a prompt for the text model to create a full narrative.

        Returns:
            A prompt for narrative generation.
        """
        raw_panels_data = []
        for panel in self.narrative.panels:
            raw_panels_data.append(
                f"Panel {panel.panel_index}: "
                f"Action='{panel.action}', "
                f"Emotion='{panel.emotion}', "
                f"Camera='{panel.camera}', "
                f"Setting='{panel.setting}', "
                f"CharacterFocus={panel.character_focus + 1}"
            )
        
        raw_narrative = "\n".join(raw_panels_data)

        prompt = f"""
You are a creative writer for a quantum comic book.
Your task is to take a raw, quantum-generated narrative structure and flesh it out into a complete, compelling comic book story with a title and dialogue.

**Instructions:**
1.  Read the provided "Raw Quantum Narrative". This contains the core emotional and action beats for each panel, derived from a real quantum computer.
2.  Invent a short, catchy title for the comic (2-5 words).
3.  For each panel, write a short line of dialogue for the focused character. The dialogue should match the character's emotion and action. If a panel should be silent, provide an empty string for the dialogue.
4.  The story should have a clear, albeit short, arc.
5.  Return a single JSON object with the following structure:
    {{
      "title": "Your Catchy Title",
      "panels": [
        {{
          "panel_index": 1,
          "dialogue": "Dialogue for panel 1."
        }},
        {{
          "panel_index": 2,
          "dialogue": "Dialogue for panel 2."
        }},
        ...
      ]
    }}

**Character & Style Guide:**
-   **Characters:** {self.narrative.character_bio}
-   **Art Style:** {self.narrative.base_style}
-   **Color Palette:** {self.narrative.style_palette}

**Raw Quantum Narrative:**
{raw_narrative}

**Output (JSON only):**
"""
        return prompt

    def generate_panel_prompt(
        self,
        panel: PanelData,
        include_previous_context: bool = False,
        previous_panel: PanelData = None,
    ) -> str:
        """
        Generate prompt for a single panel.

        Args:
            panel: Panel data
            include_previous_context: Whether to reference previous panel
            previous_panel: Previous panel data for continuity

        Returns:
            Image generation prompt
        """
        # Base style and character setup
        prompt_parts = [
            self.narrative.base_style,
            self.narrative.character_bio,
        ]

        # Panel-specific description
        panel_desc = (
            f"Panel {panel.panel_index}: {panel.camera} in a {panel.setting}. "
            f"Primary emotion: {panel.emotion}. "
            f"Action: Character {panel.character_focus + 1} {panel.action}. "
        )
        
        # Add dialogue to the prompt
        if panel.dialogue:
            panel_desc += f'A speech bubble contains the text: "{panel.dialogue}"'

        prompt_parts.append(panel_desc)

        # Add continuity context if this isn't the first panel
        if include_previous_context and previous_panel:
            continuity = (
                f"This follows from the previous scene where character "
                f"{previous_panel.character_focus + 1} was {previous_panel.emotion} "
                f"in a {previous_panel.setting}. Maintain visual continuity."
            )
            prompt_parts.append(continuity)

        # Style palette
        prompt_parts.append(f"Use {self.narrative.style_palette}.")

        return " ".join(prompt_parts)

    def generate_all_prompts(self) -> List[str]:
        """
        Generate prompts for all panels.

        Returns:
            List of prompts for each panel
        """
        prompts = []

        for i, panel in enumerate(self.narrative.panels):
            include_context = i > 0
            previous = self.narrative.panels[i - 1] if i > 0 else None

            prompt = self.generate_panel_prompt(
                panel,
                include_previous_context=include_context,
                previous_panel=previous,
            )
            prompts.append(prompt)

        return prompts

    def generate_title_prompt(self) -> str:
        """
        Generate a prompt for a title card.

        Returns:
            Title card prompt
        """
        # Analyze narrative arc
        first_emotion = self.narrative.panels[0].emotion
        last_emotion = self.narrative.panels[-1].emotion

        prompt = (
            f"{self.narrative.base_style} "
            f"Title card for a quantum comic. "
            f"The story moves from {first_emotion} to {last_emotion}. "
            f"Use {self.narrative.style_palette}. "
            f"Include subtle quantum imagery (entangled particles, wave functions). "
            f"Text area left blank for title to be added later."
        )

        return prompt


def decode_quantum_result(
    bitstring: str,
    registers: CircuitRegisters,
    config: Config,
) -> Tuple[ComicNarrative, List[str]]:
    """
    Decode quantum measurement into comic narrative and prompts.

    Args:
        bitstring: Quantum measurement result
        registers: Circuit register information
        config: Application configuration

    Returns:
        Tuple of (narrative, prompts)
    """
    # 1. Decode bitstring into raw narrative structure
    decoder = PromptDecoder(registers, config.character_style, config.art_style)
    raw_narrative = decoder.decode_bitstring(bitstring)

    # 2. Generate a prompt to flesh out the narrative
    prompt_generator = PromptGenerator(raw_narrative)
    narrative_prompt = prompt_generator.generate_narrative_prompt()

    # 3. Call Gemini to get the full narrative with dialogue
    gemini_client = GeminiClient(config)
    full_narrative_json = gemini_client.generate_narrative(narrative_prompt)
    
    # 4. Parse the response and update the narrative object
    # The model sometimes returns the JSON wrapped in markdown, so we extract it
    json_match = re.search(r"```json\n(.*?)\n```", full_narrative_json, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_str = full_narrative_json

    narrative_data = json.loads(json_str)

    raw_narrative.title = narrative_data.get("title", "Quantum Comic")
    
    panel_dialogues = {p['panel_index']: p['dialogue'] for p in narrative_data.get("panels", [])}
    for panel in raw_narrative.panels:
        panel.dialogue = panel_dialogues.get(panel.panel_index, "")

    # 5. Generate final image prompts from the rich narrative
    final_prompts = prompt_generator.generate_all_prompts()

    return raw_narrative, final_prompts
