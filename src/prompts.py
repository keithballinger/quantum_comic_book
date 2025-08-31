"""
Prompt generation system for quantum comic book.

This module decodes quantum measurement results into narrative elements
and generates prompts for image generation.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from src.quantum_circuit import CircuitRegisters


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


@dataclass
class PanelData:
    """Data for a single comic panel."""

    panel_index: int
    action: str
    emotion: str
    camera: str
    setting: str
    character_focus: int  # Which character is focus (0 or 1)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "panel_index": self.panel_index,
            "action": self.action,
            "emotion": self.emotion,
            "camera": self.camera,
            "setting": self.setting,
            "character_focus": self.character_focus,
        }


@dataclass
class ComicNarrative:
    """Complete narrative structure for a comic."""

    style_palette: str
    panels: List[PanelData]
    character_bio: str
    base_style: str
    bitstring: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "style_palette": self.style_palette,
            "panels": [p.to_dict() for p in self.panels],
            "character_bio": self.character_bio,
            "base_style": self.base_style,
            "bitstring": self.bitstring,
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
    character_style: str = "default",
    art_style: str = "comic",
) -> Tuple[ComicNarrative, List[str]]:
    """
    Decode quantum measurement into comic narrative and prompts.

    Args:
        bitstring: Quantum measurement result
        registers: Circuit register information
        character_style: Character bio style
        art_style: Art style

    Returns:
        Tuple of (narrative, prompts)
    """
    # Decode bitstring
    decoder = PromptDecoder(registers, character_style, art_style)
    narrative = decoder.decode_bitstring(bitstring)

    # Generate prompts
    generator = PromptGenerator(narrative)
    prompts = generator.generate_all_prompts()

    return narrative, prompts
