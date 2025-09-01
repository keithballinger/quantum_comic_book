from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class PanelData:
    """Data for a single comic panel."""

    panel_index: int
    action: str
    emotion: str
    camera: str
    setting: str
    character_focus: int  # Which character is focus (0 or 1)
    dialogue: str = ""  # New field for dialogue

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "panel_index": self.panel_index,
            "action": self.action,
            "emotion": self.emotion,
            "camera": self.camera,
            "setting": self.setting,
            "character_focus": self.character_focus,
            "dialogue": self.dialogue,
        }


@dataclass
class ComicNarrative:
    """Complete narrative structure for a comic."""

    title: str  # New field for the title
    style_palette: str
    panels: List[PanelData]
    character_bio: str
    base_style: str
    bitstring: str
    quantum_constraints: Optional[Dict[str, Any]] = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "style_palette": self.style_palette,
            "panels": [p.to_dict() for p in self.panels],
            "character_bio": self.character_bio,
            "base_style": self.base_style,
            "bitstring": self.bitstring,
            "quantum_constraints": self.quantum_constraints,
        }
