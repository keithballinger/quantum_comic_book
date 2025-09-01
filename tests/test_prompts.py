"""
Tests for prompt generation system.
"""

import pytest

from src.quantum_circuit import CircuitRegisters
from src.prompts import (
    PanelData,
    ComicNarrative,
    PromptDecoder,
    PromptGenerator,
    decode_quantum_result,
    ACTIONS,
    EMOTIONS,
    CAMERA_ANGLES,
    STYLE_PALETTES,
    SETTINGS,
)


class TestPanelData:
    """Test PanelData class."""

    def test_panel_data_creation(self):
        """Test creating panel data."""
        panel = PanelData(
            panel_index=1,
            action="approaches",
            emotion="curious",
            camera="medium shot",
            setting="quiet corridor",
            character_focus=0,
        )

        assert panel.panel_index == 1
        assert panel.action == "approaches"
        assert panel.emotion == "curious"
        assert panel.camera == "medium shot"
        assert panel.setting == "quiet corridor"
        assert panel.character_focus == 0

    def test_panel_data_to_dict(self):
        """Test converting panel data to dictionary."""
        panel = PanelData(
            panel_index=2,
            action="reaches out",
            emotion="determined",
            camera="close-up",
            setting="rooftop",
            character_focus=1,
        )

        data = panel.to_dict()

        assert data["panel_index"] == 2
        assert data["action"] == "reaches out"
        assert data["emotion"] == "determined"
        assert data["camera"] == "close-up"
        assert data["setting"] == "rooftop"
        assert data["character_focus"] == 1


class TestComicNarrative:
    """Test ComicNarrative class."""

    def test_comic_narrative_creation(self):
        """Test creating comic narrative."""
        panels = [
            PanelData(1, "action1", "emotion1", "camera1", "setting1", 0),
            PanelData(2, "action2", "emotion2", "camera2", "setting2", 1),
        ]

        narrative = ComicNarrative(
            style_palette="noir ink",
            panels=panels,
            character_bio="Test characters",
            base_style="Comic style",
            bitstring="101010",
        )

        assert narrative.style_palette == "noir ink"
        assert len(narrative.panels) == 2
        assert narrative.character_bio == "Test characters"
        assert narrative.base_style == "Comic style"
        assert narrative.bitstring == "101010"

    def test_comic_narrative_to_dict(self):
        """Test converting narrative to dictionary."""
        panels = [
            PanelData(1, "action1", "emotion1", "camera1", "setting1", 0),
        ]

        narrative = ComicNarrative(
            style_palette="watercolor",
            panels=panels,
            character_bio="Bio",
            base_style="Style",
            bitstring="111",
        )

        data = narrative.to_dict()

        assert data["style_palette"] == "watercolor"
        assert len(data["panels"]) == 1
        assert data["panels"][0]["panel_index"] == 1
        assert data["character_bio"] == "Bio"
        assert data["base_style"] == "Style"
        assert data["bitstring"] == "111"


class TestPromptDecoder:
    """Test PromptDecoder functionality."""

    @pytest.fixture
    def registers(self):
        """Create test registers for 3 panels."""
        return CircuitRegisters(
            time_qubits=3,
            action_qubits=6,  # 2 bits per panel
            emotion_qubits=6,  # 2 bits per panel
            camera_qubits=6,  # 2 bits per panel
            style_qubits=4,
            total_qubits=25,
        )

    @pytest.fixture
    def decoder(self, registers):
        """Create prompt decoder."""
        return PromptDecoder(registers)

    def test_decoder_initialization(self, decoder):
        """Test decoder initialization."""
        assert decoder.character_bio is not None
        assert decoder.base_style is not None

    def test_extract_register_bits(self, decoder):
        """Test extracting register bits from bitstring."""
        # Create a test bitstring (25 bits total)
        # Layout: TTT AAAAAA EEEEEE CCCCCC SSSS
        bitstring = "101" + "110011" + "001100" + "101010" + "1101"

        time_bits = decoder._extract_time_bits(bitstring)
        action_bits = decoder._extract_action_bits(bitstring)
        emotion_bits = decoder._extract_emotion_bits(bitstring)
        camera_bits = decoder._extract_camera_bits(bitstring)
        style_bits = decoder._extract_style_bits(bitstring)

        assert time_bits == "101"
        assert action_bits == "110011"
        assert emotion_bits == "001100"
        assert camera_bits == "101010"
        assert style_bits == "1101"

    def test_get_panel_value(self, decoder):
        """Test extracting panel-specific values."""
        bits = "110010"  # 3 panels, 2 bits each

        assert decoder._get_panel_value(bits, 0) == 3  # "11" = 3
        assert decoder._get_panel_value(bits, 1) == 0  # "00" = 0
        assert decoder._get_panel_value(bits, 2) == 2  # "10" = 2

    def test_decode_style(self, decoder):
        """Test style decoding."""
        # Test various 4-bit style values
        assert decoder._decode_style("0000") == STYLE_PALETTES[0]
        assert decoder._decode_style("0001") == STYLE_PALETTES[1]
        assert decoder._decode_style("1111") == STYLE_PALETTES[15]
        assert decoder._decode_style("0101") == STYLE_PALETTES[5]

    def test_determine_setting(self, decoder):
        """Test setting determination based on time bits."""
        time_bits = "101"

        # Panel 0: prev=0, curr=1, next=0
        setting0 = decoder._determine_setting(time_bits, 0)
        assert setting0 in SETTINGS

        # Panel 1: prev=1, curr=0, next=1
        setting1 = decoder._determine_setting(time_bits, 1)
        assert setting1 in SETTINGS

        # Panel 2: prev=0, curr=1, next=0
        setting2 = decoder._determine_setting(time_bits, 2)
        assert setting2 in SETTINGS

    def test_decode_bitstring(self, decoder):
        """Test complete bitstring decoding."""
        # Create a specific bitstring
        bitstring = "101" + "110010" + "001101" + "101100" + "0011"

        narrative = decoder.decode_bitstring(bitstring)

        assert isinstance(narrative, ComicNarrative)
        assert len(narrative.panels) == 3
        assert narrative.style_palette == STYLE_PALETTES[3]  # "0011" = 3

        # Check first panel
        panel0 = narrative.panels[0]
        assert panel0.panel_index == 1
        assert panel0.action == ACTIONS[3]  # "11" = 3
        assert panel0.emotion == EMOTIONS[0]  # "00" = 0
        assert panel0.camera == CAMERA_ANGLES[2]  # "10" = 2
        assert panel0.character_focus == 0

        # Check second panel
        panel1 = narrative.panels[1]
        assert panel1.panel_index == 2
        assert panel1.action == ACTIONS[0]  # "00" = 0
        assert panel1.emotion == EMOTIONS[3]  # "11" = 3
        assert panel1.camera == CAMERA_ANGLES[3]  # "11" = 3
        assert panel1.character_focus == 1


class TestPromptGenerator:
    """Test PromptGenerator functionality."""

    @pytest.fixture
    def narrative(self):
        """Create test narrative."""
        panels = [
            PanelData(
                panel_index=1,
                action="approaches cautiously",
                emotion="curious",
                camera="medium shot",
                setting="quiet corridor",
                character_focus=0,
            ),
            PanelData(
                panel_index=2,
                action="reaches out",
                emotion="determined",
                camera="close-up",
                setting="rooftop",
                character_focus=1,
            ),
        ]

        return ComicNarrative(
            style_palette="noir ink with deep shadows",
            panels=panels,
            character_bio="Two protagonists in urban setting",
            base_style="Comic book art with clean borders",
            bitstring="101010",
        )

    @pytest.fixture
    def generator(self, narrative):
        """Create prompt generator."""
        return PromptGenerator(narrative)

    def test_generate_panel_prompt_first(self, generator):
        """Test generating prompt for first panel."""
        panel = generator.narrative.panels[0]
        prompt = generator.generate_panel_prompt(panel)

        assert "Comic book art" in prompt
        assert "Two protagonists" in prompt
        assert "Panel 1" in prompt
        assert "medium shot" in prompt
        assert "quiet corridor" in prompt
        assert "curious" in prompt
        assert "approaches cautiously" in prompt
        assert "noir ink" in prompt

    def test_generate_panel_prompt_with_context(self, generator):
        """Test generating prompt with previous context."""
        panel = generator.narrative.panels[1]
        previous = generator.narrative.panels[0]

        prompt = generator.generate_panel_prompt(
            panel,
            include_previous_context=True,
            previous_panel=previous,
        )

        assert "Panel 2" in prompt
        assert "close-up" in prompt
        assert "rooftop" in prompt
        assert "determined" in prompt
        assert "reaches out" in prompt
        assert "previous scene" in prompt
        assert "curious" in prompt  # From previous panel
        assert "visual continuity" in prompt

    def test_generate_all_prompts(self, generator):
        """Test generating all panel prompts."""
        prompts = generator.generate_all_prompts()

        assert len(prompts) == 2

        # First prompt should not have context
        assert "previous scene" not in prompts[0]

        # Second prompt should have context
        assert "previous scene" in prompts[1]

    def test_generate_title_prompt(self, generator):
        """Test generating title card prompt."""
        prompt = generator.generate_title_prompt()

        assert "Title card" in prompt
        assert "quantum comic" in prompt
        assert "curious to determined" in prompt
        assert "noir ink" in prompt
        assert "quantum imagery" in prompt


class TestDecodeQuantumResult:
    """Test the main decoding function."""

    def test_decode_quantum_result(self):
        """Test complete quantum result decoding."""
        registers = CircuitRegisters(
            time_qubits=2,
            action_qubits=4,
            emotion_qubits=4,
            camera_qubits=4,
            style_qubits=4,
            total_qubits=18,
        )

        # Create test bitstring
        bitstring = "10" + "1100" + "0011" + "1010" + "0101"

        narrative, prompts = decode_quantum_result(
            bitstring,
            registers,
            character_style="default",
            art_style="comic",
        )

        assert isinstance(narrative, ComicNarrative)
        assert len(narrative.panels) == 2
        assert len(prompts) == 2

        # Check narrative structure
        assert narrative.style_palette == STYLE_PALETTES[5]  # "0101" = 5
        assert narrative.bitstring == bitstring

        # Check prompts
        for prompt in prompts:
            assert isinstance(prompt, str)
            assert len(prompt) > 0


@pytest.mark.parametrize(
    "bits,expected_value",
    [
        ("00", 0),
        ("01", 1),
        ("10", 2),
        ("11", 3),
    ],
)
def test_bit_to_value_conversion(bits, expected_value):
    """Test converting 2-bit strings to values."""
    value = int(bits, 2)
    assert value == expected_value


@pytest.mark.parametrize("style_id", range(16))
def test_all_style_palettes(style_id):
    """Test that all style palette IDs map to valid descriptions."""
    assert style_id in STYLE_PALETTES
    assert isinstance(STYLE_PALETTES[style_id], str)
    assert len(STYLE_PALETTES[style_id]) > 0


@pytest.mark.parametrize("num_panels", [1, 3, 6, 12])
def test_various_panel_counts(num_panels):
    """Test decoding with various panel counts."""
    registers = CircuitRegisters(
        time_qubits=num_panels,
        action_qubits=2 * num_panels,
        emotion_qubits=2 * num_panels,
        camera_qubits=2 * num_panels,
        style_qubits=4,
        total_qubits=num_panels + (2 * num_panels * 3) + 4,
    )

    # Create appropriate bitstring
    total_bits = registers.total_qubits
    bitstring = "1" * total_bits

    decoder = PromptDecoder(registers)
    narrative = decoder.decode_bitstring(bitstring)

    assert len(narrative.panels) == num_panels
    for i, panel in enumerate(narrative.panels):
        assert panel.panel_index == i + 1
