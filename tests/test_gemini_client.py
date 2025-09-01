"""
Tests for Gemini image generation client.
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import pytest
from PIL import Image

from src.config import Config
from src.prompts import ComicNarrative, PanelData
from src.gemini_client import (
    GeneratedPanel,
    ComicStrip,
    GeminiImageGenerator,
    GeminiError,
    create_comic_strip_from_bitstring,
    create_combined_strip_image,
)


class TestGeneratedPanel:
    """Test GeneratedPanel functionality."""
    
    @pytest.fixture
    def sample_image_data(self):
        """Create sample image data."""
        # Create a small test image
        img = Image.new('RGB', (100, 100), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        return buffer.getvalue()
    
    @pytest.fixture
    def panel(self, sample_image_data):
        """Create test panel."""
        return GeneratedPanel(
            panel_index=1,
            image_data=sample_image_data,
            mime_type="image/jpeg",
            prompt="Test prompt",
            metadata={"test": "data"},
        )
    
    def test_panel_creation(self, panel):
        """Test creating generated panel."""
        assert panel.panel_index == 1
        assert panel.mime_type == "image/jpeg"
        assert panel.prompt == "Test prompt"
        assert panel.metadata["test"] == "data"
        assert panel.file_path is None
    
    def test_panel_save(self, panel, tmp_path):
        """Test saving panel to file."""
        file_path = tmp_path / "test_panel"
        saved_path = panel.save(file_path)
        
        assert saved_path.exists()
        assert saved_path.suffix == ".jpg"  # .jpeg gets normalized to .jpg
        assert panel.file_path == saved_path
        
        # Verify file content
        saved_data = saved_path.read_bytes()
        assert len(saved_data) > 0
    
    def test_panel_save_with_extension(self, panel, tmp_path):
        """Test saving panel with explicit extension."""
        file_path = tmp_path / "test_panel.jpg"
        saved_path = panel.save(file_path)
        
        assert saved_path == file_path
        assert saved_path.suffix == ".jpg"
    
    def test_panel_to_pil_image(self, panel):
        """Test converting panel to PIL Image."""
        img = panel.to_pil_image()
        
        assert isinstance(img, Image.Image)
        assert img.size == (100, 100)


class TestComicStrip:
    """Test ComicStrip functionality."""
    
    @pytest.fixture
    def sample_panels(self):
        """Create sample panels."""
        panels = []
        for i in range(3):
            img = Image.new('RGB', (100, 100), color=['red', 'green', 'blue'][i])
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            
            panel = GeneratedPanel(
                panel_index=i + 1,
                image_data=buffer.getvalue(),
                mime_type="image/jpeg",
                prompt=f"Panel {i + 1}",
            )
            panels.append(panel)
        return panels
    
    @pytest.fixture
    def sample_narrative(self):
        """Create sample narrative."""
        panels = [
            PanelData(1, "action1", "emotion1", "camera1", "setting1", 0),
            PanelData(2, "action2", "emotion2", "camera2", "setting2", 1),
            PanelData(3, "action3", "emotion3", "camera3", "setting3", 0),
        ]
        
        return ComicNarrative(
            style_palette="test palette",
            panels=panels,
            character_bio="Test characters",
            base_style="Test style",
            bitstring="101010",
        )
    
    @pytest.fixture
    def comic_strip(self, sample_panels, sample_narrative):
        """Create test comic strip."""
        return ComicStrip(
            panels=sample_panels,
            narrative=sample_narrative,
        )
    
    def test_comic_strip_creation(self, comic_strip):
        """Test creating comic strip."""
        assert len(comic_strip.panels) == 3
        assert comic_strip.narrative is not None
        assert comic_strip.title_card is None
    
    def test_save_all_panels(self, comic_strip, tmp_path):
        """Test saving all panels."""
        output_dir = tmp_path / "comic_output"
        saved_paths = comic_strip.save_all(output_dir)
        
        assert len(saved_paths) == 3
        assert output_dir.exists()
        
        # Check each panel was saved
        for i, path in enumerate(saved_paths):
            assert path.exists()
            assert f"panel_{i+1:02d}" in path.stem
        
        # Check metadata was saved
        metadata_path = output_dir / "metadata.txt"
        assert metadata_path.exists()
        
        metadata_content = metadata_path.read_text()
        assert "101010" in metadata_content  # Bitstring
        assert "test palette" in metadata_content  # Style
    
    def test_save_with_title_card(self, comic_strip, sample_panels, tmp_path):
        """Test saving with title card."""
        # Add title card
        img = Image.new('RGB', (100, 100), color='yellow')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        
        comic_strip.title_card = GeneratedPanel(
            panel_index=0,
            image_data=buffer.getvalue(),
            mime_type="image/jpeg",
            prompt="Title card",
        )
        
        output_dir = tmp_path / "comic_with_title"
        saved_paths = comic_strip.save_all(output_dir)
        
        assert len(saved_paths) == 4  # 3 panels + title
        assert any("title_card" in str(p) for p in saved_paths)


class TestGeminiImageGenerator:
    """Test Gemini image generator."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(
            ibm_api_key="test_ibm",
            gemini_api_key="test_gemini_key",
            max_retries=2,
        )
    
    @pytest.fixture
    def mock_client(self):
        """Create mock Gemini client."""
        return MagicMock()
    
    @patch('src.gemini_client.genai.Client')
    def test_generator_initialization(self, mock_client_class, config):
        """Test generator initialization."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        generator = GeminiImageGenerator(config)
        
        assert generator.config == config
        assert generator.client == mock_client
        mock_client_class.assert_called_once_with(api_key="test_gemini_key")
    
    @patch('src.gemini_client.genai.Client')
    def test_connection_failure(self, mock_client_class, config):
        """Test handling connection failure."""
        mock_client_class.side_effect = Exception("Connection failed")
        
        with pytest.raises(GeminiError, match="Failed to connect"):
            GeminiImageGenerator(config)
    
    @patch('src.gemini_client.genai.Client')
    def test_generate_panel_success(self, mock_client_class, config):
        """Test successful panel generation."""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        generator = GeminiImageGenerator(config)
        
        # Create mock response
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.data = b"test_image_data"
        mock_part.inline_data.mime_type = "image/jpeg"
        
        mock_content = MagicMock()
        mock_content.parts = [mock_part]
        
        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        
        mock_client.models.generate_content.return_value = mock_response
        
        # Generate panel
        panel = generator.generate_panel(
            prompt="Test prompt",
            panel_index=1,
        )
        
        assert isinstance(panel, GeneratedPanel)
        assert panel.panel_index == 1
        assert panel.image_data == b"test_image_data"
        assert panel.mime_type == "image/jpeg"
        assert panel.prompt == "Test prompt"
    
    @patch('src.gemini_client.genai.Client')
    def test_generate_panel_with_conditioning(self, mock_client_class, config):
        """Test panel generation with previous image conditioning."""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        generator = GeminiImageGenerator(config)
        
        # Create previous panel
        previous = GeneratedPanel(
            panel_index=1,
            image_data=b"previous_image",
            mime_type="image/jpeg",
            prompt="Previous prompt",
        )
        
        # Setup mock response
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.data = b"new_image_data"
        mock_part.inline_data.mime_type = "image/jpeg"
        
        mock_content = MagicMock()
        mock_content.parts = [mock_part]
        
        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        
        mock_client.models.generate_content.return_value = mock_response
        
        # Generate panel with conditioning
        panel = generator.generate_panel(
            prompt="Next prompt",
            panel_index=2,
            previous_image=previous,
        )
        
        assert panel.panel_index == 2
        assert panel.metadata["conditioned"] is True
        
        # Verify the call included previous image
        call_args = mock_client.models.generate_content.call_args
        contents = call_args.kwargs['contents'][0]
        # Should have 3 parts: previous image, continuity text, new prompt
        assert len(contents.parts) == 3
    
    @patch('src.gemini_client.genai.Client')
    @patch('src.gemini_client.time.sleep')
    def test_generate_panel_with_retry(self, mock_sleep, mock_client_class, config):
        """Test panel generation with retry on failure."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        generator = GeminiImageGenerator(config)
        
        # First call fails, second succeeds
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.data = b"test_image_data"
        mock_part.inline_data.mime_type = "image/jpeg"
        
        mock_content = MagicMock()
        mock_content.parts = [mock_part]
        
        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        
        mock_client.models.generate_content.side_effect = [
            Exception("Temporary failure"),
            mock_response,
        ]
        
        # Generate with retry
        panel = generator.generate_panel(
            prompt="Test prompt",
            panel_index=1,
            max_retries=2,
        )
        
        assert panel.image_data == b"test_image_data"
        assert panel.metadata["attempt"] == 2
        mock_sleep.assert_called_once_with(1)  # Exponential backoff
    
    @patch('src.gemini_client.genai.Client')
    def test_generate_panel_max_retries_exceeded(self, mock_client_class, config):
        """Test panel generation failing after max retries."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        generator = GeminiImageGenerator(config)
        
        # All attempts fail
        mock_client.models.generate_content.side_effect = Exception("Persistent failure")
        
        with pytest.raises(GeminiError, match="Failed to generate panel 1 after 2 attempts"):
            generator.generate_panel(
                prompt="Test prompt",
                panel_index=1,
                max_retries=2,
            )
    
    @patch('src.gemini_client.genai.Client')
    def test_test_connection_success(self, mock_client_class, config):
        """Test successful connection test."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        generator = GeminiImageGenerator(config)
        
        mock_response = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        
        result = generator.test_connection()
        
        assert result is True
    
    @patch('src.gemini_client.genai.Client')
    def test_test_connection_failure(self, mock_client_class, config):
        """Test failed connection test."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        generator = GeminiImageGenerator(config)
        
        mock_client.models.generate_content.side_effect = Exception("API error")
        
        result = generator.test_connection()
        
        assert result is False


class TestCreateCombinedStripImage:
    """Test combined strip image creation."""
    
    @pytest.fixture
    def sample_panels(self):
        """Create sample panels with different colored images."""
        panels = []
        colors = ['red', 'green', 'blue']
        
        for i, color in enumerate(colors):
            img = Image.new('RGB', (200, 300), color=color)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            
            panel = GeneratedPanel(
                panel_index=i + 1,
                image_data=buffer.getvalue(),
                mime_type="image/jpeg",
                prompt=f"Panel {i + 1}",
            )
            panels.append(panel)
        
        return panels
    
    def test_create_combined_strip(self, sample_panels, tmp_path):
        """Test creating combined strip image."""
        output_path = tmp_path / "combined_strip.jpg"
        
        saved_path = create_combined_strip_image(
            panels=sample_panels,
            output_path=output_path,
            panel_width=150,
            panel_height=200,
        )
        
        assert saved_path == output_path
        assert saved_path.exists()
        
        # Check combined image dimensions
        combined_img = Image.open(saved_path)
        assert combined_img.width == 450  # 3 panels * 150
        assert combined_img.height == 200
    
    def test_create_combined_strip_empty_panels(self, tmp_path):
        """Test error handling with empty panel list."""
        output_path = tmp_path / "combined.jpg"
        
        with pytest.raises(ValueError, match="No panels to combine"):
            create_combined_strip_image(
                panels=[],
                output_path=output_path,
            )


@pytest.mark.parametrize("num_panels", [1, 3, 6])
def test_comic_strip_with_various_panel_counts(num_panels):
    """Test comic strip with various panel counts."""
    panels = []
    panel_data = []
    
    for i in range(num_panels):
        # Create panel
        img = Image.new('RGB', (100, 100), color='white')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        
        panel = GeneratedPanel(
            panel_index=i + 1,
            image_data=buffer.getvalue(),
            mime_type="image/jpeg",
            prompt=f"Panel {i + 1}",
        )
        panels.append(panel)
        
        # Create panel data
        panel_data.append(
            PanelData(i + 1, f"action{i}", f"emotion{i}", 
                     f"camera{i}", f"setting{i}", i % 2)
        )
    
    narrative = ComicNarrative(
        style_palette="test",
        panels=panel_data,
        character_bio="test",
        base_style="test",
        bitstring="1" * num_panels,
    )
    
    comic = ComicStrip(panels=panels, narrative=narrative)
    
    assert len(comic.panels) == num_panels
    assert len(comic.narrative.panels) == num_panels