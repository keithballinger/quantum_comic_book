"""
Tests for output management module.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import pytest
from PIL import Image

from src.config import Config
from src.gemini_client import GeneratedPanel, ComicStrip
from src.prompts import ComicNarrative, PanelData
from src.output_manager import OutputManager


class TestOutputManager:
    """Test OutputManager functionality."""

    @pytest.fixture
    def config(self, tmp_path):
        """Create test configuration."""
        return Config(
            ibm_api_key="test_ibm",
            gemini_api_key="test_gemini",
            output_dir=tmp_path / "output",
        )

    @pytest.fixture
    def manager(self, config):
        """Create output manager."""
        return OutputManager(config)

    @pytest.fixture
    def sample_panel(self):
        """Create sample panel."""
        img = Image.new("RGB", (200, 300), color="blue")
        import io

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")

        return GeneratedPanel(
            panel_index=1,
            image_data=buffer.getvalue(),
            mime_type="image/jpeg",
            prompt="Test panel",
        )

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
    def sample_comic(self, sample_narrative):
        """Create sample comic strip."""
        panels = []
        for i in range(3):
            img = Image.new("RGB", (200, 300), color=["red", "green", "blue"][i])
            import io

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")

            panel = GeneratedPanel(
                panel_index=i + 1,
                image_data=buffer.getvalue(),
                mime_type="image/jpeg",
                prompt=f"Panel {i + 1}",
            )
            panels.append(panel)

        return ComicStrip(
            panels=panels,
            narrative=sample_narrative,
        )

    def test_manager_initialization(self, manager, config):
        """Test output manager initialization."""
        assert manager.config == config
        assert manager.output_dir == config.output_dir
        assert manager.output_dir.exists()

    def test_create_comic_directory_with_id(self, manager):
        """Test creating comic directory with specific ID."""
        comic_dir = manager.create_comic_directory("test_comic")

        assert comic_dir.exists()
        assert comic_dir.name == "test_comic"
        assert (comic_dir / "panels").exists()
        assert (comic_dir / "metadata").exists()

    def test_create_comic_directory_auto_id(self, manager):
        """Test creating comic directory with auto-generated ID."""
        with patch("src.output_manager.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"
            comic_dir = manager.create_comic_directory()

        assert comic_dir.exists()
        assert comic_dir.name == "comic_20240101_120000"

    def test_save_comic_strip(self, manager, sample_comic):
        """Test saving complete comic strip."""
        comic_dir = manager.create_comic_directory("test_save")

        saved_files = manager.save_comic_strip(sample_comic, comic_dir)

        # Check saved files
        assert "panel_1" in saved_files
        assert "panel_2" in saved_files
        assert "panel_3" in saved_files
        assert "combined_strip" in saved_files
        assert "metadata" in saved_files
        assert "html" in saved_files

        # Verify files exist
        for path in saved_files.values():
            assert path.exists()

        # Check panel files
        panels_dir = comic_dir / "panels"
        assert len(list(panels_dir.glob("panel_*.jpg"))) == 3

    def test_save_comic_with_title_card(self, manager, sample_comic, sample_panel):
        """Test saving comic with title card."""
        sample_comic.title_card = sample_panel
        comic_dir = manager.create_comic_directory("test_title")

        saved_files = manager.save_comic_strip(sample_comic, comic_dir)

        assert "title_card" in saved_files
        assert saved_files["title_card"].exists()

    def test_save_comic_without_combined(self, manager, sample_comic):
        """Test saving comic without combined strip."""
        comic_dir = manager.create_comic_directory("test_no_combined")

        saved_files = manager.save_comic_strip(
            sample_comic, comic_dir, save_combined=False
        )

        assert "combined_strip" not in saved_files

    def test_create_combined_strip(self, manager, sample_comic, tmp_path):
        """Test creating combined strip image."""
        output_path = tmp_path / "combined.jpg"

        result_path = manager._create_combined_strip(
            sample_comic.panels,
            output_path,
            panel_width=150,
            panel_height=200,
        )

        assert result_path == output_path
        assert result_path.exists()

        # Check combined image dimensions
        img = Image.open(result_path)
        assert img.width == 450  # 3 panels * 150
        assert img.height == 200

    def test_save_metadata(self, manager, sample_comic, tmp_path):
        """Test saving metadata files."""
        comic_dir = tmp_path / "test_metadata"
        comic_dir.mkdir()
        (comic_dir / "metadata").mkdir()

        metadata_path = manager._save_metadata(sample_comic, comic_dir)

        assert metadata_path.exists()

        # Check JSON content
        with open(metadata_path) as f:
            data = json.load(f)

        assert data["bitstring"] == "101010"
        assert data["style_palette"] == "test palette"
        assert len(data["panels"]) == 3
        assert "generation_time" in data

        # Check text file
        text_path = comic_dir / "metadata" / "comic_info.txt"
        assert text_path.exists()

        content = text_path.read_text()
        assert "101010" in content
        assert "test palette" in content
        assert "Panel 1:" in content

    def test_generate_html(self, manager, sample_comic, tmp_path):
        """Test HTML generation."""
        comic_dir = tmp_path / "test_html"
        comic_dir.mkdir()
        (comic_dir / "panels").mkdir()

        # Save panels first to set file paths
        for panel in sample_comic.panels:
            panel_path = comic_dir / "panels" / f"panel_{panel.panel_index:02d}.jpg"
            panel_path.write_bytes(panel.image_data)
            panel.file_path = panel_path

        html_path = manager._generate_html(sample_comic, comic_dir)

        assert html_path.exists()
        assert html_path.name == "index.html"

        content = html_path.read_text()
        assert "Quantum Comic Strip" in content
        assert "101010" in content
        assert "test palette" in content
        assert "Panel 1" in content
        assert "panel_01.jpg" in content

    def test_cleanup_old_comics(self, manager):
        """Test cleaning up old comic directories."""
        # Create multiple comic directories
        for i in range(15):
            comic_dir = manager.output_dir / f"comic_{i:03d}"
            comic_dir.mkdir(parents=True)

            # Create a dummy file with different timestamps
            (comic_dir / "test.txt").write_text(f"Comic {i}")

        # Keep only 10 latest
        manager.cleanup_old_comics(keep_latest=10)

        remaining_dirs = list(manager.output_dir.glob("comic_*"))
        assert len(remaining_dirs) == 10

    def test_cleanup_with_fewer_comics(self, manager):
        """Test cleanup when fewer comics than limit."""
        # Create only 5 comics
        for i in range(5):
            comic_dir = manager.output_dir / f"comic_{i:03d}"
            comic_dir.mkdir(parents=True)

        manager.cleanup_old_comics(keep_latest=10)

        # All should remain
        remaining_dirs = list(manager.output_dir.glob("comic_*"))
        assert len(remaining_dirs) == 5

    def test_export_comic_archive(self, manager, sample_comic):
        """Test creating ZIP archive of comic."""
        # Create and save comic first
        comic_dir = manager.create_comic_directory("test_archive")
        manager.save_comic_strip(sample_comic, comic_dir)

        # Create archive
        archive_path = manager.export_comic_archive(comic_dir)

        assert archive_path.exists()
        assert archive_path.suffix == ".zip"
        assert "test_archive" in archive_path.name

    def test_export_comic_archive_custom_path(self, manager, sample_comic, tmp_path):
        """Test creating archive with custom path."""
        comic_dir = manager.create_comic_directory("test_custom")
        manager.save_comic_strip(sample_comic, comic_dir)

        custom_path = tmp_path / "custom_archive.zip"
        archive_path = manager.export_comic_archive(comic_dir, custom_path)

        assert archive_path.exists()
        assert archive_path == custom_path


@pytest.mark.parametrize("num_panels", [1, 3, 6])
def test_combined_strip_various_panels(tmp_path, num_panels):
    """Test combined strip with various panel counts."""
    panels = []
    for i in range(num_panels):
        img = Image.new("RGB", (100, 150), color="white")
        import io

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")

        panel = GeneratedPanel(
            panel_index=i + 1,
            image_data=buffer.getvalue(),
            mime_type="image/jpeg",
            prompt=f"Panel {i + 1}",
        )
        panels.append(panel)

    config = Config(
        ibm_api_key="test",
        gemini_api_key="test",
        output_dir=tmp_path,
    )
    manager = OutputManager(config)

    output_path = tmp_path / "combined.jpg"
    result_path = manager._create_combined_strip(panels, output_path)

    assert result_path.exists()

    img = Image.open(result_path)
    assert img.width == 400 * num_panels  # Default width per panel
