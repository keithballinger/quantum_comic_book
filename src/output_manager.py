"""
Output management for quantum comic strips.

This module handles file organization, HTML generation, and metadata storage
for generated comic strips.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from PIL import Image

from src.config import Config
from src.gemini_client import ComicStrip, GeneratedPanel
from src.prompts import ComicNarrative

logger = logging.getLogger(__name__)


class OutputManager:
    """Manages output files and directories for comic strips."""

    def __init__(self, config: Config):
        """
        Initialize output manager.

        Args:
            config: Application configuration
        """
        self.config = config
        self.output_dir = config.output_dir
        self._ensure_output_dir()

    def _ensure_output_dir(self):
        """Ensure output directory exists."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory ready: {self.output_dir}")

    def create_comic_directory(self, comic_id: Optional[str] = None) -> Path:
        """
        Create a new directory for a comic strip.

        Args:
            comic_id: Optional ID for the comic, defaults to timestamp

        Returns:
            Path to created directory
        """
        if comic_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            comic_id = f"comic_{timestamp}"

        comic_dir = self.output_dir / comic_id
        comic_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (comic_dir / "panels").mkdir(exist_ok=True)
        (comic_dir / "metadata").mkdir(exist_ok=True)

        logger.info(f"Created comic directory: {comic_dir}")
        return comic_dir

    def save_comic_strip(
        self,
        comic: ComicStrip,
        comic_dir: Path,
        save_combined: bool = True,
    ) -> Dict[str, Path]:
        """
        Save a complete comic strip to directory.

        Args:
            comic: Comic strip to save
            comic_dir: Directory to save to
            save_combined: Whether to create combined strip image

        Returns:
            Dictionary of saved file paths
        """
        saved_files = {}

        # Save individual panels
        panels_dir = comic_dir / "panels"
        for panel in comic.panels:
            panel_path = panels_dir / f"panel_{panel.panel_index:02d}"
            saved_path = panel.save(panel_path)
            saved_files[f"panel_{panel.panel_index}"] = saved_path

        # Save title card if present
        if comic.title_card:
            title_path = panels_dir / "title_card"
            saved_files["title_card"] = comic.title_card.save(title_path)

        # Save combined strip
        if save_combined and comic.panels:
            combined_path = self._create_combined_strip(
                comic.panels, comic_dir / "combined_strip.jpg"
            )
            saved_files["combined_strip"] = combined_path

        # Save metadata
        metadata_path = self._save_metadata(comic, comic_dir)
        saved_files["metadata"] = metadata_path

        # Generate HTML viewer
        html_path = self._generate_html(comic, comic_dir)
        saved_files["html"] = html_path

        logger.info(f"Saved comic strip with {len(saved_files)} files")
        return saved_files

    def _create_combined_strip(
        self,
        panels: List[GeneratedPanel],
        output_path: Path,
        panel_width: int = 400,
        panel_height: int = 600,
    ) -> Path:
        """
        Create a single image combining all panels.

        Args:
            panels: List of panels to combine
            output_path: Path to save combined image
            panel_width: Width of each panel
            panel_height: Height of each panel

        Returns:
            Path to saved combined image
        """
        num_panels = len(panels)
        combined_width = panel_width * num_panels
        combined = Image.new("RGB", (combined_width, panel_height), "white")

        for i, panel in enumerate(panels):
            img = panel.to_pil_image()

            # Resize to fit
            img.thumbnail((panel_width, panel_height), Image.Resampling.LANCZOS)

            # Calculate position (center if smaller)
            x_offset = i * panel_width + (panel_width - img.width) // 2
            y_offset = (panel_height - img.height) // 2

            combined.paste(img, (x_offset, y_offset))

        combined.save(output_path, quality=95)
        logger.info(f"Created combined strip: {output_path}")
        return output_path

    def _save_metadata(self, comic: ComicStrip, comic_dir: Path) -> Path:
        """
        Save comic metadata to JSON and text files.

        Args:
            comic: Comic strip with metadata
            comic_dir: Directory to save to

        Returns:
            Path to metadata JSON file
        """
        metadata_dir = comic_dir / "metadata"

        # Save as JSON
        json_path = metadata_dir / "comic_data.json"
        metadata = {
            "bitstring": comic.narrative.bitstring,
            "style_palette": comic.narrative.style_palette,
            "character_bio": comic.narrative.character_bio,
            "base_style": comic.narrative.base_style,
            "panels": [p.to_dict() for p in comic.narrative.panels],
            "generation_time": datetime.now().isoformat(),
            "num_panels": len(comic.panels),
        }

        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save human-readable text
        text_path = metadata_dir / "comic_info.txt"
        with open(text_path, "w") as f:
            f.write("Quantum Comic Strip Information\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Bitstring: {comic.narrative.bitstring}\n")
            f.write(f"Style: {comic.narrative.style_palette}\n")
            f.write(f"Panels: {len(comic.panels)}\n\n")

            f.write("Panel Breakdown:\n")
            f.write("-" * 20 + "\n")
            for panel_data in comic.narrative.panels:
                f.write(f"\nPanel {panel_data.panel_index}:\n")
                f.write(f"  Setting: {panel_data.setting}\n")
                f.write(f"  Action: {panel_data.action}\n")
                f.write(f"  Emotion: {panel_data.emotion}\n")
                f.write(f"  Camera: {panel_data.camera}\n")
                f.write(f"  Character Focus: {panel_data.character_focus + 1}\n")

        logger.info(f"Saved metadata to {json_path}")
        return json_path

    def _generate_html(self, comic: ComicStrip, comic_dir: Path) -> Path:
        """
        Generate HTML viewer for comic strip.

        Args:
            comic: Comic strip to display
            comic_dir: Directory containing comic files

        Returns:
            Path to generated HTML file
        """
        html_path = comic_dir / "index.html"

        # Build panel image paths
        panel_images = []
        if comic.title_card and comic.title_card.file_path:
            panel_images.append(f"panels/{comic.title_card.file_path.name}")

        for panel in comic.panels:
            if panel.file_path:
                panel_images.append(f"panels/{panel.file_path.name}")

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Comic Strip - {comic.narrative.bitstring}</title>
    <style>
        body {{
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            color: white;
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        .subtitle {{
            color: rgba(255,255,255,0.9);
            text-align: center;
            font-size: 1.2em;
            margin-bottom: 30px;
        }}
        .comic-strip {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 20px;
            margin-bottom: 40px;
        }}
        .panel {{
            background: white;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        }}
        .panel:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 50px rgba(0,0,0,0.3);
        }}
        .panel img {{
            max-width: 400px;
            max-height: 600px;
            width: auto;
            height: auto;
            display: block;
            border-radius: 4px;
        }}
        .panel-number {{
            text-align: center;
            margin-top: 10px;
            font-weight: bold;
            color: #667eea;
        }}
        .metadata {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-top: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        .metadata h2 {{
            color: #667eea;
            margin-bottom: 15px;
        }}
        .metadata-item {{
            margin: 10px 0;
            padding: 10px;
            background: #f5f5f5;
            border-radius: 4px;
        }}
        .metadata-label {{
            font-weight: bold;
            color: #764ba2;
        }}
        .panel-details {{
            margin-top: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }}
        .panel-info {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 4px;
            border-left: 4px solid #667eea;
        }}
        .panel-info h3 {{
            margin-top: 0;
            color: #667eea;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌌 Quantum Comic Strip 🎨</h1>
        <p class="subtitle">Generated from quantum measurement: {comic.narrative.bitstring}</p>
        
        <div class="comic-strip">
"""

        # Add panels
        for i, img_path in enumerate(panel_images):
            panel_title = (
                "Title Card"
                if i == 0 and comic.title_card
                else f"Panel {i if not comic.title_card else i}"
            )
            html_content += f"""
            <div class="panel">
                <img src="{img_path}" alt="{panel_title}">
                <div class="panel-number">{panel_title}</div>
            </div>
"""

        # Add metadata section
        html_content += f"""
        </div>
        
        <div class="metadata">
            <h2>Comic Information</h2>
            
            <div class="metadata-item">
                <span class="metadata-label">Quantum Bitstring:</span> {comic.narrative.bitstring}
            </div>
            
            <div class="metadata-item">
                <span class="metadata-label">Style Palette:</span> {comic.narrative.style_palette}
            </div>
            
            <div class="metadata-item">
                <span class="metadata-label">Number of Panels:</span> {len(comic.panels)}
            </div>
            
            <div class="metadata-item">
                <span class="metadata-label">Characters:</span> {comic.narrative.character_bio}
            </div>
"""
        
        # Add quantum constraints if available
        if hasattr(comic.narrative, 'quantum_constraints') and comic.narrative.quantum_constraints:
            constraints = comic.narrative.quantum_constraints
            html_content += f"""
            <h2>Quantum Signature</h2>
            
            <div class="metadata-item">
                <span class="metadata-label">Backend:</span> {constraints.get('backend', 'unknown')}
            </div>
            
            <div class="metadata-item">
                <span class="metadata-label">Tone:</span> {constraints.get('tone', 'contemporary')}
            </div>
            
            <div class="metadata-item">
                <span class="metadata-label">Style Parity (σS):</span> {constraints.get('style_parity', 0)}
            </div>
            
            <div class="metadata-item">
                <span class="metadata-label">Rhetorical Bias:</span> {constraints.get('rhetorical_bias', 'anaphora')}
            </div>
            
            <div class="metadata-item">
                <span class="metadata-label">Recurring Phrase:</span> "{constraints.get('recurring_phrase', '')}"
            </div>
"""
        
        html_content += """
            <h2>Panel Details</h2>
            <div class="panel-details">
"""

        # Add panel details
        for panel_data in comic.narrative.panels:
            html_content += f"""
                <div class="panel-info">
                    <h3>Panel {panel_data.panel_index}</h3>
                    <p><strong>Setting:</strong> {panel_data.setting}</p>
                    <p><strong>Action:</strong> {panel_data.action}</p>
                    <p><strong>Emotion:</strong> {panel_data.emotion}</p>
                    <p><strong>Camera:</strong> {panel_data.camera}</p>
                    <p><strong>Character Focus:</strong> Character {panel_data.character_focus + 1}</p>
                </div>
"""

        html_content += """
            </div>
        </div>
    </div>
</body>
</html>
"""

        with open(html_path, "w") as f:
            f.write(html_content)

        logger.info(f"Generated HTML viewer: {html_path}")
        return html_path

    def cleanup_old_comics(self, keep_latest: int = 10):
        """
        Clean up old comic directories, keeping only the latest ones.

        Args:
            keep_latest: Number of latest comics to keep
        """
        comic_dirs = [
            d
            for d in self.output_dir.iterdir()
            if d.is_dir() and d.name.startswith("comic_")
        ]

        if len(comic_dirs) <= keep_latest:
            return

        # Sort by modification time
        comic_dirs.sort(key=lambda d: d.stat().st_mtime)

        # Remove oldest directories
        to_remove = comic_dirs[:-keep_latest]
        for dir_path in to_remove:
            shutil.rmtree(dir_path)
            logger.info(f"Removed old comic directory: {dir_path}")

    def export_comic_archive(
        self, comic_dir: Path, archive_path: Optional[Path] = None
    ) -> Path:
        """
        Create a ZIP archive of a comic directory.

        Args:
            comic_dir: Directory to archive
            archive_path: Optional path for archive

        Returns:
            Path to created archive
        """
        if archive_path is None:
            archive_path = self.output_dir / f"{comic_dir.name}.zip"

        # Remove .zip extension as shutil adds it
        base_path = str(archive_path).replace(".zip", "")

        shutil.make_archive(base_path, "zip", comic_dir)

        logger.info(f"Created archive: {archive_path}")
        return Path(f"{base_path}.zip")
