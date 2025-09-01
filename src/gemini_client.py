"""
Gemini image generation client for comic panel creation.

This module handles image generation using Google's Gemini API,
including image-to-image conditioning for visual consistency.
"""

import base64
import io
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from PIL import Image
from google import genai
from google.genai import types

from src.config import Config
from src.narrative import ComicNarrative, PanelData





class GeminiError(Exception):
    """Gemini API error."""
    pass


@dataclass
class GeneratedPanel:
    """Container for a generated comic panel."""
    
    panel_index: int
    image_data: bytes
    mime_type: str
    prompt: str
    file_path: Optional[Path] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def save(self, file_path: Path) -> Path:
        """
        Save panel image to file.
        
        Args:
            file_path: Path to save image
            
        Returns:
            Path where image was saved
        """
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine extension from mime type
        ext = mimetypes.guess_extension(self.mime_type)
        if not ext or ext == ".jpe":  # Handle .jpe -> .jpg
            ext = ".jpg"
        if not file_path.suffix:
            file_path = file_path.with_suffix(ext)
        
        # Save image data
        file_path.write_bytes(self.image_data)
        self.file_path = file_path
        
        return file_path
    
    def to_pil_image(self) -> Image.Image:
        """
        Convert to PIL Image.
        
        Returns:
            PIL Image object
        """
        return Image.open(io.BytesIO(self.image_data))


@dataclass
class ComicStrip:
    """Container for a complete comic strip."""
    
    panels: List[GeneratedPanel]
    narrative: ComicNarrative
    title_card: Optional[GeneratedPanel] = None
    
    def save_all(self, output_dir: Path, prefix: str = "panel") -> List[Path]:
        """
        Save all panels to directory.
        
        Args:
            output_dir: Directory to save panels
            prefix: Filename prefix for panels
            
        Returns:
            List of saved file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        
        # Save title card if present
        if self.title_card:
            title_path = output_dir / "title_card"
            saved_paths.append(self.title_card.save(title_path))
        
        # Save all panels
        for panel in self.panels:
            panel_path = output_dir / f"{prefix}_{panel.panel_index:02d}"
            saved_paths.append(panel.save(panel_path))
        
        # Save metadata
        metadata_path = output_dir / "metadata.txt"
        self._save_metadata(metadata_path)
        
        return saved_paths
    
    def _save_metadata(self, file_path: Path):
        """Save comic metadata to file."""
        with open(file_path, 'w') as f:
            f.write(f"Quantum Comic Strip Metadata\n")
            f.write(f"============================\n\n")
            f.write(f"Bitstring: {self.narrative.bitstring}\n")
            f.write(f"Style: {self.narrative.style_palette}\n")
            f.write(f"Panels: {len(self.panels)}\n\n")
            
            for i, panel_data in enumerate(self.narrative.panels):
                f.write(f"Panel {i+1}:\n")
                f.write(f"  Setting: {panel_data.setting}\n")
                f.write(f"  Action: {panel_data.action}\n")
                f.write(f"  Emotion: {panel_data.emotion}\n")
                f.write(f"  Camera: {panel_data.camera}\n")
                f.write(f"  Focus: Character {panel_data.character_focus + 1}\n\n")


class GeminiClient:
    """Client for Gemini API for text and image generation."""
    
    TEXT_MODEL_NAME = "gemini-2.5-flash"
    IMAGE_MODEL_NAME = "gemini-2.5-flash-image-preview"
    
    def __init__(self, config: Config):
        """
        Initialize Gemini image generator.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.client: Optional[genai.Client] = None
        self._connect()
    
    def _connect(self):
        """Connect to Gemini API."""
        try:
            self.client = genai.Client(api_key=self.config.gemini_api_key)
            if self.config.verbose:
                print("Connected to Gemini API")
        except Exception as e:
            raise GeminiError(f"Failed to connect to Gemini: {e}")

    def generate_narrative(self, narrative_prompt: str, max_retries: int = 3) -> str:
        """
        Generate a structured narrative using Gemini text model.

        Args:
            narrative_prompt: The prompt to generate the narrative from.
            max_retries: Maximum number of retries.

        Returns:
            A JSON string representing the comic narrative.
        """
        last_error = None
        for attempt in range(max_retries):
            try:
                if self.config.verbose:
                    print(f"Generating narrative (attempt {attempt + 1})")
                response = self.client.models.generate_content(
                    model=self.TEXT_MODEL_NAME,
                    contents=[types.Content(role="user", parts=[types.Part.from_text(text=narrative_prompt)])],
                )
                if response and response.text:
                    if self.config.verbose:
                        print("Successfully generated narrative.")
                    return response.text
                raise GeminiError("Empty response from narrative generation.")
            except Exception as e:
                last_error = e
                if self.config.verbose:
                    print(f"Narrative generation attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)
        
        raise GeminiError(f"Failed to generate narrative after {max_retries} attempts: {last_error}")
    
    def generate_panel(
        self,
        prompt: str,
        panel_index: int,
        previous_image: Optional[GeneratedPanel] = None,
        max_retries: int = 3,
    ) -> GeneratedPanel:
        """
        Generate a single comic panel.
        
        Args:
            prompt: Text prompt for generation
            panel_index: Panel number
            previous_image: Previous panel for conditioning
            max_retries: Maximum retry attempts
            
        Returns:
            Generated panel
            
        Raises:
            GeminiError: If generation fails
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                if self.config.verbose:
                    print(f"Generating panel {panel_index} (attempt {attempt + 1})")
                
                # Build prompt parts
                parts = []
                
                # Add previous image for conditioning if available
                if previous_image:
                    parts.append(
                        types.Part.from_bytes(
                            mime_type=previous_image.mime_type,
                            data=previous_image.image_data,
                        )
                    )
                    parts.append(
                        types.Part.from_text(
                            text="Create the next comic panel based on the attached previous panel. "
                            "Preserve character identity, outfit, palette, and artistic style. "
                            "Maintain visual consistency while evolving the scene."
                        )
                    )
                
                # Add main prompt
                parts.append(types.Part.from_text(text=prompt))
                
                # Configure generation
                generate_config = types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                    temperature=0.8,  # Some creativity but not too wild
                )
                
                # Generate image using streaming
                image_data = None
                mime_type = "image/jpeg"
                
                for chunk in self.client.models.generate_content_stream(
                    model=self.IMAGE_MODEL_NAME,
                    contents=[types.Content(role="user", parts=parts)],
                    config=generate_config,
                ):
                    if (chunk.candidates is None or 
                        chunk.candidates[0].content is None or 
                        chunk.candidates[0].content.parts is None):
                        continue
                    
                    # Check for image data in the response
                    for part in chunk.candidates[0].content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data and part.inline_data.data:
                            image_data = part.inline_data.data
                            mime_type = part.inline_data.mime_type or "image/jpeg"
                            break
                    
                    if image_data:
                        break
                
                if image_data:
                    panel = GeneratedPanel(
                        panel_index=panel_index,
                        image_data=image_data,
                        mime_type=mime_type,
                        prompt=prompt,
                        metadata={
                            "attempt": attempt + 1,
                            "conditioned": previous_image is not None,
                        }
                    )
                    
                    if self.config.verbose:
                        print(f"Successfully generated panel {panel_index}")
                    return panel
                
                raise GeminiError(f"No image in response for panel {panel_index}")
                
            except Exception as e:
                last_error = e
                if self.config.verbose:
                    print(f"Generation attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    # Wait before retry with exponential backoff
                    wait_time = 2 ** attempt
                    if self.config.verbose:
                        print(f"Waiting {wait_time} seconds before retry")
                    time.sleep(wait_time)
        
        raise GeminiError(
            f"Failed to generate panel {panel_index} after {max_retries} attempts: {last_error}"
        )
    
    def generate_comic_strip(
        self,
        prompts: List[str],
        narrative: ComicNarrative,
        generate_title: bool = True,
    ) -> ComicStrip:
        """
        Generate complete comic strip from prompts.
        
        Args:
            prompts: List of prompts for each panel
            narrative: Comic narrative structure
            generate_title: Whether to generate title card
            
        Returns:
            Complete comic strip
            
        Raises:
            GeminiError: If generation fails
        """
        panels = []
        previous_panel = None
        
        # Generate each panel sequentially
        for i, prompt in enumerate(prompts):
            panel = self.generate_panel(
                prompt=prompt,
                panel_index=i + 1,
                previous_image=previous_panel,
                max_retries=self.config.max_retries,
            )
            panels.append(panel)
            
            # Use this panel as reference for next
            previous_panel = panel
            
            # Small delay to avoid rate limiting
            if i < len(prompts) - 1:
                time.sleep(1)
        
        # Create comic strip
        comic = ComicStrip(
            panels=panels,
            narrative=narrative,
        )
        
        # Generate title card if requested
        if generate_title:
            try:
                from src.prompts import PromptGenerator
                generator = PromptGenerator(narrative)
                title_prompt = generator.generate_title_prompt()
                
                # Use first panel as style reference for title
                title_card = self.generate_panel(
                    prompt=title_prompt,
                    panel_index=0,
                    previous_image=panels[0] if panels else None,
                )
                comic.title_card = title_card
                
            except Exception as e:
                if self.config.verbose:
                    print(f"Failed to generate title card: {e}")
        
        return comic
    
    def test_connection(self) -> bool:
        """
        Test connection to Gemini API.
        
        Returns:
            True if connection successful
        """
        try:
            # Try a simple text generation to test API
            response = self.client.models.generate_content(
                model=self.IMAGE_MODEL_NAME,
                contents=[types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="Hello, testing connection")]
                )],
            )
            return response is not None
            
        except Exception as e:
            if self.config.verbose:
                print(f"Connection test failed: {e}")
            return False


def create_comic_strip_from_bitstring(
    bitstring: str,
    registers: Any,  # CircuitRegisters
    config: Config,
    output_dir: Optional[Path] = None,
) -> Tuple[ComicStrip, Path]:
    """
    Create complete comic strip from quantum measurement.
    
    Args:
        bitstring: Quantum measurement result
        registers: Circuit register information
        config: Application configuration
        output_dir: Optional output directory
        
    Returns:
        Tuple of (comic strip, output directory)
    """
    from src.prompts import decode_quantum_result
    from datetime import datetime
    
    # Decode quantum result into narrative and prompts
    narrative, prompts = decode_quantum_result(bitstring, registers, config)
    
    # Generate images
    generator = GeminiClient(config)
    comic = generator.generate_comic_strip(prompts, narrative)
    
    # Determine output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = config.output_dir / f"comic_{timestamp}"
    
    # Save all panels
    comic.save_all(output_dir)
    
    return comic, output_dir


def create_combined_strip_image(
    panels: List[GeneratedPanel],
    output_path: Path,
    panel_width: int = 400,
    panel_height: int = 600,
) -> Path:
    """
    Create a single image combining all panels.
    
    Args:
        panels: List of generated panels
        output_path: Path to save combined image
        panel_width: Width of each panel in combined image
        panel_height: Height of each panel in combined image
        
    Returns:
        Path to saved combined image
    """
    if not panels:
        raise ValueError("No panels to combine")
    
    # Create combined image
    num_panels = len(panels)
    combined_width = panel_width * num_panels
    combined = Image.new('RGB', (combined_width, panel_height), 'white')
    
    # Paste each panel
    for i, panel in enumerate(panels):
        img = panel.to_pil_image()
        
        # Resize to fit
        img.thumbnail((panel_width, panel_height), Image.Resampling.LANCZOS)
        
        # Calculate position (center if smaller)
        x_offset = i * panel_width + (panel_width - img.width) // 2
        y_offset = (panel_height - img.height) // 2
        
        combined.paste(img, (x_offset, y_offset))
    
    # Save combined image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output_path, quality=95)
    
    return output_path