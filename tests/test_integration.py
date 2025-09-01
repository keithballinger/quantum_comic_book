"""
Integration tests for the quantum comic book generator.

These tests verify the full workflow from quantum circuit to comic output.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from PIL import Image

from src.config import Config, load_config
from src.quantum_circuit import create_quantum_circuit, CircuitRegisters
from src.prompts import decode_quantum_result
from src.main import QuantumComicGenerator


class TestFullWorkflow:
    """Test complete workflow integration."""
    
    @pytest.fixture
    def test_config(self, tmp_path):
        """Create test configuration."""
        config = Config(
            ibm_api_key="test_ibm",
            gemini_api_key="test_gemini",
            panels=3,
            characters=2,
            output_dir=tmp_path / "test_output",
            use_simulator=True,
        )
        # Add missing attributes
        config.character_style = "default"
        config.art_style = "comic"
        config.random_seed = None
        config.generate_title = True
        config.create_archive = False
        config.cleanup_old = False
        config.keep_latest_comics = 10
        return config
    
    def test_circuit_to_narrative_flow(self, test_config):
        """Test flow from circuit creation to narrative generation."""
        # Create circuit
        test_config.seed = 42
        circuit, registers = create_quantum_circuit(test_config)
        
        assert circuit is not None
        assert circuit.num_qubits == 25  # 3 + 6 + 6 + 6 + 4
        
        # Simulate measurement result
        test_bitstring = "101" + "110011" + "001100" + "101010" + "1101"
        
        # Create registers
        registers = CircuitRegisters(
            time_qubits=3,
            action_qubits=6,
            emotion_qubits=6,
            camera_qubits=6,
            style_qubits=4,
            total_qubits=25,
        )
        
        # Decode into narrative
        narrative, prompts = decode_quantum_result(
            test_bitstring,
            registers,
            character_style="default",
            art_style="comic",
        )
        
        assert narrative is not None
        assert len(narrative.panels) == 3
        assert len(prompts) == 3
        assert narrative.bitstring == test_bitstring
        
        # Verify narrative structure
        for i, panel in enumerate(narrative.panels):
            assert panel.panel_index == i + 1
            assert panel.action is not None
            assert panel.emotion is not None
            assert panel.camera is not None
            assert panel.setting is not None
    
    @patch('src.main.GeminiImageGenerator')
    @patch('src.main.execute_quantum_circuit')
    @patch('src.main.create_quantum_circuit')
    def test_end_to_end_generation(
        self, mock_create_circuit, mock_execute, mock_gemini_class, test_config
    ):
        """Test complete generation from start to finish."""
        # Setup mocks
        mock_circuit = MagicMock()
        mock_create_circuit.return_value = mock_circuit
        
        mock_result = MagicMock()
        mock_result.bitstring = "101110011001100101010110"
        mock_result.backend_name = "simulator"
        mock_execute.return_value = mock_result
        
        # Mock Gemini
        mock_gemini = MagicMock()
        mock_gemini.test_connection.return_value = True
        
        # Create mock panels with proper image data
        mock_panels = []
        for i in range(3):
            # Create a real image
            img = Image.new('RGB', (100, 100), color='red')
            import io
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            
            mock_panel = MagicMock()
            mock_panel.panel_index = i + 1
            mock_panel.image_data = buffer.getvalue()
            mock_panel.mime_type = "image/jpeg"
            mock_panel.file_path = None
            # Add mock methods
            mock_panel.save = MagicMock(return_value=test_config.output_dir / f"panel_{i+1}.jpg")
            mock_panel.to_pil_image = MagicMock(return_value=img)
            mock_panels.append(mock_panel)
        
        # Create proper narrative mock
        from src.prompts import ComicNarrative, PanelData
        mock_panel_data = [
            PanelData(i+1, f"action{i}", f"emotion{i}", f"camera{i}", f"setting{i}", i % 2)
            for i in range(3)
        ]
        mock_narrative = ComicNarrative(
            style_palette="test palette",
            panels=mock_panel_data,
            character_bio="Test characters",
            base_style="Test style",
            bitstring="101110011001100101010110",
        )
        
        mock_comic = MagicMock()
        mock_comic.panels = mock_panels
        mock_comic.narrative = mock_narrative
        mock_comic.title_card = None
        
        mock_gemini.generate_comic_strip.return_value = mock_comic
        mock_gemini_class.return_value = mock_gemini
        
        # Generate comic
        generator = QuantumComicGenerator(test_config)
        comic_dir, bitstring = generator.generate_comic()
        
        # Verify workflow
        assert comic_dir.exists()
        assert bitstring == "101110011001100101010110"
        
        # Check that all components were called
        mock_create_circuit.assert_called_once()
        mock_execute.assert_called_once()
        mock_gemini.generate_comic_strip.assert_called_once()
    
    def test_config_validation_integration(self, tmp_path):
        """Test configuration validation in integration."""
        # Test with invalid panels (panels are validated in post_init)
        config = Config(
            ibm_api_key="test",
            gemini_api_key="test",
            panels=50,  # Too many panels
        )
        
        from src.config import validate_config, ConfigError
        # This should raise an error due to too many qubits
        with pytest.raises(ConfigError, match="maximum is 127"):
            validate_config(config)
    
    def test_bitstring_length_consistency(self):
        """Test that bitstring length matches circuit requirements."""
        for panels in [1, 3, 6, 12]:
            # Calculate expected length
            expected_length = panels + (2 * panels * 3) + 4
            
            # Create registers
            registers = CircuitRegisters(
                time_qubits=panels,
                action_qubits=2 * panels,
                emotion_qubits=2 * panels,
                camera_qubits=2 * panels,
                style_qubits=4,
                total_qubits=expected_length,
            )
            
            # Create test bitstring
            test_bitstring = "1" * expected_length
            
            # Decode should work without errors
            narrative, prompts = decode_quantum_result(
                test_bitstring,
                registers,
            )
            
            assert len(narrative.panels) == panels
            assert len(prompts) == panels


class TestErrorHandling:
    """Test error handling across modules."""
    
    @pytest.fixture
    def test_config(self, tmp_path):
        """Create test configuration."""
        config = Config(
            ibm_api_key="test_ibm",
            gemini_api_key="test_gemini",
            panels=3,
            characters=2,
            output_dir=tmp_path / "test_output",
        )
        # Add missing attributes
        config.character_style = "default"
        config.art_style = "comic"
        config.random_seed = None
        config.generate_title = True
        config.create_archive = False
        config.cleanup_old = False
        config.keep_latest_comics = 10
        return config
    
    @patch('src.main.GeminiImageGenerator')
    def test_gemini_connection_failure_handling(self, mock_gemini_class, test_config):
        """Test handling of Gemini connection failure."""
        mock_gemini = MagicMock()
        mock_gemini.test_connection.return_value = False
        mock_gemini_class.return_value = mock_gemini
        
        generator = QuantumComicGenerator(test_config)
        
        with pytest.raises(RuntimeError, match="Failed to connect to Gemini"):
            generator.generate_comic(
                skip_quantum=True,
                test_bitstring="101010",
            )
    
    @patch('src.main.execute_quantum_circuit')
    @patch('src.main.create_quantum_circuit')
    def test_quantum_execution_failure_handling(
        self, mock_create_circuit, mock_execute, test_config
    ):
        """Test handling of quantum execution failure."""
        mock_circuit = MagicMock()
        mock_create_circuit.return_value = mock_circuit
        
        # Simulate execution failure
        mock_execute.side_effect = Exception("Quantum execution failed")
        
        generator = QuantumComicGenerator(test_config)
        
        with pytest.raises(Exception, match="Quantum execution failed"):
            generator.generate_comic()
    
    def test_invalid_bitstring_handling(self):
        """Test handling of invalid bitstrings."""
        registers = CircuitRegisters(
            time_qubits=3,
            action_qubits=6,
            emotion_qubits=6,
            camera_qubits=6,
            style_qubits=4,
            total_qubits=25,
        )
        
        # Test with wrong length bitstring
        short_bitstring = "101"  # Too short
        
        # Should handle gracefully
        narrative, prompts = decode_quantum_result(
            short_bitstring,
            registers,
        )
        
        # Should still produce output (with defaults)
        assert narrative is not None
        assert len(narrative.panels) == 3


class TestOutputGeneration:
    """Test output file generation integration."""
    
    @pytest.fixture
    def mock_comic(self):
        """Create mock comic for testing."""
        # Create mock panels with real image data
        panels = []
        for i in range(3):
            img = Image.new('RGB', (200, 300), color=['red', 'green', 'blue'][i])
            import io
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            
            from src.gemini_client import GeneratedPanel
            panel = GeneratedPanel(
                panel_index=i + 1,
                image_data=buffer.getvalue(),
                mime_type="image/jpeg",
                prompt=f"Test panel {i + 1}",
            )
            panels.append(panel)
        
        # Create narrative
        from src.prompts import ComicNarrative, PanelData
        panel_data = [
            PanelData(i + 1, f"action{i}", f"emotion{i}", 
                     f"camera{i}", f"setting{i}", i % 2)
            for i in range(3)
        ]
        
        narrative = ComicNarrative(
            style_palette="test palette",
            panels=panel_data,
            character_bio="Test characters",
            base_style="Test style",
            bitstring="101010101010",
        )
        
        from src.gemini_client import ComicStrip
        return ComicStrip(panels=panels, narrative=narrative)
    
    def test_output_directory_structure(self, mock_comic, tmp_path):
        """Test that output directory structure is created correctly."""
        from src.output_manager import OutputManager
        from src.config import Config
        
        config = Config(
            ibm_api_key="test",
            gemini_api_key="test",
            output_dir=tmp_path,
        )
        
        manager = OutputManager(config)
        comic_dir = manager.create_comic_directory("test_comic")
        
        # Save comic
        saved_files = manager.save_comic_strip(mock_comic, comic_dir)
        
        # Check directory structure
        assert comic_dir.exists()
        assert (comic_dir / "panels").exists()
        assert (comic_dir / "metadata").exists()
        
        # Check files
        assert (comic_dir / "index.html").exists()
        assert (comic_dir / "combined_strip.jpg").exists()
        assert (comic_dir / "metadata" / "comic_data.json").exists()
        assert (comic_dir / "metadata" / "comic_info.txt").exists()
        
        # Check panel files
        for i in range(3):
            panel_file = comic_dir / "panels" / f"panel_{i+1:02d}.jpg"
            assert panel_file.exists()
    
    def test_metadata_integrity(self, mock_comic, tmp_path):
        """Test that metadata is saved correctly."""
        from src.output_manager import OutputManager
        from src.config import Config
        
        config = Config(
            ibm_api_key="test",
            gemini_api_key="test",
            output_dir=tmp_path,
        )
        
        manager = OutputManager(config)
        comic_dir = manager.create_comic_directory("test_metadata")
        manager.save_comic_strip(mock_comic, comic_dir)
        
        # Load and verify metadata
        metadata_file = comic_dir / "metadata" / "comic_data.json"
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        assert metadata["bitstring"] == "101010101010"
        assert metadata["style_palette"] == "test palette"
        assert metadata["num_panels"] == 3
        assert len(metadata["panels"]) == 3
        
        # Verify panel data
        for i, panel in enumerate(metadata["panels"]):
            assert panel["panel_index"] == i + 1
            assert panel["action"] == f"action{i}"
            assert panel["emotion"] == f"emotion{i}"


class TestCLIIntegration:
    """Test CLI argument handling integration."""
    
    @patch('src.main.QuantumComicGenerator')
    @patch('src.main.validate_config')
    @patch('src.main.load_config')
    @patch('src.main.setup_logging')
    @patch('sys.argv', ['main.py', '--panels', '6', '--simulator', '--verbose'])
    def test_cli_argument_propagation(
        self, mock_logging, mock_load, mock_validate, mock_generator_class
    ):
        """Test that CLI arguments propagate correctly."""
        from src.main import main
        
        mock_config = MagicMock()
        mock_load.return_value = mock_config
        
        mock_generator = MagicMock()
        mock_generator.generate_comic.return_value = (Path("/test"), "101010")
        mock_generator_class.return_value = mock_generator
        
        result = main()
        
        assert result == 0
        
        # Verify config was modified
        assert mock_config.panels == 6
        assert mock_config.use_simulator is True
        
        # Verify verbose logging was enabled
        mock_logging.assert_called_once_with(True)
    
    @patch('src.main.list_available_backends')
    @patch('src.main.validate_config')
    @patch('src.main.load_config')
    @patch('src.main.setup_logging')
    @patch('sys.argv', ['main.py', '--list-backends'])
    def test_cli_list_backends_integration(
        self, mock_logging, mock_load, mock_validate, mock_list_backends
    ):
        """Test backend listing through CLI."""
        from src.main import main
        
        mock_config = MagicMock()
        mock_load.return_value = mock_config
        
        mock_list_backends.return_value = [
            {"name": "ibm_kyoto", "operational": True, "num_qubits": 127},
        ]
        
        result = main()
        
        assert result == 0
        mock_list_backends.assert_called_once_with(mock_config)


@pytest.mark.parametrize("panels,characters,expected_qubits", [
    (1, 1, 11),
    (3, 2, 25),
    (6, 2, 46),
    (12, 4, 88),
])
def test_qubit_calculation_consistency(panels, characters, expected_qubits):
    """Test that qubit calculations are consistent across modules."""
    from src.config import get_circuit_parameters
    
    config = Config(
        ibm_api_key="test",
        gemini_api_key="test",
        panels=panels,
        characters=characters,
    )
    
    params = get_circuit_parameters(config)
    assert params["total_qubits"] == expected_qubits
    
    # Create circuit and verify
    circuit, registers = create_quantum_circuit(config)
    assert circuit.num_qubits == expected_qubits