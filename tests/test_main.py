"""
Tests for main application module.
"""

import argparse
import logging
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call

import pytest

from src.config import Config
from src.main import (
    QuantumComicGenerator,
    setup_logging,
    create_parser,
    main,
)


class TestSetupLogging:
    """Test logging configuration."""

    @patch("src.main.logging.basicConfig")
    @patch("src.main.logging.FileHandler")
    @patch("src.main.logging.StreamHandler")
    def test_setup_logging_default(self, mock_stream, mock_file, mock_config):
        """Test default logging setup."""
        setup_logging(verbose=False)

        mock_config.assert_called_once()
        call_args = mock_config.call_args
        assert call_args.kwargs["level"] == logging.INFO

    @patch("src.main.logging.basicConfig")
    @patch("src.main.logging.FileHandler")
    @patch("src.main.logging.StreamHandler")
    def test_setup_logging_verbose(self, mock_stream, mock_file, mock_config):
        """Test verbose logging setup."""
        setup_logging(verbose=True)

        mock_config.assert_called_once()
        call_args = mock_config.call_args
        assert call_args.kwargs["level"] == logging.DEBUG


class TestQuantumComicGenerator:
    """Test QuantumComicGenerator class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Config(
            ibm_api_key="test_ibm",
            gemini_api_key="test_gemini",
            panels=3,
            characters=2,
        )
        # Add missing attributes for testing
        config.character_style = "default"
        config.art_style = "comic"
        config.generate_title = True
        config.create_archive = False
        config.cleanup_old = False
        config.keep_latest_comics = 10
        config.random_seed = None
        return config

    @pytest.fixture
    def generator(self, config):
        """Create generator instance."""
        return QuantumComicGenerator(config)

    def test_generator_initialization(self, generator, config):
        """Test generator initialization."""
        assert generator.config == config
        assert generator.logger is not None

    @patch("src.main.OutputManager")
    @patch("src.main.GeminiImageGenerator")
    @patch("src.main.decode_quantum_result")
    def test_generate_comic_with_test_bitstring(
        self, mock_decode, mock_gemini_class, mock_output_class, generator
    ):
        """Test comic generation with test bitstring."""
        # Setup mocks
        mock_narrative = MagicMock()
        mock_narrative.style_palette = "test palette"
        mock_prompts = ["prompt1", "prompt2", "prompt3"]
        mock_decode.return_value = (mock_narrative, mock_prompts)

        mock_gemini = MagicMock()
        mock_gemini.test_connection.return_value = True
        mock_comic = MagicMock()
        mock_gemini.generate_comic_strip.return_value = mock_comic
        mock_gemini_class.return_value = mock_gemini

        mock_output = MagicMock()
        mock_comic_dir = Path("/test/comic_dir")
        mock_output.create_comic_directory.return_value = mock_comic_dir
        mock_output.save_comic_strip.return_value = {
            "html": Path("/test/comic_dir/index.html")
        }
        mock_output_class.return_value = mock_output

        # Generate comic with test bitstring
        result_dir, bitstring = generator.generate_comic(
            skip_quantum=True,
            test_bitstring="101010",
        )

        assert result_dir == mock_comic_dir
        assert bitstring == "101010"

        # Verify calls
        mock_decode.assert_called_once()
        mock_gemini.generate_comic_strip.assert_called_once_with(
            mock_prompts, mock_narrative, generate_title=True
        )
        mock_output.save_comic_strip.assert_called_once()

    @patch("src.main.OutputManager")
    @patch("src.main.GeminiImageGenerator")
    @patch("src.main.decode_quantum_result")
    @patch("src.main.execute_quantum_circuit")
    @patch("src.main.create_quantum_circuit")
    def test_generate_comic_with_quantum(
        self,
        mock_create_circuit,
        mock_execute,
        mock_decode,
        mock_gemini_class,
        mock_output_class,
        generator,
    ):
        """Test comic generation with quantum execution."""
        # Setup mocks
        mock_circuit = MagicMock()
        mock_create_circuit.return_value = mock_circuit

        mock_result = MagicMock()
        mock_result.bitstring = "111000"
        mock_result.backend_name = "test_backend"
        mock_execute.return_value = mock_result

        mock_narrative = MagicMock()
        mock_narrative.style_palette = "quantum palette"
        mock_prompts = ["prompt1", "prompt2"]
        mock_decode.return_value = (mock_narrative, mock_prompts)

        mock_gemini = MagicMock()
        mock_gemini.test_connection.return_value = True
        mock_comic = MagicMock()
        mock_gemini.generate_comic_strip.return_value = mock_comic
        mock_gemini_class.return_value = mock_gemini

        mock_output = MagicMock()
        mock_comic_dir = Path("/test/quantum_comic")
        mock_output.create_comic_directory.return_value = mock_comic_dir
        mock_output.save_comic_strip.return_value = {
            "html": Path("/test/quantum_comic/index.html")
        }
        mock_output_class.return_value = mock_output

        # Generate comic with quantum
        result_dir, bitstring = generator.generate_comic()

        assert result_dir == mock_comic_dir
        assert bitstring == "111000"

        # Verify quantum execution
        mock_create_circuit.assert_called_once_with(panels=3, characters=2, seed=None)
        mock_execute.assert_called_once()

    @patch("src.main.OutputManager")
    @patch("src.main.GeminiImageGenerator")
    @patch("src.main.decode_quantum_result")
    def test_generate_comic_with_archive(
        self, mock_decode, mock_gemini_class, mock_output_class, generator
    ):
        """Test comic generation with archive creation."""
        generator.config.create_archive = True

        # Setup mocks
        mock_narrative = MagicMock()
        mock_prompts = ["prompt1"]
        mock_decode.return_value = (mock_narrative, mock_prompts)

        mock_gemini = MagicMock()
        mock_gemini.test_connection.return_value = True
        mock_comic = MagicMock()
        mock_gemini.generate_comic_strip.return_value = mock_comic
        mock_gemini_class.return_value = mock_gemini

        mock_output = MagicMock()
        mock_comic_dir = Path("/test/comic")
        mock_output.create_comic_directory.return_value = mock_comic_dir
        mock_output.save_comic_strip.return_value = {"html": Path("/test/index.html")}
        mock_output_class.return_value = mock_output

        # Generate comic
        generator.generate_comic(skip_quantum=True, test_bitstring="101")

        # Verify archive was created
        mock_output.export_comic_archive.assert_called_once_with(mock_comic_dir)

    @patch("src.main.OutputManager")
    @patch("src.main.GeminiImageGenerator")
    @patch("src.main.decode_quantum_result")
    def test_generate_comic_with_cleanup(
        self, mock_decode, mock_gemini_class, mock_output_class, generator
    ):
        """Test comic generation with cleanup."""
        generator.config.cleanup_old = True
        generator.config.keep_latest_comics = 5

        # Setup mocks
        mock_narrative = MagicMock()
        mock_prompts = ["prompt1"]
        mock_decode.return_value = (mock_narrative, mock_prompts)

        mock_gemini = MagicMock()
        mock_gemini.test_connection.return_value = True
        mock_comic = MagicMock()
        mock_gemini.generate_comic_strip.return_value = mock_comic
        mock_gemini_class.return_value = mock_gemini

        mock_output = MagicMock()
        mock_comic_dir = Path("/test/comic")
        mock_output.create_comic_directory.return_value = mock_comic_dir
        mock_output.save_comic_strip.return_value = {"html": Path("/test/index.html")}
        mock_output_class.return_value = mock_output

        # Generate comic
        generator.generate_comic(skip_quantum=True, test_bitstring="101")

        # Verify cleanup was called
        mock_output.cleanup_old_comics.assert_called_once_with(keep_latest=5)

    @patch("src.main.GeminiImageGenerator")
    def test_generate_comic_connection_failure(self, mock_gemini_class, generator):
        """Test comic generation with connection failure."""
        mock_gemini = MagicMock()
        mock_gemini.test_connection.return_value = False
        mock_gemini_class.return_value = mock_gemini

        with pytest.raises(RuntimeError, match="Failed to connect to Gemini API"):
            generator.generate_comic(skip_quantum=True, test_bitstring="101")

    @patch("src.main.GeminiImageGenerator")
    @patch("src.main.check_ibm_connection")
    def test_test_connections_success(
        self, mock_ibm_check, mock_gemini_class, generator
    ):
        """Test successful connection tests."""
        mock_ibm_check.return_value = True

        mock_gemini = MagicMock()
        mock_gemini.test_connection.return_value = True
        mock_gemini_class.return_value = mock_gemini

        result = generator.test_connections()

        assert result is True
        mock_ibm_check.assert_called_once_with(generator.config)

    @patch("src.main.GeminiImageGenerator")
    @patch("src.main.check_ibm_connection")
    def test_test_connections_ibm_failure(
        self, mock_ibm_check, mock_gemini_class, generator
    ):
        """Test connection test with IBM failure."""
        mock_ibm_check.return_value = False

        mock_gemini = MagicMock()
        mock_gemini.test_connection.return_value = True
        mock_gemini_class.return_value = mock_gemini

        result = generator.test_connections()

        assert result is False

    @patch("src.main.GeminiImageGenerator")
    @patch("src.main.check_ibm_connection")
    def test_test_connections_gemini_failure(
        self, mock_ibm_check, mock_gemini_class, generator
    ):
        """Test connection test with Gemini failure."""
        mock_ibm_check.return_value = True

        mock_gemini = MagicMock()
        mock_gemini.test_connection.return_value = False
        mock_gemini_class.return_value = mock_gemini

        result = generator.test_connections()

        assert result is False


class TestCreateParser:
    """Test argument parser creation."""

    def test_create_parser(self):
        """Test parser creation."""
        parser = create_parser()

        assert isinstance(parser, argparse.ArgumentParser)

        # Test some basic arguments
        args = parser.parse_args(["--panels", "6"])
        assert args.panels == 6

        args = parser.parse_args(["--simulator"])
        assert args.simulator is True

        args = parser.parse_args(["--test-bitstring", "101010"])
        assert args.test_bitstring == "101010"


class TestMain:
    """Test main function."""

    @patch("src.main.QuantumComicGenerator")
    @patch("src.main.validate_config")
    @patch("src.main.load_config")
    @patch("src.main.setup_logging")
    @patch("sys.argv", ["main.py"])
    def test_main_basic_generation(
        self, mock_logging, mock_load, mock_validate, mock_generator_class
    ):
        """Test basic comic generation."""
        # Setup mocks
        mock_config = MagicMock()
        mock_load.return_value = mock_config

        mock_generator = MagicMock()
        mock_comic_dir = Path("/test/comic")
        mock_generator.generate_comic.return_value = (mock_comic_dir, "101010")
        mock_generator_class.return_value = mock_generator

        # Run main
        result = main()

        assert result == 0
        mock_generator.generate_comic.assert_called_once()

    @patch("src.main.list_available_backends")
    @patch("src.main.validate_config")
    @patch("src.main.load_config")
    @patch("src.main.setup_logging")
    @patch("sys.argv", ["main.py", "--list-backends"])
    def test_main_list_backends(
        self, mock_logging, mock_load, mock_validate, mock_list_backends
    ):
        """Test listing backends."""
        mock_config = MagicMock()
        mock_load.return_value = mock_config

        mock_list_backends.return_value = [
            {
                "name": "backend1",
                "operational": True,
                "simulator": False,
                "num_qubits": 127,
            },
            {
                "name": "simulator",
                "operational": True,
                "simulator": True,
                "num_qubits": 32,
            },
        ]

        result = main()

        assert result == 0
        mock_list_backends.assert_called_once()

    @patch("src.main.QuantumComicGenerator")
    @patch("src.main.validate_config")
    @patch("src.main.load_config")
    @patch("src.main.setup_logging")
    @patch("sys.argv", ["main.py", "--test-connections"])
    def test_main_test_connections(
        self, mock_logging, mock_load, mock_validate, mock_generator_class
    ):
        """Test connection testing."""
        mock_config = MagicMock()
        mock_load.return_value = mock_config

        mock_generator = MagicMock()
        mock_generator.test_connections.return_value = True
        mock_generator_class.return_value = mock_generator

        result = main()

        assert result == 0
        mock_generator.test_connections.assert_called_once()

    @patch("src.main.QuantumComicGenerator")
    @patch("src.main.validate_config")
    @patch("src.main.load_config")
    @patch("src.main.setup_logging")
    @patch("sys.argv", ["main.py", "--test-bitstring", "101010"])
    def test_main_with_test_bitstring(
        self, mock_logging, mock_load, mock_validate, mock_generator_class
    ):
        """Test generation with test bitstring."""
        mock_config = MagicMock()
        mock_load.return_value = mock_config

        mock_generator = MagicMock()
        mock_comic_dir = Path("/test/comic")
        mock_generator.generate_comic.return_value = (mock_comic_dir, "101010")
        mock_generator_class.return_value = mock_generator

        result = main()

        assert result == 0
        mock_generator.generate_comic.assert_called_once_with(
            output_dir=None, skip_quantum=True, test_bitstring="101010"
        )

    @patch("src.main.QuantumComicGenerator")
    @patch("src.main.validate_config")
    @patch("src.main.load_config")
    @patch("src.main.setup_logging")
    @patch("sys.argv", ["main.py"])
    def test_main_keyboard_interrupt(
        self, mock_logging, mock_load, mock_validate, mock_generator_class
    ):
        """Test handling keyboard interrupt."""
        mock_config = MagicMock()
        mock_load.return_value = mock_config

        mock_generator = MagicMock()
        mock_generator.generate_comic.side_effect = KeyboardInterrupt()
        mock_generator_class.return_value = mock_generator

        result = main()

        assert result == 130

    @patch("src.main.QuantumComicGenerator")
    @patch("src.main.validate_config")
    @patch("src.main.load_config")
    @patch("src.main.setup_logging")
    @patch("sys.argv", ["main.py"])
    def test_main_error(
        self, mock_logging, mock_load, mock_validate, mock_generator_class
    ):
        """Test handling general error."""
        mock_config = MagicMock()
        mock_load.return_value = mock_config

        mock_generator = MagicMock()
        mock_generator.generate_comic.side_effect = Exception("Test error")
        mock_generator_class.return_value = mock_generator

        result = main()

        assert result == 1
