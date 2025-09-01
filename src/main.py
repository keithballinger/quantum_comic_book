"""
Main application for quantum comic book generator.

This module provides the CLI interface and orchestration for generating
quantum-driven comic strips.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

from src.config import Config, load_config, validate_config, get_circuit_parameters
from src.quantum_circuit import create_quantum_circuit, CircuitRegisters
from src.ibm_runtime import (
    execute_quantum_circuit,
    check_ibm_connection,
    list_available_backends,
)
from src.prompts import decode_quantum_result
from src.gemini_client import GeminiClient
from src.output_manager import OutputManager


# Configure logging
def setup_logging(verbose: bool = False):
    """
    Configure logging for the application.

    Args:
        verbose: Enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("quantum_comic.log"),
        ],
    )

    # Reduce noise from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("qiskit").setLevel(logging.WARNING)


class QuantumComicGenerator:
    """Main application class for quantum comic generation."""

    def __init__(self, config: Config):
        """
        Initialize quantum comic generator.

        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def generate_comic(
        self,
        output_dir: Optional[Path] = None,
        skip_quantum: bool = False,
        test_bitstring: Optional[str] = None,
    ) -> Tuple[Path, str]:
        """
        Generate a complete quantum comic strip.

        Args:
            output_dir: Optional output directory
            skip_quantum: Skip quantum execution and use test bitstring
            test_bitstring: Optional bitstring for testing

        Returns:
            Tuple of (output directory, bitstring)
        """
        self.logger.info("Starting quantum comic generation")

        # Step 1: Get quantum measurement
        if skip_quantum and test_bitstring:
            self.logger.info(f"Using test bitstring: {test_bitstring}")
            bitstring = test_bitstring
        else:
            bitstring = self._execute_quantum_circuit()

        # Step 2: Decode into narrative and prompts
        self.logger.info("Decoding quantum result into narrative")
        circuit_params = get_circuit_parameters(self.config)
        registers = CircuitRegisters(
            time_qubits=circuit_params["time_qubits"],
            action_qubits=circuit_params["action_qubits"],
            emotion_qubits=circuit_params["emotion_qubits"],
            camera_qubits=circuit_params["camera_qubits"],
            style_qubits=circuit_params["style_qubits"],
            total_qubits=circuit_params["total_qubits"],
        )
        narrative, prompts = decode_quantum_result(
            bitstring,
            registers,
            config=self.config,
        )

        self.logger.info(f"Generated narrative with {len(prompts)} panels")
        self.logger.info(f"Style: {narrative.style_palette}")

        # Step 3: Generate images
        self.logger.info("Generating comic panel images")
        generator = GeminiClient(self.config)

        if not generator.test_connection():
            raise RuntimeError("Failed to connect to Gemini API")

        comic = generator.generate_comic_strip(
            prompts,
            narrative,
            generate_title=self.config.generate_title,
        )

        # Step 4: Save output
        self.logger.info("Saving comic strip to disk")
        output_manager = OutputManager(self.config)

        if output_dir is None:
            comic_dir = output_manager.create_comic_directory()
        else:
            comic_dir = output_dir
            comic_dir.mkdir(parents=True, exist_ok=True)

        saved_files = output_manager.save_comic_strip(
            comic,
            comic_dir,
            save_combined=True,
        )

        # Step 5: Create archive if requested
        if self.config.create_archive:
            archive_path = output_manager.export_comic_archive(comic_dir)
            self.logger.info(f"Created archive: {archive_path}")

        # Step 6: Cleanup old comics if needed
        if self.config.cleanup_old:
            output_manager.cleanup_old_comics(
                keep_latest=self.config.keep_latest_comics
            )

        self.logger.info(f"Comic strip saved to: {comic_dir}")
        self.logger.info(f"View in browser: {saved_files['html']}")

        return comic_dir, bitstring

    def _execute_quantum_circuit(self) -> str:
        """
        Execute quantum circuit and get measurement.

        Returns:
            Measurement bitstring
        """
        self.logger.info("Creating quantum circuit")
        circuit, registers = create_quantum_circuit(self.config)

        self.logger.info(f"Executing circuit with {registers.total_qubits} qubits")
        
        result = execute_quantum_circuit(
            circuit,
            self.config,
            shots=1,  # Single shot for pure quantum collapse
        )
        self.logger.info(f"Quantum measurement: {result.bitstring}")
        self.logger.info(f"Backend used: {result.backend_name}")
        return result.bitstring

    def test_connections(self) -> bool:
        """
        Test connections to IBM and Gemini services.

        Returns:
            True if all connections successful
        """
        self.logger.info("Testing service connections")

        # Test IBM connection
        self.logger.info("Testing IBM Quantum connection")
        ibm_ok = check_ibm_connection(self.config)
        if ibm_ok:
            self.logger.info("✓ IBM Quantum connection successful")
        else:
            self.logger.error("✗ IBM Quantum connection failed")

        # Test Gemini connection
        self.logger.info("Testing Gemini API connection")
        try:
            generator = GeminiClient(self.config)
            gemini_ok = generator.test_connection()
            if gemini_ok:
                self.logger.info("✓ Gemini API connection successful")
            else:
                self.logger.error("✗ Gemini API connection failed")
        except Exception as e:
            self.logger.error(f"✗ Gemini API connection failed: {e}")
            gemini_ok = False

        return ibm_ok and gemini_ok


def create_parser() -> argparse.ArgumentParser:
    """
    Create command-line argument parser.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Generate quantum-driven comic strips",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a comic with default settings
  python -m src.main
  
  # Use specific number of panels
  python -m src.main --panels 6
  
  # Test with a specific bitstring (skip quantum execution)
  python -m src.main --test-bitstring 101010101010
  
  # List available quantum backends
  python -m src.main --list-backends
  
  # Test connections only
  python -m src.main --test-connections
        """,
    )

    # Main options
    parser.add_argument(
        "--panels",
        type=int,
        default=6,
        help="Number of comic panels to generate (default: 6)",
    )

    parser.add_argument(
        "--characters",
        type=int,
        default=2,
        help="Number of characters in the story (default: 2)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for comic files",
    )

    # Style options
    parser.add_argument(
        "--character-style",
        choices=["default", "scifi", "noir"],
        default="default",
        help="Character bio style (default: default)",
    )

    parser.add_argument(
        "--art-style",
        choices=["comic", "graphic_novel", "manga"],
        default="comic",
        help="Art style for panels (default: comic)",
    )

    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Skip title card generation",
    )

    # Backend options
    parser.add_argument(
        "--backend",
        type=str,
        help="Specific IBM backend to use",
    )

    parser.add_argument(
        "--simulator",
        action="store_true",
        help="Use IBM quantum simulator instead of real hardware",
    )

    # Testing options
    parser.add_argument(
        "--test-bitstring",
        type=str,
        help="Use specific bitstring (skip quantum execution)",
    )

    parser.add_argument(
        "--test-connections",
        action="store_true",
        help="Test service connections and exit",
    )

    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="List available quantum backends and exit",
    )

    # Output options
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Skip creating ZIP archive",
    )

    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up old comic directories",
    )

    parser.add_argument(
        "--keep-latest",
        type=int,
        default=10,
        help="Number of latest comics to keep when cleaning up (default: 10)",
    )

    # Other options
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for circuit generation",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--config-file",
        type=Path,
        help="Path to configuration file",
    )

    return parser


def main():
    """Main entry point for the application."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        config = load_config(env_file=args.config_file)

        # Override with command-line arguments
        if args.panels:
            config.panels = args.panels
        if args.characters:
            config.characters = args.characters
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.backend:
            config.ibm_backend = args.backend
        if args.simulator:
            config.use_simulator = True
        if args.character_style:
            config.character_style = args.character_style
        if args.art_style:
            config.art_style = args.art_style
        if args.no_title:
            config.generate_title = False
        if args.no_archive:
            config.create_archive = False
        if args.cleanup:
            config.cleanup_old = True
        if args.keep_latest:
            config.keep_latest_comics = args.keep_latest
        if args.seed:
            config.random_seed = args.seed

        # Validate configuration
        validate_config(config)

        # Handle special commands
        if args.list_backends:
            logger.info("Listing available quantum backends")
            backends = list_available_backends(config)

            print("\nAvailable Quantum Backends:")
            print("-" * 50)
            for backend in backends:
                status = "✓" if backend.get("operational") else "✗"
                sim = " (simulator)" if backend.get("simulator") else ""
                print(f"{status} {backend['name']}{sim}")
                print(f"   Qubits: {backend.get('num_qubits', 'N/A')}")
            return 0

        # Create generator
        generator = QuantumComicGenerator(config)

        if args.test_connections:
            success = generator.test_connections()
            return 0 if success else 1

        # Generate comic
        skip_quantum = args.test_bitstring is not None
        comic_dir, bitstring = generator.generate_comic(
            output_dir=args.output_dir,
            skip_quantum=skip_quantum,
            test_bitstring=args.test_bitstring,
        )

        print(f"\n✨ Quantum Comic Generated Successfully!")
        print(f"📁 Output directory: {comic_dir}")
        print(f"🎲 Quantum measurement: {bitstring}")
        print(f"🌐 View in browser: file://{comic_dir}/index.html")

        return 0

    except KeyboardInterrupt:
        logger.info("Generation cancelled by user")
        return 130

    except Exception as e:
        logger.error(f"Error generating comic: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
