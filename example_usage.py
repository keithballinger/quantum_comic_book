#!/usr/bin/env python
"""
Example usage of the quantum comic book generator.

This script demonstrates various ways to use the quantum comic generator.
"""

import logging
from pathlib import Path

from src.config import Config
from src.main import QuantumComicGenerator, setup_logging


def example_basic_generation():
    """Generate a comic with default settings."""
    print("\n=== Basic Comic Generation ===")

    # Setup logging
    setup_logging(verbose=False)

    # Load configuration from environment
    config = Config(
        ibm_api_key="your_ibm_api_key",  # Set via environment variable
        gemini_api_key="your_gemini_api_key",  # Set via environment variable
        panels=6,
        characters=2,
    )

    # Create generator
    generator = QuantumComicGenerator(config)

    # Generate comic
    comic_dir, bitstring = generator.generate_comic()

    print(f"Comic saved to: {comic_dir}")
    print(f"Quantum measurement: {bitstring}")


def example_test_mode():
    """Generate a comic using a test bitstring (no quantum execution)."""
    print("\n=== Test Mode Generation ===")

    setup_logging(verbose=True)

    config = Config(
        ibm_api_key="not_needed_for_test",
        gemini_api_key="your_gemini_api_key",
        panels=3,
        characters=2,
        character_style="scifi",
        art_style="graphic_novel",
    )

    generator = QuantumComicGenerator(config)

    # Use a specific bitstring for testing
    test_bitstring = "101" + "110011" + "001100" + "101010" + "1101"
    comic_dir, bitstring = generator.generate_comic(
        skip_quantum=True,
        test_bitstring=test_bitstring,
    )

    print(f"Test comic saved to: {comic_dir}")


def example_custom_settings():
    """Generate a comic with custom settings."""
    print("\n=== Custom Settings Generation ===")

    setup_logging(verbose=False)

    config = Config(
        ibm_api_key="your_ibm_api_key",
        gemini_api_key="your_gemini_api_key",
        panels=12,  # More panels
        characters=2,
        character_style="noir",  # Noir style characters
        art_style="manga",  # Manga art style
        use_simulator=True,  # Use quantum simulator
        generate_title=True,  # Generate title card
        create_archive=True,  # Create ZIP archive
        random_seed=42,  # Fixed seed for reproducibility
    )

    generator = QuantumComicGenerator(config)

    # Custom output directory
    output_dir = Path("output/custom_comic")
    comic_dir, bitstring = generator.generate_comic(output_dir=output_dir)

    print(f"Custom comic saved to: {comic_dir}")


def example_batch_generation():
    """Generate multiple comics in a batch."""
    print("\n=== Batch Generation ===")

    setup_logging(verbose=False)

    config = Config(
        ibm_api_key="your_ibm_api_key",
        gemini_api_key="your_gemini_api_key",
        panels=6,
        characters=2,
        cleanup_old=True,  # Cleanup old comics
        keep_latest_comics=5,  # Keep only 5 latest
    )

    generator = QuantumComicGenerator(config)

    # Generate 3 comics
    for i in range(3):
        print(f"\nGenerating comic {i + 1}/3...")
        comic_dir, bitstring = generator.generate_comic()
        print(f"Comic {i + 1} saved to: {comic_dir}")


def example_test_connections():
    """Test connections to IBM and Gemini services."""
    print("\n=== Testing Service Connections ===")

    setup_logging(verbose=True)

    config = Config(
        ibm_api_key="your_ibm_api_key",
        gemini_api_key="your_gemini_api_key",
    )

    generator = QuantumComicGenerator(config)

    # Test connections
    success = generator.test_connections()

    if success:
        print("✓ All connections successful!")
    else:
        print("✗ Some connections failed. Check the logs.")


def example_minimal_circuit():
    """Generate a minimal comic for quick testing."""
    print("\n=== Minimal Circuit Generation ===")

    setup_logging(verbose=False)

    config = Config(
        ibm_api_key="your_ibm_api_key",
        gemini_api_key="your_gemini_api_key",
        panels=1,  # Minimal panels
        characters=1,  # Minimal characters
        use_simulator=True,  # Use simulator for speed
        generate_title=False,  # Skip title card
        create_archive=False,  # Skip archive
    )

    generator = QuantumComicGenerator(config)
    comic_dir, bitstring = generator.generate_comic()

    print(f"Minimal comic saved to: {comic_dir}")


if __name__ == "__main__":
    print("Quantum Comic Book Generator - Examples")
    print("=" * 50)

    # NOTE: Set your API keys as environment variables:
    # export IBM_API_KEY="your_key_here"
    # export GEMINI_API_KEY="your_key_here"

    # Uncomment the example you want to run:

    # example_basic_generation()
    # example_test_mode()
    # example_custom_settings()
    # example_batch_generation()
    # example_test_connections()
    # example_minimal_circuit()

    print("\nTo run examples, uncomment the desired function call above.")
    print("Make sure to set IBM_API_KEY and GEMINI_API_KEY environment variables.")
