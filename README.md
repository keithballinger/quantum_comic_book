# Quantum Comic Book Generator 🎨🔬

A unique application that uses quantum computing to generate comic strips, combining IBM Quantum circuits with Google's Gemini AI for a truly quantum-driven creative experience.

## Overview

This project leverages real quantum measurements from IBM Quantum computers to generate narrative structures for comic strips. Each quantum measurement creates a unique story path, with the resulting bitstring decoded into narrative elements like actions, emotions, camera angles, and visual styles. The narrative is then brought to life using Gemini's image generation capabilities.

## Features

- **Quantum Circuit Execution**: Run custom quantum circuits on IBM Quantum hardware or simulators
- **Narrative Generation**: Decode quantum measurements into structured comic narratives
- **AI Image Generation**: Create visually consistent comic panels using Gemini AI
- **Multiple Art Styles**: Support for comic, graphic novel, and manga styles
- **Character Variations**: Different character archetypes (default, sci-fi, noir)
- **Visual Continuity**: Image-to-image conditioning for consistent characters across panels
- **HTML Viewer**: Beautiful web-based comic strip viewer with metadata
- **Archive Export**: ZIP archive creation for easy sharing

## Installation

### Prerequisites

- Python 3.8+
- IBM Quantum account (free at [quantum.ibm.com](https://quantum.ibm.com))
- Google AI Studio API key (free at [aistudio.google.com](https://aistudio.google.com))

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/quantum-comic-book.git
cd quantum-comic-book
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys:
# IBM_API_KEY=your_ibm_quantum_api_key
# GEMINI_API_KEY=your_gemini_api_key
```

## Usage

### Basic Generation

Generate a quantum comic with default settings:
```bash
python -m src.main
```

### Custom Settings

Generate with specific number of panels:
```bash
python -m src.main --panels 6 --characters 2
```

Use different art styles:
```bash
python -m src.main --art-style manga --character-style scifi
```

### Test Mode

Test without quantum execution using a specific bitstring:
```bash
python -m src.main --test-bitstring 101010110011001100101010110
```

### Other Commands

List available quantum backends:
```bash
python -m src.main --list-backends
```

Test service connections:
```bash
python -m src.main --test-connections
```

### Command Line Options

```
Options:
  --panels N              Number of comic panels (default: 6)
  --characters N          Number of characters (default: 2)
  --output-dir PATH       Output directory for comics
  --character-style       Style: default, scifi, noir
  --art-style            Art style: comic, graphic_novel, manga
  --no-title             Skip title card generation
  --backend NAME         Specific IBM backend to use
  --simulator            Use quantum simulator
  --test-bitstring STR   Test with specific bitstring
  --test-connections     Test service connections
  --list-backends        List available backends
  --no-archive           Skip ZIP archive creation
  --cleanup              Clean up old comics
  --keep-latest N        Keep N latest comics (default: 10)
  --seed N               Random seed for reproducibility
  --verbose              Enable verbose logging
```

## How It Works

### 1. Quantum Circuit Design

The application creates a quantum circuit with specialized registers:
- **Time Register**: Controls narrative flow and scene progression
- **Action Register**: Determines character actions in each panel
- **Emotion Register**: Sets emotional states for characters
- **Camera Register**: Controls visual perspective and framing
- **Style Register**: Determines overall visual palette

### 2. Quantum Entanglement

The circuit uses various entanglement patterns:
- **Time Chain**: Creates narrative continuity
- **Panel Coupling**: Links adjacent panels for flow
- **Style Anchoring**: Maintains consistent visual style

### 3. Measurement & Decoding

The quantum measurement produces a bitstring that is decoded into:
- Panel-specific actions and emotions
- Camera angles and settings
- Overall style palette
- Character focus for each panel

### 4. Image Generation

Using the decoded narrative:
1. Prompts are generated for each panel
2. Gemini generates images with visual continuity
3. Panels use image-to-image conditioning for consistency
4. Optional title card is generated

### 5. Output Generation

The final comic is saved with:
- Individual panel images
- Combined strip image
- HTML viewer with metadata
- JSON narrative data
- Optional ZIP archive

## Project Structure

```
quantum-comic-book/
├── src/
│   ├── config.py           # Configuration management
│   ├── quantum_circuit.py  # Quantum circuit creation
│   ├── ibm_runtime.py      # IBM Quantum integration
│   ├── prompts.py          # Narrative decoding
│   ├── gemini_client.py    # Image generation
│   ├── output_manager.py   # File management
│   └── main.py             # CLI application
├── tests/                   # Comprehensive test suite
├── output/                  # Generated comics
├── docs/                    # Documentation
├── requirements.txt         # Python dependencies
└── example_usage.py        # Usage examples
```

## Philosophy: Many-Worlds Narrative

Every quantum measurement collapses into a unique narrative universe. The same quantum circuit, when measured, produces different stories - embodying the many-worlds interpretation where every possibility exists until observation collapses it into one reality. Each comic strip is literally a different universe of the same story.
