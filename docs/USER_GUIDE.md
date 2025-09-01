# Quantum Comic Book Generator - User Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [Advanced Features](#advanced-features)
4. [Troubleshooting](#troubleshooting)
5. [Examples](#examples)

## Getting Started

### Prerequisites

Before using the Quantum Comic Book Generator, you need:

1. **Python 3.8 or higher** installed on your system
2. **IBM Quantum Account** (free)
   - Sign up at [quantum.ibm.com](https://quantum.ibm.com)
   - Get your API token from the IBM Quantum dashboard
3. **Google AI Studio API Key** (free)
   - Get your key at [aistudio.google.com](https://aistudio.google.com)

### Installation

1. **Clone or download the project**:
   ```bash
   git clone https://github.com/yourusername/quantum-comic-book.git
   cd quantum-comic-book
   ```

2. **Set up Python environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure API keys**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` file and add your keys:
   ```
   IBM_API_KEY=your_ibm_quantum_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. **Test your setup**:
   ```bash
   python -m src.main --test-connections
   ```
   
   You should see:
   ```
   ✓ IBM Quantum connection successful
   ✓ Gemini API connection successful
   ```

## Basic Usage

### Generate Your First Comic

The simplest way to generate a quantum comic:

```bash
python -m src.main
```

This will:
1. Create a quantum circuit with 6 panels
2. Execute it on IBM Quantum
3. Generate comic panels with Gemini AI
4. Save the comic to the `output/` directory
5. Display the output location

### View Your Comic

After generation, you'll see output like:
```
✨ Quantum Comic Generated Successfully!
📁 Output directory: output/comic_20240315_142035
🎲 Quantum measurement: 101010110011...
🌐 View in browser: file:///path/to/output/comic_20240315_142035/index.html
```

Open the HTML file in your browser to view your comic!

### Customize Panel Count

Generate comics with different numbers of panels:

```bash
# 3-panel comic (quick story)
python -m src.main --panels 3

# 12-panel comic (extended narrative)
python -m src.main --panels 12
```

### Choose Art Styles

Select different visual styles:

```bash
# Manga style
python -m src.main --art-style manga

# Graphic novel style
python -m src.main --art-style graphic_novel

# Classic comic book style (default)
python -m src.main --art-style comic
```

### Character Styles

Change character types:

```bash
# Sci-fi characters
python -m src.main --character-style scifi

# Noir detectives
python -m src.main --character-style noir

# Contemporary urban (default)
python -m src.main --character-style default
```

## Advanced Features

### Combine Multiple Options

Create a sci-fi manga with 9 panels:

```bash
python -m src.main --panels 9 --art-style manga --character-style scifi
```

### Use Quantum Simulator

For faster testing without real quantum hardware:

```bash
python -m src.main --simulator
```

### Specify IBM Backend

Choose a specific quantum computer:

```bash
# List available backends
python -m src.main --list-backends

# Use a specific backend
python -m src.main --backend ibm_brisbane
```

### Test Mode

Test with a predefined bitstring (skips quantum execution):

```bash
python -m src.main --test-bitstring 101010110011001100101010110
```

This is useful for:
- Testing without quantum access
- Reproducing specific narratives
- Debugging image generation

### Custom Output Directory

Save comics to a specific location:

```bash
python -m src.main --output-dir /path/to/my/comics
```

### Archive Creation

Skip ZIP archive creation for faster generation:

```bash
python -m src.main --no-archive
```

### Cleanup Old Comics

Automatically remove old comics to save space:

```bash
# Keep only the 5 most recent comics
python -m src.main --cleanup --keep-latest 5
```

### Reproducible Results

Use a seed for consistent quantum circuit generation:

```bash
python -m src.main --seed 42
```

Note: The quantum measurement will still be probabilistic!

### Verbose Output

See detailed logs during generation:

```bash
python -m src.main --verbose
```

## Troubleshooting

### Common Issues and Solutions

#### "Failed to connect to IBM Quantum"

**Problem**: Cannot connect to IBM Quantum services.

**Solutions**:
1. Verify your IBM API key is correct in `.env`
2. Check your internet connection
3. Ensure your IBM Quantum account is active
4. Try using the simulator: `--simulator`

#### "Failed to connect to Gemini API"

**Problem**: Cannot connect to Google's Gemini service.

**Solutions**:
1. Verify your Gemini API key in `.env`
2. Check if you've exceeded API quotas
3. Ensure the API key has the correct permissions

#### "Circuit execution timeout"

**Problem**: Quantum execution takes too long.

**Solutions**:
1. Use a different backend: `--list-backends` then `--backend [name]`
2. Use the simulator for instant results: `--simulator`
3. Reduce panel count: `--panels 3`

#### "No image in response"

**Problem**: Gemini fails to generate images.

**Solutions**:
1. Check API quotas in Google AI Studio
2. Try again with fewer panels
3. Use test mode to skip image generation

#### Images don't match the narrative

**Problem**: Generated images don't reflect the quantum narrative.

**Solutions**:
1. This can happen due to AI limitations
2. Try different art styles
3. Generate multiple comics - each quantum measurement is unique!

### Getting Help

1. **Check logs**: Look at `quantum_comic.log` for detailed error messages
2. **Verbose mode**: Run with `--verbose` for more information
3. **Test connections**: Use `--test-connections` to verify setup

## Examples

### Example 1: Quick Test Comic

Generate a minimal comic for testing:

```bash
python -m src.main --panels 1 --simulator --no-title
```

### Example 2: High-Quality Graphic Novel

Create a detailed graphic novel style comic:

```bash
python -m src.main \
  --panels 12 \
  --art-style graphic_novel \
  --character-style noir \
  --verbose
```

### Example 3: Reproducible Sci-Fi Series

Generate consistent sci-fi comics:

```bash
# Episode 1
python -m src.main --character-style scifi --seed 100 --output-dir comics/episode1

# Episode 2 (same style, different quantum story)
python -m src.main --character-style scifi --seed 200 --output-dir comics/episode2
```

### Example 4: Batch Generation

Generate multiple comics automatically:

```python
# Save as generate_batch.py
from src.config import Config
from src.main import QuantumComicGenerator

config = Config(
    ibm_api_key="your_key",
    gemini_api_key="your_key",
    panels=6,
    use_simulator=True,
)

generator = QuantumComicGenerator(config)

for i in range(5):
    print(f"Generating comic {i+1}/5...")
    comic_dir, bitstring = generator.generate_comic()
    print(f"Saved to: {comic_dir}")
```

Run with: `python generate_batch.py`

## Tips for Best Results

1. **Start small**: Begin with 3-panel comics to test your setup
2. **Use simulator first**: Test with `--simulator` before using real quantum hardware
3. **Experiment with styles**: Different art styles produce very different results
4. **Save interesting bitstrings**: If you get a great narrative, save the bitstring to reproduce it
5. **Peak hours**: Quantum computers may be busy during peak hours - try different times
6. **Multiple attempts**: Each quantum measurement is unique - generate multiple comics for variety

## Understanding Your Comic

Each comic includes metadata showing:
- **Bitstring**: The quantum measurement that created this narrative
- **Panel breakdown**: Actions, emotions, and camera angles for each panel
- **Style palette**: The visual style determined by quantum measurement
- **Character focus**: Which character is featured in each panel

This metadata helps you understand how quantum mechanics shaped your unique story!

## Next Steps

- Explore different combinations of panels and styles
- Create comic series with consistent characters
- Share your quantum comics with others
- Experiment with the philosophical implications of quantum narratives

Remember: Each comic is a unique collapse of quantum possibilities into a single narrative reality!