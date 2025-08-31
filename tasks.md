# Quantum Comic Task Breakdown

## General Development Guidance

### **Core Principles**
- **Use Python:** Implement all components using Python with standard project layout
- **Test-Driven Development:** Write tests before implementing each task
- **Build and Test:** Use `make build` and `make test` commands consistently

### **Post-Task Checklist**
1. Update `arch.md` if any architectural changes were made
2. Mark the task as complete in `tasks.md`
3. Document implementation notes and architectural decisions in `tasks.md`
4. Update remaining tasks if architecture changes affected dependencies
5. Ensure `make build` and `make test` run successfully with no warnings
6. Run a linter and fix any issues, run tests and fix any issues
7. Commit changes with descriptive commit message following conventional commits
8. Don't include Claude as an author or coauthor

### **Code Quality Standards**
- **Testing:** Table-driven tests with subtests, >80% coverage, mock external dependencies

---

## Task List

### Phase 1: Project Setup
- [ ] **Task 1.1:** Set up virtual environment
  - Create Python virtual environment with `venv`
  - Create `requirements.txt` with dependencies:
    - qiskit
    - qiskit-ibm-runtime
    - google-genai
    - pillow
    - numpy
    - pytest (for testing)
    - black (for formatting)
    - pylint (for linting)
  - Create `.env.example` file with required environment variables

- [ ] **Task 1.2:** Create project structure
  - Create directories: `src/`, `tests/`, `output/`, `docs/`
  - Create `Makefile` with build, test, lint, and format targets
  - Create `.gitignore` for Python projects
  - Initialize git repository and create initial branch

- [ ] **Task 1.3:** Environment configuration module
  - Create `src/config.py` for managing environment variables (GEMINI_API_KEY, IBM_API_KEY)
  - Add validation for required environment variables
  - Create settings for configurable parameters (panels, characters, etc.)
  - Write tests for configuration loading

### Phase 2: Quantum Circuit Module
- [ ] **Task 2.1:** Circuit builder foundation
  - Create `src/quantum_circuit.py`
  - Implement register management (Time, Actions, Emotions, Camera, Style)
  - Add configurable qubit count calculator
  - Write unit tests for register indexing

- [ ] **Task 2.2:** Entanglement patterns
  - Implement time chain entanglement (T register)
  - Add panel-coupling entanglement (A/E/C to T)
  - Implement style anchor entanglement (S to T)
  - Write tests for circuit depth and gate counts

- [ ] **Task 2.3:** Circuit measurement
  - Add measurement operations for all qubits
  - Create circuit validation utilities
  - Write tests for complete circuit construction

### Phase 3: IBM Quantum Runtime Integration
- [ ] **Task 3.1:** Runtime service wrapper
  - Create `src/ibm_runtime.py`
  - Implement authentication using IBM_API_KEY
  - Add backend selection logic (least busy, specific backend)
  - Write mock-based tests for service connection

- [ ] **Task 3.2:** Sampler execution
  - Implement one-shot execution with Sampler
  - Add error handling for runtime failures
  - Create fallback logic for backend unavailability
  - Write tests with mocked IBM responses

- [ ] **Task 3.3:** Result processing
  - Parse bitstring from sampler results
  - Handle endianness conversion
  - Add result validation
  - Write tests for various bitstring formats

### Phase 4: Prompt Generation System
- [ ] **Task 4.1:** Lookup table manager
  - Create `src/prompts.py`
  - Implement lookup tables for Actions, Emotions, Camera angles, Style palettes
  - Add setting progression logic based on time bits
  - Write tests for all lookup combinations

- [ ] **Task 4.2:** Bitstring decoder
  - Implement bitstring slicing for each register
  - Create panel data extraction logic
  - Add global style decoding
  - Write tests for various bitstring inputs

- [ ] **Task 4.3:** Prompt composition
  - Create prompt templates with character bios
  - Implement panel-specific prompt generation
  - Add style and setting integration
  - Write tests for prompt consistency

### Phase 5: Gemini Image Generation
- [ ] **Task 5.1:** Gemini client wrapper
  - Create `src/gemini_client.py`
  - Implement authentication using GEMINI_API_KEY
  - Add error handling for API failures
  - Write mock-based tests for client operations

- [ ] **Task 5.2:** Image generation pipeline
  - Implement first panel generation (no prior image)
  - Add image-to-image conditioning for panels 2-N
  - Create image saving utilities
  - Write tests for generation workflow

- [ ] **Task 5.3:** Consistency management
  - Implement previous image tracking
  - Add prompt enhancement for consistency
  - Create retry logic for failed generations
  - Write tests for multi-panel sequences

### Phase 6: Output Generation
- [ ] **Task 6.1:** File management
  - Create `src/output_manager.py`
  - Implement timestamped output directory creation
  - Add image file saving with proper naming
  - Write tests for file operations

- [ ] **Task 6.2:** HTML strip generator
  - Implement HTML template creation
  - Add image embedding with metadata
  - Create responsive CSS styling
  - Write tests for HTML generation

- [ ] **Task 6.3:** Comic metadata
  - Store quantum bitstring with output
  - Create JSON export of panel data
  - Add generation timestamp and parameters
  - Write tests for metadata persistence

### Phase 7: Main Application
- [ ] **Task 7.1:** CLI interface
  - Create `src/main.py` with argument parsing
  - Add options for panels, characters, backend selection
  - Implement dry-run mode (circuit only, no execution)
  - Write integration tests for CLI

- [ ] **Task 7.2:** Orchestration logic
  - Wire together all modules
  - Add comprehensive error handling
  - Implement progress reporting
  - Write end-to-end tests

- [ ] **Task 7.3:** Logging and monitoring
  - Add structured logging throughout
  - Create debug mode with circuit visualization
  - Implement execution timing metrics
  - Write tests for logging output

### Phase 8: Testing and Documentation
- [ ] **Task 8.1:** Unit test suite
  - Achieve >80% code coverage
  - Add parameterized tests for edge cases
  - Create test fixtures for common data
  - Set up CI/CD with GitHub Actions

- [ ] **Task 8.2:** Integration tests
  - Test full pipeline with mocked external services
  - Add smoke tests for real services (optional run)
  - Create test data generators
  - Write performance benchmarks

- [ ] **Task 8.3:** Documentation
  - Create API documentation
  - Write user guide with examples
  - Add troubleshooting section
  - Document architectural decisions

### Phase 9: Optimization and Polish
- [ ] **Task 9.1:** Performance optimization
  - Profile quantum circuit construction
  - Optimize image generation pipeline
  - Add caching for repeated operations
  - Write performance regression tests

- [ ] **Task 9.2:** Error recovery
  - Implement graceful degradation
  - Add automatic retries with backoff
  - Create fallback to simulator if hardware fails
  - Write chaos tests for resilience

- [ ] **Task 9.3:** Final polish
  - Run full linting and formatting
  - Update all documentation
  - Create example outputs
  - Prepare release notes

## Dependencies
- Task 2.* depends on Task 1.*
- Task 3.* depends on Task 2.1
- Task 4.* can run parallel to Task 3.*
- Task 5.* depends on Task 1.3
- Task 6.* depends on Task 5.*
- Task 7.* depends on Tasks 2-6
- Task 8.* can start after Task 7.1
- Task 9.* depends on Task 8.*

## Notes
- Use environment variables for API keys: `GEMINI_API_KEY` and `IBM_API_KEY`
- Virtual environment should be named `venv` and excluded from git
- Output directory should be timestamped for each run
- Keep circuit depth minimal for real hardware execution
- Test with IBM simulator before using real quantum hardware