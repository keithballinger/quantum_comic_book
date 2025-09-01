"""
Tests for quantum circuit builder.
"""

import pytest
from unittest.mock import MagicMock, patch

from qiskit import QuantumCircuit

from src.config import Config
from src.quantum_circuit import (
    CircuitBuilder,
    CircuitRegisters,
    create_quantum_circuit,
    validate_circuit,
)


class TestCircuitRegisters:
    """Test circuit register management."""

    def test_register_initialization(self):
        """Test register initialization with correct indices."""
        registers = CircuitRegisters(
            time_qubits=6,
            action_qubits=12,
            emotion_qubits=12,
            camera_qubits=12,
            style_qubits=4,
            total_qubits=46,
        )

        assert registers.time_start == 0
        assert registers.action_start == 6
        assert registers.emotion_start == 18
        assert registers.camera_start == 30
        assert registers.style_start == 42

    def test_register_size_mismatch(self):
        """Test that mismatched total raises assertion error."""
        with pytest.raises(AssertionError):
            CircuitRegisters(
                time_qubits=6,
                action_qubits=12,
                emotion_qubits=12,
                camera_qubits=12,
                style_qubits=4,
                total_qubits=100,  # Wrong total
            )

    def test_time_index(self):
        """Test time register indexing."""
        registers = CircuitRegisters(
            time_qubits=6,
            action_qubits=12,
            emotion_qubits=12,
            camera_qubits=12,
            style_qubits=4,
            total_qubits=46,
        )

        assert registers.time_index(0) == 0
        assert registers.time_index(5) == 5

        with pytest.raises(ValueError):
            registers.time_index(6)  # Out of bounds

    def test_action_index(self):
        """Test action register indexing."""
        registers = CircuitRegisters(
            time_qubits=6,
            action_qubits=12,
            emotion_qubits=12,
            camera_qubits=12,
            style_qubits=4,
            total_qubits=46,
        )

        # Panel 0, bit 0: should be at index 6
        assert registers.action_index(0, 0) == 6
        # Panel 0, bit 1: should be at index 7
        assert registers.action_index(0, 1) == 7
        # Panel 5, bit 1: should be at index 17
        assert registers.action_index(5, 1) == 17

        with pytest.raises(ValueError):
            registers.action_index(6, 0)  # Panel out of bounds

    def test_emotion_index(self):
        """Test emotion register indexing."""
        registers = CircuitRegisters(
            time_qubits=6,
            action_qubits=12,
            emotion_qubits=12,
            camera_qubits=12,
            style_qubits=4,
            total_qubits=46,
        )

        # Panel 0, bit 0: should be at index 18
        assert registers.emotion_index(0, 0) == 18
        # Panel 0, bit 1: should be at index 19
        assert registers.emotion_index(0, 1) == 19
        # Panel 5, bit 1: should be at index 29
        assert registers.emotion_index(5, 1) == 29

    def test_camera_index(self):
        """Test camera register indexing."""
        registers = CircuitRegisters(
            time_qubits=6,
            action_qubits=12,
            emotion_qubits=12,
            camera_qubits=12,
            style_qubits=4,
            total_qubits=46,
        )

        # Panel 0, bit 0: should be at index 30
        assert registers.camera_index(0, 0) == 30
        # Panel 5, bit 1: should be at index 41
        assert registers.camera_index(5, 1) == 41

    def test_style_index(self):
        """Test style register indexing."""
        registers = CircuitRegisters(
            time_qubits=6,
            action_qubits=12,
            emotion_qubits=12,
            camera_qubits=12,
            style_qubits=4,
            total_qubits=46,
        )

        assert registers.style_index(0) == 42
        assert registers.style_index(3) == 45

        with pytest.raises(ValueError):
            registers.style_index(4)  # Out of bounds


class TestCircuitBuilder:
    """Test circuit builder functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(
            ibm_api_key="test",
            gemini_api_key="test",
            panels=6,
            characters=2,
            seed=42,
        )

    @pytest.fixture
    def builder(self, config):
        """Create circuit builder."""
        return CircuitBuilder(config)

    def test_builder_initialization(self, builder):
        """Test circuit builder initialization."""
        assert builder.config.panels == 6
        assert builder.registers.total_qubits == 46
        assert builder.circuit is None

    def test_build_circuit(self, builder):
        """Test building a complete circuit."""
        circuit, registers = builder.build()

        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 46
        assert circuit.num_clbits == 46
        assert circuit.name == "quantum_comic"
        assert registers.total_qubits == 46

    def test_circuit_has_hadamard_gates(self, builder):
        """Test that circuit contains Hadamard gates for superposition."""
        circuit, _ = builder.build()

        gate_counts = builder.get_gate_count()
        assert "h" in gate_counts
        # Should have H gates on T[0], all style bits, and all panel bits
        # T[0]: 1, Style: 4, Panels: 6*2*3 = 36, Total: 41
        assert gate_counts["h"] == 41

    def test_circuit_has_entanglement_gates(self, builder):
        """Test that circuit contains entanglement gates."""
        circuit, _ = builder.build()

        gate_counts = builder.get_gate_count()

        # Should have CNOT gates for time chain
        assert "cx" in gate_counts
        assert gate_counts["cx"] == 5  # 5 connections in 6-panel chain

        # Should have controlled rotations
        assert "crx" in gate_counts
        assert "cry" in gate_counts
        assert "crz" in gate_counts

        # Should have CZ gates for style anchoring
        assert "cz" in gate_counts

    def test_circuit_has_measurements(self, builder):
        """Test that all qubits are measured."""
        circuit, _ = builder.build()

        gate_counts = builder.get_gate_count()
        assert "measure" in gate_counts
        assert gate_counts["measure"] == 46  # All qubits measured

    def test_circuit_depth(self, builder):
        """Test circuit depth calculation."""
        circuit, _ = builder.build()

        depth = builder.get_circuit_depth()
        assert depth > 0
        # Circuit should be shallow (<= 20 for noise resilience)
        assert depth <= 20

    def test_get_gate_count(self, builder):
        """Test gate counting."""
        circuit, _ = builder.build()

        gate_counts = builder.get_gate_count()
        assert isinstance(gate_counts, dict)
        assert len(gate_counts) > 0

        # Verify total gate count
        total_gates = sum(gate_counts.values())
        assert total_gates > 0

    def test_visualize(self, builder, tmp_path):
        """Test circuit visualization."""
        circuit, _ = builder.build()

        # Test text visualization
        text_repr = builder.visualize()
        assert isinstance(text_repr, str)
        assert len(text_repr) > 0

        # Test saving to file
        output_file = tmp_path / "circuit.txt"
        builder.visualize(str(output_file))
        assert output_file.exists()

        with open(output_file, "r") as f:
            content = f.read()
            assert content == text_repr

    def test_build_before_operations(self, builder):
        """Test that operations fail before building."""
        with pytest.raises(ValueError, match="Circuit not built"):
            builder.get_circuit_depth()

        with pytest.raises(ValueError, match="Circuit not built"):
            builder.get_gate_count()

        with pytest.raises(ValueError, match="Circuit not built"):
            builder.visualize()

    def test_seed_reproducibility(self, config):
        """Test that same seed produces same circuit."""
        builder1 = CircuitBuilder(config)
        circuit1, _ = builder1.build()

        builder2 = CircuitBuilder(config)
        circuit2, _ = builder2.build()

        # Same seed should produce identical circuits
        assert circuit1 == circuit2


class TestCreateQuantumCircuit:
    """Test circuit creation helper function."""

    def test_create_quantum_circuit(self):
        """Test creating circuit from config."""
        config = Config(
            ibm_api_key="test",
            gemini_api_key="test",
            panels=4,
            characters=2,
        )

        circuit, registers = create_quantum_circuit(config)

        assert isinstance(circuit, QuantumCircuit)
        assert registers.time_qubits == 4
        assert registers.total_qubits == 32  # 4 + 4*2*3 + 4


class TestValidateCircuit:
    """Test circuit validation."""

    def test_validate_valid_circuit(self):
        """Test validating a valid circuit."""
        config = Config(
            ibm_api_key="test",
            gemini_api_key="test",
            panels=3,
            characters=1,
        )

        circuit, registers = create_quantum_circuit(config)
        assert validate_circuit(circuit, registers) is True

    def test_validate_wrong_qubit_count(self):
        """Test validation fails with wrong qubit count."""
        # Create a circuit with wrong number of qubits
        circuit = QuantumCircuit(10, 10)
        registers = CircuitRegisters(
            time_qubits=3,
            action_qubits=6,
            emotion_qubits=6,
            camera_qubits=6,
            style_qubits=4,
            total_qubits=25,
        )

        with pytest.raises(ValueError, match="qubit count"):
            validate_circuit(circuit, registers)

    def test_validate_wrong_classical_bits(self):
        """Test validation fails with wrong classical bit count."""
        circuit = QuantumCircuit(25, 20)  # Wrong classical bit count
        registers = CircuitRegisters(
            time_qubits=3,
            action_qubits=6,
            emotion_qubits=6,
            camera_qubits=6,
            style_qubits=4,
            total_qubits=25,
        )

        with pytest.raises(ValueError, match="classical bit count"):
            validate_circuit(circuit, registers)

    def test_validate_unmeasured_qubits(self):
        """Test validation fails with unmeasured qubits."""
        circuit = QuantumCircuit(3, 3)
        circuit.h(0)
        circuit.h(1)
        circuit.h(2)
        circuit.measure(0, 0)  # Only measure first qubit

        registers = CircuitRegisters(
            time_qubits=1,
            action_qubits=0,
            emotion_qubits=0,
            camera_qubits=0,
            style_qubits=2,
            total_qubits=3,
        )

        with pytest.raises(ValueError, match="Not all qubits are measured"):
            validate_circuit(circuit, registers)


@pytest.mark.parametrize(
    "panels,expected_depth_range",
    [
        (1, (1, 10)),  # Very small circuit
        (6, (5, 20)),  # Default size
        (12, (10, 35)),  # Large circuit - slightly more depth allowed
    ],
)
def test_circuit_depth_scales(panels, expected_depth_range):
    """Test that circuit depth scales appropriately with panels."""
    config = Config(
        ibm_api_key="test",
        gemini_api_key="test",
        panels=panels,
        characters=2,
    )

    builder = CircuitBuilder(config)
    circuit, _ = builder.build()

    depth = builder.get_circuit_depth()
    min_depth, max_depth = expected_depth_range

    assert min_depth <= depth <= max_depth
