"""
Quantum circuit builder for the comic book generator.

This module creates an entangled quantum circuit that encodes the entire
comic narrative structure into quantum states.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

from src.config import Config, get_circuit_parameters


@dataclass
class CircuitRegisters:
    """Container for quantum circuit register information."""

    # Register sizes
    time_qubits: int
    action_qubits: int
    emotion_qubits: int
    camera_qubits: int
    style_qubits: int
    total_qubits: int

    # Starting indices for each register
    time_start: int = 0
    action_start: int = 0
    emotion_start: int = 0
    camera_start: int = 0
    style_start: int = 0

    def __post_init__(self):
        """Calculate register starting indices."""
        self.time_start = 0
        self.action_start = self.time_start + self.time_qubits
        self.emotion_start = self.action_start + self.action_qubits
        self.camera_start = self.emotion_start + self.emotion_qubits
        self.style_start = self.camera_start + self.camera_qubits

        # Verify total
        calculated_total = (
            self.time_qubits
            + self.action_qubits
            + self.emotion_qubits
            + self.camera_qubits
            + self.style_qubits
        )
        assert (
            calculated_total == self.total_qubits
        ), f"Register size mismatch: {calculated_total} != {self.total_qubits}"

    def time_index(self, panel: int) -> int:
        """Get qubit index for time register at panel."""
        if panel >= self.time_qubits:
            raise ValueError(
                f"Panel {panel} exceeds time register size {self.time_qubits}"
            )
        return self.time_start + panel

    def action_index(self, panel: int, bit: int) -> int:
        """Get qubit index for action register at panel and bit."""
        if panel * 2 + bit >= self.action_qubits:
            raise ValueError(f"Action index {panel}:{bit} exceeds register size")
        return self.action_start + (panel * 2) + bit

    def emotion_index(self, panel: int, bit: int) -> int:
        """Get qubit index for emotion register at panel and bit."""
        if panel * 2 + bit >= self.emotion_qubits:
            raise ValueError(f"Emotion index {panel}:{bit} exceeds register size")
        return self.emotion_start + (panel * 2) + bit

    def camera_index(self, panel: int, bit: int) -> int:
        """Get qubit index for camera register at panel and bit."""
        if panel * 2 + bit >= self.camera_qubits:
            raise ValueError(f"Camera index {panel}:{bit} exceeds register size")
        return self.camera_start + (panel * 2) + bit

    def style_index(self, bit: int) -> int:
        """Get qubit index for style register at bit."""
        if bit >= self.style_qubits:
            raise ValueError(
                f"Style bit {bit} exceeds register size {self.style_qubits}"
            )
        return self.style_start + bit


class CircuitBuilder:
    """Builds quantum circuits for comic generation."""

    def __init__(self, config: Config):
        """
        Initialize circuit builder with configuration.

        Args:
            config: Application configuration
        """
        self.config = config
        self.params = get_circuit_parameters(config)

        # Create register info
        self.registers = CircuitRegisters(
            time_qubits=self.params["time_qubits"],
            action_qubits=self.params["action_qubits"],
            emotion_qubits=self.params["emotion_qubits"],
            camera_qubits=self.params["camera_qubits"],
            style_qubits=self.params["style_qubits"],
            total_qubits=self.params["total_qubits"],
        )

        # Initialize circuit
        self.circuit: Optional[QuantumCircuit] = None

    def build(self) -> Tuple[QuantumCircuit, CircuitRegisters]:
        """
        Build the complete quantum circuit.

        Returns:
            Tuple of (QuantumCircuit, CircuitRegisters)
        """
        # Create quantum and classical registers
        qreg = QuantumRegister(self.registers.total_qubits, "q")
        creg = ClassicalRegister(self.registers.total_qubits, "c")
        self.circuit = QuantumCircuit(qreg, creg, name="quantum_comic")

        # Set random seed if specified
        if self.params["seed"] is not None:
            np.random.seed(self.params["seed"])

        # Build circuit layers
        self._add_initial_superposition()
        self._add_time_entanglement()
        self._add_panel_coupling()
        self._add_style_anchoring()
        self._add_measurements()

        return self.circuit, self.registers

    def _add_initial_superposition(self):
        """Add initial superposition to all registers."""
        # Time register: Start with T[0] in superposition
        self.circuit.h(self.registers.time_index(0))

        # Style register: All qubits in superposition
        for i in range(self.registers.style_qubits):
            self.circuit.h(self.registers.style_index(i))

        # Panel registers: Put action/emotion/camera bits in superposition
        for panel in range(self.config.panels):
            for bit in range(2):
                self.circuit.h(self.registers.action_index(panel, bit))
                self.circuit.h(self.registers.emotion_index(panel, bit))
                self.circuit.h(self.registers.camera_index(panel, bit))

    def _add_time_entanglement(self):
        """Create entanglement along the time chain."""
        panels = self.config.panels

        # Chain time qubits together
        for i in range(panels - 1):
            # CNOT from T[i] to T[i+1]
            self.circuit.cx(
                self.registers.time_index(i), self.registers.time_index(i + 1)
            )

            # Add small rotation for richer correlations
            self.circuit.rz(0.17, self.registers.time_index(i + 1))

    def _add_panel_coupling(self):
        """Couple panel registers to their time nodes."""
        panels = self.config.panels

        for panel in range(panels):
            time_idx = self.registers.time_index(panel)

            # Couple action bits to time
            for bit in range(2):
                self.circuit.crx(
                    0.11, time_idx, self.registers.action_index(panel, bit)
                )

            # Couple emotion bits to time
            for bit in range(2):
                self.circuit.cry(
                    0.09, time_idx, self.registers.emotion_index(panel, bit)
                )

            # Couple camera bits to time
            for bit in range(2):
                self.circuit.crz(
                    0.07, time_idx, self.registers.camera_index(panel, bit)
                )

    def _add_style_anchoring(self):
        """Anchor style register to key time nodes."""
        panels = self.config.panels

        # Connect first time node to first 2 style bits
        for i in range(min(2, self.registers.style_qubits)):
            self.circuit.cz(self.registers.time_index(0), self.registers.style_index(i))

        # Connect last time node to last 2 style bits
        if panels > 1:
            for i in range(2, min(4, self.registers.style_qubits)):
                self.circuit.cz(
                    self.registers.time_index(panels - 1), self.registers.style_index(i)
                )

    def _add_measurements(self):
        """Add measurement operations to all qubits."""
        for i in range(self.registers.total_qubits):
            self.circuit.measure(i, i)

    def get_circuit_depth(self) -> int:
        """
        Get the depth of the built circuit.

        Returns:
            Circuit depth
        """
        if self.circuit is None:
            raise ValueError("Circuit not built yet")
        return self.circuit.depth()

    def get_gate_count(self) -> dict:
        """
        Get gate counts for the built circuit.

        Returns:
            Dictionary of gate types and counts
        """
        if self.circuit is None:
            raise ValueError("Circuit not built yet")

        gate_counts = {}
        for instruction in self.circuit.data:
            gate_name = instruction.operation.name
            gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1

        return gate_counts

    def visualize(self, output_file: Optional[str] = None) -> str:
        """
        Generate circuit visualization.

        Args:
            output_file: Optional file path to save visualization

        Returns:
            Text representation of the circuit
        """
        if self.circuit is None:
            raise ValueError("Circuit not built yet")

        # Generate text representation
        text_repr = self.circuit.draw(output="text", fold=-1)

        if output_file:
            with open(output_file, "w") as f:
                f.write(str(text_repr))

        return str(text_repr)


def create_quantum_circuit(config: Config) -> Tuple[QuantumCircuit, CircuitRegisters]:
    """
    Create a quantum circuit from configuration.

    Args:
        config: Application configuration

    Returns:
        Tuple of (QuantumCircuit, CircuitRegisters)
    """
    builder = CircuitBuilder(config)
    return builder.build()


def validate_circuit(circuit: QuantumCircuit, registers: CircuitRegisters) -> bool:
    """
    Validate a quantum circuit.

    Args:
        circuit: Quantum circuit to validate
        registers: Register information

    Returns:
        True if circuit is valid

    Raises:
        ValueError: If circuit is invalid
    """
    # Check qubit count
    if circuit.num_qubits != registers.total_qubits:
        raise ValueError(
            f"Circuit qubit count {circuit.num_qubits} doesn't match "
            f"expected {registers.total_qubits}"
        )

    # Check classical bit count
    if circuit.num_clbits != registers.total_qubits:
        raise ValueError(
            f"Circuit classical bit count {circuit.num_clbits} doesn't match "
            f"expected {registers.total_qubits}"
        )

    # Check that all qubits are measured
    measured_qubits = set()
    for instruction in circuit.data:
        if instruction.operation.name == "measure":
            measured_qubits.add(instruction.qubits[0]._index)

    if len(measured_qubits) != registers.total_qubits:
        raise ValueError(
            f"Not all qubits are measured: {len(measured_qubits)} of "
            f"{registers.total_qubits}"
        )

    return True
