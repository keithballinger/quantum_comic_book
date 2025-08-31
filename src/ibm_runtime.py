"""
IBM Quantum Runtime integration for circuit execution.

This module handles authentication, backend selection, and circuit execution
on IBM Quantum hardware or simulators.
"""

import logging
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import (
    QiskitRuntimeService,
    Session,
    Sampler,
    SamplerOptions,
    IBMBackend,
)
from qiskit_ibm_runtime.exceptions import IBMRuntimeError

from src.config import Config, get_runtime_options


logger = logging.getLogger(__name__)


class RuntimeError(Exception):
    """Runtime execution error."""

    pass


@dataclass
class ExecutionResult:
    """Container for quantum execution results."""

    bitstring: str
    backend_name: str
    job_id: Optional[str] = None
    execution_time: Optional[float] = None
    shots: int = 1
    quasi_probabilities: Optional[Dict[int, float]] = None
    metadata: Optional[Dict[str, Any]] = None


class IBMRuntimeManager:
    """Manages IBM Quantum Runtime service and execution."""

    def __init__(self, config: Config):
        """
        Initialize runtime manager with configuration.

        Args:
            config: Application configuration
        """
        self.config = config
        self.options = get_runtime_options(config)
        self.service: Optional[QiskitRuntimeService] = None
        self.backend: Optional[IBMBackend] = None

    def connect(self) -> None:
        """
        Connect to IBM Quantum service.

        Raises:
            RuntimeError: If connection fails
        """
        try:
            # Initialize service with API key
            self.service = QiskitRuntimeService(
                channel="ibm_quantum",
                token=self.config.ibm_api_key,
            )
            logger.info("Connected to IBM Quantum service")

        except Exception as e:
            raise RuntimeError(f"Failed to connect to IBM Quantum: {e}")

    def select_backend(self) -> IBMBackend:
        """
        Select appropriate backend based on configuration.

        Returns:
            Selected backend

        Raises:
            RuntimeError: If no suitable backend found
        """
        if not self.service:
            raise RuntimeError("Service not connected")

        try:
            if self.options["use_simulator"]:
                # Use simulator
                backend = self.service.backend("ibmq_qasm_simulator")
                logger.info("Using simulator backend")

            elif self.options["backend"]:
                # Use specified backend
                backend = self.service.backend(self.options["backend"])
                logger.info(f"Using specified backend: {self.options['backend']}")

            else:
                # Auto-select least busy backend
                backend = self.service.least_busy(
                    operational=True,
                    simulator=False,
                    min_num_qubits=self.config.panels
                    + (2 * self.config.panels * 3)
                    + 4,
                )
                if not backend:
                    # Fallback to any available backend
                    backends = self.service.backends(
                        operational=True,
                        simulator=False,
                        min_num_qubits=self.config.panels
                        + (2 * self.config.panels * 3)
                        + 4,
                    )
                    if not backends:
                        raise RuntimeError("No suitable quantum backends available")
                    backend = backends[0]

                logger.info(f"Auto-selected backend: {backend.name}")

            self.backend = backend
            return backend

        except Exception as e:
            raise RuntimeError(f"Failed to select backend: {e}")

    def execute_circuit(
        self,
        circuit: QuantumCircuit,
        shots: int = 1,
        retry_on_error: bool = True,
    ) -> ExecutionResult:
        """
        Execute quantum circuit on selected backend.

        Args:
            circuit: Quantum circuit to execute
            shots: Number of measurement shots (default 1 for pure collapse)
            retry_on_error: Whether to retry on execution errors

        Returns:
            Execution result with bitstring

        Raises:
            RuntimeError: If execution fails
        """
        if not self.backend:
            self.backend = self.select_backend()

        retries = self.options["max_retries"] if retry_on_error else 1
        last_error = None

        for attempt in range(retries):
            try:
                logger.info(f"Executing circuit (attempt {attempt + 1}/{retries})")

                # Configure sampler options
                sampler_options = SamplerOptions(
                    default_shots=shots,
                )

                # Create session and execute
                with Session(service=self.service, backend=self.backend) as session:
                    sampler = Sampler(session=session, options=sampler_options)

                    # Run the circuit
                    job = sampler.run([circuit])
                    result = job.result()

                    # Extract bitstring from results
                    bitstring = self._extract_bitstring(result, circuit.num_clbits)

                    # Create execution result
                    execution_result = ExecutionResult(
                        bitstring=bitstring,
                        backend_name=self.backend.name,
                        job_id=getattr(job, "job_id", None),
                        shots=shots,
                        quasi_probabilities=(
                            result.quasi_dists[0] if result.quasi_dists else None
                        ),
                        metadata={
                            "backend_version": getattr(
                                self.backend, "version", "unknown"
                            ),
                            "attempt": attempt + 1,
                        },
                    )

                    logger.info(f"Circuit executed successfully on {self.backend.name}")
                    return execution_result

            except IBMRuntimeError as e:
                last_error = e
                logger.warning(f"Execution attempt {attempt + 1} failed: {e}")

                if attempt < retries - 1:
                    # Try different backend on retry
                    if not self.options["backend"]:  # Only if not fixed backend
                        logger.info("Selecting alternative backend for retry")
                        self.backend = None
                        self.backend = self.select_backend()

            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error during execution: {e}")
                break

        raise RuntimeError(
            f"Circuit execution failed after {retries} attempts: {last_error}"
        )

    def _extract_bitstring(self, result: Any, num_bits: int) -> str:
        """
        Extract bitstring from sampler results.

        Args:
            result: Sampler result object
            num_bits: Expected number of bits

        Returns:
            Bitstring as string
        """
        # Get quasi-probability distribution
        if not result.quasi_dists or not result.quasi_dists[0]:
            raise RuntimeError("No results returned from quantum execution")

        dist = result.quasi_dists[0]

        # For single shot, take highest probability outcome
        # (In practice with 1 shot, there should be only one outcome)
        if len(dist) == 1:
            bit_int = list(dist.keys())[0]
        else:
            # Take maximum probability outcome
            bit_int = max(dist, key=dist.get)

        # Convert to bitstring
        bitstring = format(bit_int, f"0{num_bits}b")

        # Reverse for consistent ordering (Qiskit uses little-endian)
        return bitstring[::-1]

    def get_backend_info(self) -> Dict[str, Any]:
        """
        Get information about the selected backend.

        Returns:
            Backend information dictionary
        """
        if not self.backend:
            self.backend = self.select_backend()

        return {
            "name": self.backend.name,
            "num_qubits": self.backend.num_qubits,
            "operational": getattr(self.backend, "operational", True),
            "simulator": getattr(self.backend, "simulator", False),
            "description": getattr(self.backend, "description", "N/A"),
        }

    def disconnect(self) -> None:
        """Disconnect from IBM Quantum service."""
        self.service = None
        self.backend = None
        logger.info("Disconnected from IBM Quantum service")


def execute_quantum_circuit(
    circuit: QuantumCircuit,
    config: Config,
    shots: int = 1,
) -> ExecutionResult:
    """
    Execute a quantum circuit on IBM Quantum.

    Args:
        circuit: Quantum circuit to execute
        config: Application configuration
        shots: Number of measurement shots

    Returns:
        Execution result with bitstring

    Raises:
        RuntimeError: If execution fails
    """
    manager = IBMRuntimeManager(config)

    try:
        manager.connect()
        result = manager.execute_circuit(circuit, shots=shots)
        return result

    finally:
        manager.disconnect()


def check_ibm_connection(config: Config) -> bool:
    """
    Check connection to IBM Quantum service.

    Args:
        config: Application configuration

    Returns:
        True if connection successful
    """
    try:
        manager = IBMRuntimeManager(config)
        manager.connect()
        backend_info = manager.get_backend_info()
        logger.info(f"Connection test successful. Backend: {backend_info['name']}")
        manager.disconnect()
        return True

    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False


def list_available_backends(config: Config) -> List[Dict[str, Any]]:
    """
    List available quantum backends.

    Args:
        config: Application configuration

    Returns:
        List of backend information dictionaries
    """
    manager = IBMRuntimeManager(config)
    manager.connect()

    try:
        backends = manager.service.backends()
        backend_list = []

        for backend in backends:
            backend_list.append(
                {
                    "name": backend.name,
                    "num_qubits": backend.num_qubits,
                    "operational": getattr(backend, "operational", True),
                    "simulator": getattr(backend, "simulator", False),
                }
            )

        return backend_list

    finally:
        manager.disconnect()
