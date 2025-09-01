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
            # Connect to IBM Quantum service
            if self.config.ibm_api_key:
                # Use API key from config
                self.service = QiskitRuntimeService(
                    channel="ibm_quantum_platform",
                    token=self.config.ibm_api_key,
                )
                logger.info("Connected to IBM Quantum service with API key")
            else:
                # Use saved credentials
                self.service = QiskitRuntimeService()
                logger.info("Connected to IBM Quantum service with saved credentials")

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
                backends = self.service.backends(simulator=True, operational=True)
                if not backends:
                    raise RuntimeError("No simulator backends available")
                backend = backends[0]
                logger.info(f"Using simulator backend: {backend.name}")

            elif self.options["backend"]:
                # Use specified backend
                backend = self.service.backend(self.options["backend"])
                logger.info(f"Using specified backend: {self.options['backend']}")

            else:
                # Auto-select hardware backend
                backends = self.service.backends(simulator=False, operational=True)
                
                if not backends:
                    raise RuntimeError("No quantum hardware backends available")
                
                # Sort by queue length (pending jobs)
                backend_status = []
                for b in backends:
                    status = b.status()
                    backend_status.append({
                        'backend': b,
                        'queue': status.pending_jobs if hasattr(status, 'pending_jobs') else 0
                    })
                
                # Sort by queue length
                backend_status.sort(key=lambda x: x['queue'])
                
                # Select backend with shortest queue
                backend = backend_status[0]['backend']

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
                with Session(backend=self.backend) as session:
                    # Create sampler with the session
                    sampler = Sampler(mode=session, options=sampler_options)
                    
                    # Transpile circuit for the backend
                    from qiskit import transpile
                    transpiled_circuit = transpile(
                        circuit,
                        self.backend,
                        optimization_level=3,
                        seed_transpiler=42
                    )
                    logger.info(f"Circuit transpiled: depth={transpiled_circuit.depth()}, gates={transpiled_circuit.size()}")

                    # Run the transpiled circuit
                    job = sampler.run([transpiled_circuit])
                    result = job.result()

                    # Extract bitstring from results
                    bitstring = self._extract_bitstring(result, transpiled_circuit.num_clbits)

                    # Create execution result
                    execution_result = ExecutionResult(
                        bitstring=bitstring,
                        backend_name=self.backend.name,
                        job_id=getattr(job, "job_id", None),
                        shots=shots,
                        quasi_probabilities=None,  # Removed quasi_dists which doesn't exist in v2
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
        # Get measurement data from PrimitiveResult
        pub_result = result[0]
        data = pub_result.data
        
        # Extract counts - try different methods based on backend
        counts = None
        
        # Try method 1: meas attribute
        if hasattr(data, 'meas'):
            counts = data.meas.get_counts()
        # Try method 2: check for classical register by name
        elif hasattr(data, 'c'):
            counts = data.c.get_counts()
        # Try method 3: data might have get_counts
        elif hasattr(data, 'get_counts'):
            counts = data.get_counts()
        # Try method 4: Look for any BitArray attribute
        else:
            # Find the measurement data - it's the first BitArray attribute
            for attr_name in dir(data):
                if not attr_name.startswith('_'):
                    attr = getattr(data, attr_name)
                    if hasattr(attr, 'get_counts'):
                        counts = attr.get_counts()
                        break
        
        if counts is None:
            raise RuntimeError(f"Could not extract counts from result. Data type: {type(data)}, attributes: {[a for a in dir(data) if not a.startswith('_')]}")
        
        # For single shot, there should be only one outcome
        # Otherwise take the most frequent
        if len(counts) == 1:
            bitstring = list(counts.keys())[0]
        else:
            # Take maximum count outcome
            bitstring = max(counts, key=counts.get)
        
        # Ensure correct length (pad with zeros if needed)
        if len(bitstring) < num_bits:
            bitstring = '0' * (num_bits - len(bitstring)) + bitstring
        elif len(bitstring) > num_bits:
            bitstring = bitstring[:num_bits]
        
        return bitstring

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
