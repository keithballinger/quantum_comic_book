"""
Tests for IBM Quantum Runtime integration.
"""

from unittest.mock import MagicMock, Mock, patch, PropertyMock
from dataclasses import dataclass

import pytest
from qiskit import QuantumCircuit

from src.config import Config
from src.ibm_runtime import (
    IBMRuntimeManager,
    ExecutionResult,
    RuntimeError,
    execute_quantum_circuit,
    check_ibm_connection,
    list_available_backends,
)


class TestIBMRuntimeManager:
    """Test IBM Runtime manager functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(
            ibm_api_key="test_ibm_key",
            gemini_api_key="test_gemini_key",
            panels=3,
            characters=2,
            ibm_backend=None,
            use_simulator=False,
        )

    @pytest.fixture
    def manager(self, config):
        """Create runtime manager."""
        return IBMRuntimeManager(config)

    def test_manager_initialization(self, manager):
        """Test runtime manager initialization."""
        assert manager.service is None
        assert manager.backend is None
        assert manager.config.panels == 3

    @patch("src.ibm_runtime.QiskitRuntimeService")
    def test_connect_success(self, mock_service_class, manager):
        """Test successful connection to IBM Quantum."""
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service

        manager.connect()

        assert manager.service == mock_service
        mock_service_class.assert_called_once_with(
            channel="ibm_quantum",
            token="test_ibm_key",
        )

    @patch("src.ibm_runtime.QiskitRuntimeService")
    def test_connect_failure(self, mock_service_class, manager):
        """Test connection failure handling."""
        mock_service_class.side_effect = Exception("Connection failed")

        with pytest.raises(RuntimeError, match="Failed to connect"):
            manager.connect()

    @patch("src.ibm_runtime.QiskitRuntimeService")
    def test_select_backend_simulator(self, mock_service_class, config):
        """Test backend selection for simulator."""
        config.use_simulator = True
        manager = IBMRuntimeManager(config)

        mock_service = MagicMock()
        mock_backend = MagicMock()
        mock_backend.name = "ibmq_qasm_simulator"
        mock_service.backend.return_value = mock_backend
        manager.service = mock_service

        backend = manager.select_backend()

        assert backend == mock_backend
        mock_service.backend.assert_called_once_with("ibmq_qasm_simulator")

    @patch("src.ibm_runtime.QiskitRuntimeService")
    def test_select_backend_specified(self, mock_service_class, config):
        """Test backend selection with specified backend."""
        config.ibm_backend = "ibm_brisbane"
        manager = IBMRuntimeManager(config)

        mock_service = MagicMock()
        mock_backend = MagicMock()
        mock_backend.name = "ibm_brisbane"
        mock_service.backend.return_value = mock_backend
        manager.service = mock_service

        backend = manager.select_backend()

        assert backend == mock_backend
        mock_service.backend.assert_called_once_with("ibm_brisbane")

    @patch("src.ibm_runtime.QiskitRuntimeService")
    def test_select_backend_auto(self, mock_service_class, manager):
        """Test automatic backend selection."""
        mock_service = MagicMock()
        mock_backend = MagicMock()
        mock_backend.name = "ibm_kyoto"
        mock_service.least_busy.return_value = mock_backend
        manager.service = mock_service

        backend = manager.select_backend()

        assert backend == mock_backend
        mock_service.least_busy.assert_called_once()

    @patch("src.ibm_runtime.QiskitRuntimeService")
    def test_select_backend_fallback(self, mock_service_class, manager):
        """Test backend selection fallback when least_busy returns None."""
        mock_service = MagicMock()
        mock_backend = MagicMock()
        mock_backend.name = "ibm_torino"

        # least_busy returns None
        mock_service.least_busy.return_value = None
        # backends returns a list
        mock_service.backends.return_value = [mock_backend]
        manager.service = mock_service

        backend = manager.select_backend()

        assert backend == mock_backend
        mock_service.backends.assert_called_once()

    @patch("src.ibm_runtime.QiskitRuntimeService")
    def test_select_backend_no_backends(self, mock_service_class, manager):
        """Test error when no backends available."""
        mock_service = MagicMock()
        mock_service.least_busy.return_value = None
        mock_service.backends.return_value = []
        manager.service = mock_service

        with pytest.raises(RuntimeError, match="No suitable quantum backends"):
            manager.select_backend()

    def test_extract_bitstring_single_outcome(self, manager):
        """Test bitstring extraction with single outcome."""
        # Create mock result
        mock_result = MagicMock()
        mock_result.quasi_dists = [{5: 1.0}]  # Binary: 101

        bitstring = manager._extract_bitstring(mock_result, 3)

        # Should reverse for consistent ordering
        assert bitstring == "101"

    def test_extract_bitstring_multiple_outcomes(self, manager):
        """Test bitstring extraction with multiple outcomes."""
        # Create mock result with multiple outcomes
        mock_result = MagicMock()
        mock_result.quasi_dists = [
            {
                5: 0.3,  # Binary: 101
                7: 0.7,  # Binary: 111 (highest probability)
                2: 0.0,  # Binary: 010
            }
        ]

        bitstring = manager._extract_bitstring(mock_result, 3)

        # Should select highest probability (7 = 111)
        assert bitstring == "111"

    def test_extract_bitstring_no_results(self, manager):
        """Test error handling when no results returned."""
        mock_result = MagicMock()
        mock_result.quasi_dists = []

        with pytest.raises(RuntimeError, match="No results returned"):
            manager._extract_bitstring(mock_result, 3)

    @patch("src.ibm_runtime.Session")
    @patch("src.ibm_runtime.Sampler")
    def test_execute_circuit_success(
        self, mock_sampler_class, mock_session_class, manager
    ):
        """Test successful circuit execution."""
        # Setup mocks
        mock_backend = MagicMock()
        mock_backend.name = "test_backend"
        manager.backend = mock_backend
        manager.service = MagicMock()

        mock_session = MagicMock()
        mock_session_class.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_class.return_value.__exit__ = MagicMock(return_value=None)

        mock_sampler = MagicMock()
        mock_job = MagicMock()
        mock_job.job_id = "test_job_123"
        mock_result = MagicMock()
        mock_result.quasi_dists = [{15: 1.0}]  # Binary: 1111
        mock_job.result.return_value = mock_result
        mock_sampler.run.return_value = mock_job
        mock_sampler_class.return_value = mock_sampler

        # Create test circuit
        circuit = QuantumCircuit(4, 4)

        # Execute
        result = manager.execute_circuit(circuit, shots=1)

        assert isinstance(result, ExecutionResult)
        assert result.bitstring == "1111"
        assert result.backend_name == "test_backend"
        assert result.job_id == "test_job_123"
        assert result.shots == 1

    @patch("src.ibm_runtime.Session")
    @patch("src.ibm_runtime.Sampler")
    def test_execute_circuit_with_retry(
        self, mock_sampler_class, mock_session_class, manager
    ):
        """Test circuit execution with retry on failure."""
        manager.service = MagicMock()
        manager.config.max_retries = 3

        # Create mock backends
        mock_backend1 = MagicMock()
        mock_backend1.name = "backend1"
        mock_backend2 = MagicMock()
        mock_backend2.name = "backend2"

        # First attempt fails, second succeeds
        mock_session = MagicMock()
        mock_session_class.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_class.return_value.__exit__ = MagicMock(return_value=None)

        mock_sampler = MagicMock()
        mock_job = MagicMock()
        mock_result = MagicMock()
        mock_result.quasi_dists = [{7: 1.0}]

        # First call raises error, second succeeds
        from qiskit_ibm_runtime.exceptions import IBMRuntimeError

        mock_job.result.side_effect = [
            IBMRuntimeError("Temporary failure"),
            mock_result,
        ]
        mock_sampler.run.return_value = mock_job
        mock_sampler_class.return_value = mock_sampler

        # Setup backend selection
        manager.backend = mock_backend1
        with patch.object(manager, "select_backend", return_value=mock_backend2):
            circuit = QuantumCircuit(3, 3)
            result = manager.execute_circuit(circuit, shots=1, retry_on_error=True)

        assert result.bitstring == "111"
        assert mock_job.result.call_count == 2

    def test_get_backend_info(self, manager):
        """Test getting backend information."""
        mock_backend = MagicMock()
        mock_backend.name = "test_backend"
        mock_backend.num_qubits = 127
        mock_backend.operational = True
        mock_backend.simulator = False
        mock_backend.description = "Test backend"

        manager.backend = mock_backend

        info = manager.get_backend_info()

        assert info["name"] == "test_backend"
        assert info["num_qubits"] == 127
        assert info["operational"] is True
        assert info["simulator"] is False
        assert info["description"] == "Test backend"

    def test_disconnect(self, manager):
        """Test disconnection from service."""
        manager.service = MagicMock()
        manager.backend = MagicMock()

        manager.disconnect()

        assert manager.service is None
        assert manager.backend is None


class TestExecuteQuantumCircuit:
    """Test circuit execution helper function."""

    @patch("src.ibm_runtime.IBMRuntimeManager")
    def test_execute_quantum_circuit(self, mock_manager_class):
        """Test execute_quantum_circuit helper function."""
        # Setup mock
        mock_manager = MagicMock()
        mock_result = ExecutionResult(
            bitstring="101010",
            backend_name="test_backend",
            shots=1,
        )
        mock_manager.execute_circuit.return_value = mock_result
        mock_manager_class.return_value = mock_manager

        # Create test inputs
        config = Config(
            ibm_api_key="test_key",
            gemini_api_key="test_key",
        )
        circuit = QuantumCircuit(6, 6)

        # Execute
        result = execute_quantum_circuit(circuit, config, shots=1)

        assert result.bitstring == "101010"
        assert result.backend_name == "test_backend"
        mock_manager.connect.assert_called_once()
        mock_manager.disconnect.assert_called_once()


class TestTestConnection:
    """Test connection testing function."""

    @patch("src.ibm_runtime.IBMRuntimeManager")
    def test_connection_success(self, mock_manager_class):
        """Test successful connection test."""
        mock_manager = MagicMock()
        mock_manager.get_backend_info.return_value = {"name": "test_backend"}
        mock_manager_class.return_value = mock_manager

        config = Config(
            ibm_api_key="test_key",
            gemini_api_key="test_key",
        )

        result = check_ibm_connection(config)

        assert result is True
        mock_manager.connect.assert_called_once()
        mock_manager.disconnect.assert_called_once()

    @patch("src.ibm_runtime.IBMRuntimeManager")
    def test_connection_failure(self, mock_manager_class):
        """Test connection test failure."""
        mock_manager = MagicMock()
        mock_manager.connect.side_effect = Exception("Connection failed")
        mock_manager_class.return_value = mock_manager

        config = Config(
            ibm_api_key="test_key",
            gemini_api_key="test_key",
        )

        result = check_ibm_connection(config)

        assert result is False


class TestListAvailableBackends:
    """Test backend listing function."""

    @patch("src.ibm_runtime.IBMRuntimeManager")
    def test_list_available_backends(self, mock_manager_class):
        """Test listing available backends."""
        # Create mock backends
        mock_backend1 = MagicMock()
        mock_backend1.name = "backend1"
        mock_backend1.num_qubits = 127
        mock_backend1.operational = True
        mock_backend1.simulator = False

        mock_backend2 = MagicMock()
        mock_backend2.name = "simulator"
        mock_backend2.num_qubits = 32
        mock_backend2.operational = True
        mock_backend2.simulator = True

        mock_manager = MagicMock()
        mock_service = MagicMock()
        mock_service.backends.return_value = [mock_backend1, mock_backend2]
        mock_manager.service = mock_service
        mock_manager_class.return_value = mock_manager

        config = Config(
            ibm_api_key="test_key",
            gemini_api_key="test_key",
        )

        backends = list_available_backends(config)

        assert len(backends) == 2
        assert backends[0]["name"] == "backend1"
        assert backends[0]["num_qubits"] == 127
        assert backends[1]["name"] == "simulator"
        assert backends[1]["simulator"] is True

        mock_manager.connect.assert_called_once()
        mock_manager.disconnect.assert_called_once()


@pytest.mark.parametrize(
    "shots,expected_shots",
    [
        (1, 1),  # Single shot for pure collapse
        (10, 10),  # Multiple shots for statistics
        (1024, 1024),  # Standard shot count
    ],
)
def test_execution_result_shots(shots, expected_shots):
    """Test ExecutionResult with different shot counts."""
    result = ExecutionResult(
        bitstring="101",
        backend_name="test",
        shots=shots,
    )

    assert result.shots == expected_shots
