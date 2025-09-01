# src/chsh_weirdness.py
"""
CHSH Bell inequality test to measure quantum "weirdness" and use it to crank
surrealism / metaphor density when the score is high.
"""

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import Sampler
import numpy as np

def chsh_score(session, shots=256):
    """
    Simple CHSH with measurement settings (A0,A1,B0,B1)
    Return an empirical S in [~1.6 .. ~2.6] on real hardware (rarely perfect).
    """
    def bell(angle_a, angle_b):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.ry(angle_a, 0)
        qc.ry(angle_b, 1)
        qc.measure([0, 1], [0, 1])
        return qc
    
    # angles for A0,A1,B0,B1
    A0, A1 = 0.0, np.pi/4
    B0, B1 = np.pi/8, -np.pi/8
    settings = [(A0, B0), (A0, B1), (A1, B0), (A1, B1)]
    sampler = Sampler(session=session)
    
    # Run all Bell circuits
    circuits = [bell(a, b) for a, b in settings]
    job = sampler.run(circuits, shots=shots)
    result = job.result()
    
    # Get quasi distributions
    dists = [result.quasi_dists[i] for i in range(len(circuits))]
    
    def corr(dist):
        # Expectation of Z⊗Z from bitstring distribution
        e = 0
        for k, v in dist.items():
            b = bin(k)[2:].zfill(2)[::-1]
            z = (1 if b[0] == '0' else -1) * (1 if b[1] == '0' else -1)
            e += z * v
        return e
    
    E = [corr(d) for d in dists]
    S = abs(E[0] - E[1] + E[2] + E[3])
    return S

def weirdness_from_chsh(S):
    """
    Map S to a 0..1 weirdness knob
    S ~1.8 -> 0, S ~2.4 -> 1
    """
    return max(0.0, min(1.0, (S - 1.8) / 0.6))