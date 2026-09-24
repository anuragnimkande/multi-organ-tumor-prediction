"""
Quantum Classification Circuit — PennyLane 0.44+
=================================================
4-qubit variational quantum circuit for binary brain tumor classification.

Circuit design (v2 — improved):
    1. AngleEmbedding (RY) — encodes 4 classical features as qubit rotation angles
    2. StronglyEntanglingLayers (high-expressibility parameterised ansatz, 3 layers)
       - Each layer: Rot(phi,theta,omega) per qubit + multi-range CNOT entanglement
       - Superior to simple RX/RY/RZ + CNOT-ring in expressibility & gradient flow
    3. Measurement: PauliZ expectation values for all 4 qubits -> (4,) output

Key improvements over v1:
    - StronglyEntanglingLayers vs custom loop ansatz -> better representational power
    - 3 variational layers (was 2) -> deeper circuit, more expressibility
    - lightning.qubit device (C++ acceleration) when available

NOTE on double embedding: PennyLane 0.44 TorchLayer executes the circuit with
the full batch tensor (batch, n_feats) where qml.AngleEmbedding broadcasts
correctly over the batch. Per-qubit indexing (inputs[q]) references batch rows,
not feature columns, so we rely on AngleEmbedding for correct batch semantics.

Wrapped as a qml.qnn.TorchLayer for seamless PyTorch integration.
"""

import pennylane as qml
import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────────────────────
# Device configuration
# ──────────────────────────────────────────────────────────────────────────────

N_QUBITS      = 4    # 4-qubit system
N_LAYERS      = 3    # 3 variational layers (was 2)
N_INPUT_FEATS = 4    # 4 classical features (matches N_QUBITS for AngleEmbedding)

# Prefer lightning.qubit (C++ accelerated) for faster simulation
try:
    dev = qml.device("lightning.qubit", wires=N_QUBITS)
    _DEVICE_NAME = "lightning.qubit"
except Exception:
    dev = qml.device("default.qubit", wires=N_QUBITS)
    _DEVICE_NAME = "default.qubit"


# ──────────────────────────────────────────────────────────────────────────────
# Quantum circuit definition
# ──────────────────────────────────────────────────────────────────────────────

@qml.qnode(dev, interface="torch", diff_method="best")
def quantum_circuit(inputs: torch.Tensor, weights: torch.Tensor):
    """
    Variational quantum circuit with angle embedding and strongly entangling layers.

    Args:
        inputs:  (4,) or (batch, 4)   -- classical feature vector(s)
        weights: (n_layers, 4, 3)     -- StronglyEntanglingLayers parameters

    Returns:
        List[float] of length 4 -- PauliZ expectation values in [-1, 1]
    """
    # -- Feature Encoding ---------------------------------------------------
    # AngleEmbedding maps each feature to an RY rotation on its qubit.
    # Operates correctly with PennyLane 0.44 batch semantics.
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")

    # -- High-Expressibility Variational Ansatz -----------------------------
    # StronglyEntanglingLayers: full Rot(phi,theta,omega) per qubit
    # + long-range CNOT entanglement (range scales with layer depth).
    # This achieves higher expressibility than the simple CNOT-ring + RX/RY/RZ.
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))

    # -- Measurement --------------------------------------------------------
    return [qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS)]


# ──────────────────────────────────────────────────────────────────────────────
# TorchLayer factory
# ──────────────────────────────────────────────────────────────────────────────

def build_quantum_layer(n_layers: int = N_LAYERS) -> qml.qnn.TorchLayer:
    """
    Create a PennyLane TorchLayer wrapping the quantum circuit.

    The returned module is a standard nn.Module-compatible layer:
        Input:  (batch, 4)   -- classical features in [-1, 1]
        Output: (batch, 4)   -- PauliZ expectation values

    Args:
        n_layers: Number of variational layers (3 recommended)

    Returns:
        qml.qnn.TorchLayer
    """
    weight_shapes = {
        # StronglyEntanglingLayers shape: (n_layers, n_wires, 3)
        "weights": (n_layers, N_QUBITS, 3)
    }
    return qml.qnn.TorchLayer(quantum_circuit, weight_shapes)


# ──────────────────────────────────────────────────────────────────────────────
# Circuit metadata
# ──────────────────────────────────────────────────────────────────────────────

def get_circuit_info() -> dict:
    """Return metadata about the quantum circuit for logging and display."""
    return {
        "n_qubits":         N_QUBITS,
        "n_layers":         N_LAYERS,
        "device":           _DEVICE_NAME,
        "encoding":         "AngleEmbedding (RY)",
        "ansatz":           "StronglyEntanglingLayers (3 layers)",
        "gates_per_layer":  "Rot(phi,theta,omega) + multi-range CNOT entanglement",
        "measurement":      "PauliZ expectation values",
        "trainable_params": N_LAYERS * N_QUBITS * 3,
        "input_features":   N_INPUT_FEATS,
    }


def draw_circuit():
    """Print a text diagram of the quantum circuit (for debugging)."""
    import torch as _torch
    sample_inputs  = _torch.zeros(N_INPUT_FEATS)
    sample_weights = _torch.zeros(N_LAYERS, N_QUBITS, 3)
    print(qml.draw(quantum_circuit)(sample_inputs, sample_weights))
