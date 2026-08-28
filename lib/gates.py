
from qiskit import QuantumCircuit
from qiskit.circuit import Gate


def big_endian_bit_str(n: int, x: int | str) -> str:
    if isinstance(x, int):
        return f"{x:>0{n}b}"
    else:
        return f"{x:>0{n}}"


class TranspositionGate(Gate):
    def __init__(self, num_qubits: int, a: str | int, b: str | int):
        a = big_endian_bit_str(num_qubits, a)
        b = big_endian_bit_str(num_qubits, b)

        assert a != b

        super().__init__(
            "TranspositionGate", num_qubits, [], label=f"$T^{{{a}}}_{{{b}}}$"
        )

        self.a = a
        self.b = b

    def _define(self):
        # Based on https://quantumcomputing.stackexchange.com/a/44536/42000
        self.definition = QuantumCircuit(self.num_qubits)

        if self.num_qubits == 1:
            self.definition.x(0)
            return

        target = 0
        a1 = False
        for i in range(self.num_qubits):
            bi = self.num_qubits - i - 1
            if self.a[bi] != self.b[bi]:
                target = i
                a1 = self.a[bi] == "1"
                break

        # Map b (or a if a1 == True) to a ^ (1 << target), without affecting a
        for i in range(target + 1, self.num_qubits):
            bi = self.num_qubits - i - 1
            if self.a[bi] != self.b[bi]:
                self.definition.cx(target, i)

        control = list(self.b if a1 else self.a)
        del control[self.num_qubits - target - 1]
        control = "".join(control)
        self.definition.mcx(
            [*range(target), *range(target + 1, self.num_qubits)],
            target,
            ctrl_state=control,
        )

        for i in reversed(range(target + 1, self.num_qubits)):
            bi = self.num_qubits - i - 1
            if self.a[bi] != self.b[bi]:
                self.definition.cx(target, i)


class IncrementGate(Gate):
    def __init__(self, num_qubits: int):
        super().__init__("IncrementGate", num_qubits, [], label="$+1$")

    def _define(self):
        self.definition = QuantumCircuit(self.num_qubits)

        for i in range(self.num_qubits - 1, 0, -1):
            self.definition.mcx(list(range(i)), i)

        self.definition.x(0)


def apply_on_buffer(circ: QuantumCircuit, b_reg, q_reg, op: Gate, buffer: int):
    circ.append(
        op.control(len(b_reg), ctrl_state=buffer, annotated=True),
        [*b_reg, *q_reg[: op.num_qubits]],
    )


def delay_gate(
    circ: QuantumCircuit,
    b_reg,
    q_reg,
    target_buffer: int,
    target_basis: str,
    *,
    delay: int,
    buffer: int,
):
    """Apply delay to the `target` basis state. State space stored in `buffer`."""

    state_q_needed = (delay - 1).bit_length()
    assert delay == 2**state_q_needed, "Only powers of two supported"

    target = target_basis + big_endian_bit_str(len(b_reg), target_buffer)
    total_q = max(len(target_basis), state_q_needed)
    tr_gate = TranspositionGate(len(b_reg) + total_q, target, buffer)
    circ.append(tr_gate, [*b_reg, *q_reg[:total_q]])
    if delay > 1:
        apply_on_buffer(circ, b_reg, q_reg, IncrementGate(state_q_needed), buffer)

def process_sample(
    circ_target: QuantumCircuit,
    b_reg,
    q_reg,
    circ_proc,
    *,
    sample_buffer: int,
    sample_pos: int,
    proc_buffer: int,
):
    """Process sample with given circuit `circ_proc`"""

    sample_basis = (
        big_endian_bit_str(len(q_reg), sample_pos) +
        big_endian_bit_str(len(b_reg), sample_buffer)
    )
    proc_io_basis = big_endian_bit_str(len(b_reg), proc_buffer)
    tr_gate = TranspositionGate(len(b_reg) + len(q_reg), sample_basis, proc_io_basis)
    circ_target.append(tr_gate, [*b_reg, *q_reg])
    circ_target.compose(circ_proc, inplace=True)
    circ_target.append(tr_gate, [*b_reg, *q_reg])