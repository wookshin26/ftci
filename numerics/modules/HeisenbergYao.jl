"""
HeisenbergYao.jl
================
Statevector oracle for the Heisenberg chain using Yao.jl.
Used only for benchmarking / correctness comparison against HeisenbergMPS.jl.

Hamiltonian and observable are identical to HeisenbergMPS.jl.
Scaling: O(2^N) memory and O(N · K · 2^N) time per call.
"""
module HeisenbergYao

using Yao, YaoBlocks
using LinearAlgebra

export make_heisenberg_oracle_yao,
       heisenberg_oracle_yao

# ── Constant 2-qubit Pauli-product gates ────────────────────────────────────
@const_gate XX2 = mat(kron(X, X))
@const_gate YY2 = mat(kron(Y, Y))
@const_gate ZZ2 = mat(kron(Z, Z))

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
Bond unitary: exp(-i·dt·(J1·XX + J2·YY + J3·ZZ)) on 2 qubits.
Yao's `rot(P, θ) = exp(-i·θ/2·P)`, so `rot(XX2, 2J·dt) = exp(-i·J·dt·XX)`.
"""
function _bond_block(dt, J1, J2, J3)
    return chain(rot(XX2, 2J1*dt),
                 rot(YY2, 2J2*dt),
                 rot(ZZ2, 2J3*dt))
end

"""
One Trotter layer: all even or odd bonds with the same (J1,J2,J3).
parity = :even → bonds (1,2),(3,4),…;  :odd → (2,3),(4,5),…
"""
function _bond_layer(N, dt, parity, J1, J2, J3)
    start = parity == :even ? 1 : 2
    blk   = _bond_block(dt, J1, J2, J3)
    return chain(N, (put(N, (i, i+1) => blk) for i in start:2:(N-1))...)
end

"""
Z-field layer: exp(-i·dt·J4·Z_i) = Rz(2·J4·dt) on every site.
Yao Rz(θ) = exp(-i·θ/2·Z), so Rz(2·J4·dt) = exp(-i·J4·dt·Z).
"""
function _field_layer(N, dt, J4)
    return chain(N, [put(N, i => Rz(2J4*dt)) for i in 1:N]...)
end

# ─────────────────────────────────────────────────────────────────────────────
# Factory  (recommended)
# ─────────────────────────────────────────────────────────────────────────────

"""
    make_heisenberg_oracle_yao(N; n_steps, t) -> Function

Factory that pre-builds the Néel-state circuit once and returns a closure
`fq(J)` where `J = [J1, J2, J3, J4]`.

For Yao, the main Trotter circuit must be rebuilt each call because its
gates depend on J.  But the initial state preparation (`neel_circ`) is
pre-built and reused.
"""
function make_heisenberg_oracle_yao(N::Int; n_steps::Int, t::Real = 1.0)
    dt        = t / n_steps
    neel_circ = repeat(N, X, 1:2:N)   # X on odd sites: |0…0⟩ → |↓↑↓↑…⟩

    return function(J::AbstractVector{<:Real})
        length(J) == 4 ||
            throw(ArgumentError("J must have length 4: [J1, J2, J3, J4]"))
        J1, J2, J3, J4 = J

        # Build Trotter circuit (depends on J — must rebuild each call)
        layers = Any[]
        for _ in 1:n_steps
            push!(layers, _bond_layer(N, dt, :even, J1, J2, J3))
            push!(layers, _bond_layer(N, dt, :odd,  J1, J2, J3))
            if J4 != 0.0
                push!(layers, _field_layer(N, dt, J4))
            end
        end
        trotter_circ = chain(N, layers...)

        # Prepare state and evolve
        reg = zero_state(N) |> neel_circ |> trotter_circ

        # Staggered magnetization O_Mst = (1/N) Σ_i (-1)^i ⟨Z_i⟩
        O_Mst = sum((-1)^i * real(expect(put(N, i => Z), reg)) for i in 1:N) / N
        return O_Mst
    end
end


# ─────────────────────────────────────────────────────────────────────────────
# Simple one-shot oracle
# ─────────────────────────────────────────────────────────────────────────────

"""
    heisenberg_oracle_yao(J; N, n_steps, t) -> Float64

One-shot Yao oracle.  Convenience wrapper around `make_heisenberg_oracle_yao`.
"""
function heisenberg_oracle_yao(J::AbstractVector{<:Real};
                                N::Int, n_steps::Int, t::Real = 1.0)
    fq = make_heisenberg_oracle_yao(N; n_steps=n_steps, t=t)
    return fq(J)
end

end # module HeisenbergYao
