"""
HeisenbergMPS.jl
================
MPS-based oracle for the Heisenberg chain Hamiltonian dynamics experiment
(Chapter VII.D of the manuscript).

Hamiltonian:
    H(J1,J2,J3,J4) = Σ_{i=1}^{N-1} (J1 X_i X_{i+1} + J2 Y_i Y_{i+1} + J3 Z_i Z_{i+1})
                   + Σ_{i=1}^N J4 Z_i

where X, Y, Z are Pauli matrices (eigenvalues ±1).

Circuit:  K-step first-order Trotter from Néel initial state |↓↑↓↑…⟩.
Observable: staggered magnetization O_Mst = (1/N) Σ_i (-1)^i ⟨Z_i⟩.

## Key functions

- `make_heisenberg_oracle(N; n_steps, ...)` — **recommended**
    Factory that pre-computes all J-independent objects (sites, initial state,
    operator tensors) once, then returns a fast closure `fq(J)`.
    Use this when calling the oracle many times (e.g., inside TCI).

- `heisenberg_oracle_mps(J; N, n_steps, ...)` — simple one-shot call.
    Recreates everything on every call; fine for a few evaluations or testing.

## Conventions (ITensors vs Yao)

    Quantity          Yao                   ITensors
    ─────────────────────────────────────────────────────────
    Spin-Z operator   Z = σz  (±1)          Sz = σz/2  (±1/2)
    ⟨Z_i⟩            direct                 2 × ⟨Sz_i⟩
    XX product        X⊗X = σx⊗σx          4 × Sx⊗Sx
    YY product        Y⊗Y = σy⊗σy          4 × Sy⊗Sy
    ZZ product        Z⊗Z = σz⊗σz          4 × Sz⊗Sz
    Field J4·Z        J4·σz                 2J4·Sz
    Néel state        X on odd sites of|0⟩  "Dn","Up","Dn","Up",…

## Scaling

    Method            Memory       Time per call
    ──────────────────────────────────────────────────────
    Statevector (Yao) O(2^N)       O(N · K · 2^N)
    MPS (this file)   O(N · χ²)    O(N · K · χ^3)
    Crossover: MPS wins roughly for N ≳ 15 (with χ ~ 50, K=4, t=1).
"""
module HeisenbergMPS

using ITensors, ITensorMPS
using LinearAlgebra

export neel_state_mps,
       staggered_magnetization,
       make_heisenberg_oracle,
       heisenberg_oracle_mps

# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    neel_state_mps(sites) -> MPS

Create the Néel state |↓↑↓↑…⟩ as a product MPS (bond dim = 1).
Matches the Yao initial state (X applied to odd sites of |00…0⟩).
"""
function neel_state_mps(sites)
    N     = length(sites)
    state = [isodd(i) ? "Dn" : "Up" for i in 1:N]
    return MPS(sites, state)
end


"""
    staggered_magnetization(psi) -> Float64

Compute O_Mst = (1/N) Σ_{i=1}^N (-1)^i ⟨Z_i⟩  using ⟨Z_i⟩ = 2⟨Sz_i⟩.
"""
function staggered_magnetization(psi::MPS)
    N       = length(psi)
    sz_vals = real.(expect(psi, "Sz"))   # Vector{Float64} length N
    return sum((-1)^i * 2.0 * sz_vals[i] for i in 1:N) / N
end


# ─────────────────────────────────────────────────────────────────────────────
# Optimised factory  (recommended for TCI usage)
# ─────────────────────────────────────────────────────────────────────────────

"""
    make_heisenberg_oracle(N; n_steps, t, maxdim, cutoff) -> Function

**Factory** that pre-computes all J-independent objects once:
  - site indices
  - initial Néel MPS
  - bond operator products (SxSx, SySy, SzSz for each bond)
  - single-site Sz for each site

Returns a closure `fq(J)` where `J = [J1, J2, J3, J4]` that reuses these
on every call, rebuilding only the cheap exponentials `exp(-i·dt·hj(J))`.

This avoids the per-call overhead of `siteinds`, `op(…)`, MPS construction,
etc., which would otherwise dominate for millions of TCI queries.

# Arguments
- `N`:       chain length
- `n_steps`: Trotter steps K
- `t`:       evolution time (default 1.0)
- `maxdim`:  MPS bond-dimension cap (default 256)
- `cutoff`:  SVD cutoff for MPS truncation (default 1e-12)
"""
function make_heisenberg_oracle(N::Int;
                                n_steps::Int,
                                t::Real     = 1.0,
                                maxdim::Int = 256,
                                cutoff::Real = 1e-12)

    # ── Pre-compute once ────────────────────────────────────────────────────
    sites = siteinds("S=1/2", N; conserve_qns=false)
    psi0  = neel_state_mps(sites)
    dt    = t / n_steps

    # Bond operator products (do NOT depend on J)
    # Factor 4: σα = 2·Sα  →  σα⊗σα = 4·Sα⊗Sα
    SxSx = [op("Sx", sites[i]) * op("Sx", sites[i+1]) for i in 1:(N-1)]
    SySy = [op("Sy", sites[i]) * op("Sy", sites[i+1]) for i in 1:(N-1)]
    SzSz = [op("Sz", sites[i]) * op("Sz", sites[i+1]) for i in 1:(N-1)]

    # Single-site Sz (do NOT depend on J)
    # Factor 2: Z = σz = 2·Sz
    Sz_s = [op("Sz", sites[i]) for i in 1:N]

    # ── Inner helper: build gates for one Trotter step (depends on J) ───────
    function _step_gates(J1, J2, J3, J4)
        gates = ITensor[]
        # Even bonds: (1,2), (3,4), ...
        for i in 1:2:(N-1)
            hj = 4J1 * SxSx[i] + 4J2 * SySy[i] + 4J3 * SzSz[i]
            push!(gates, exp(-im * dt * hj))
        end
        # Odd bonds: (2,3), (4,5), ...
        for i in 2:2:(N-1)
            hj = 4J1 * SxSx[i] + 4J2 * SySy[i] + 4J3 * SzSz[i]
            push!(gates, exp(-im * dt * hj))
        end
        # Single-site Z-field gates
        if J4 != 0.0
            for i in 1:N
                hi = 2J4 * Sz_s[i]
                push!(gates, exp(-im * dt * hi))
            end
        end
        return gates
    end

    # ── Return the closure (J → scalar) ─────────────────────────────────────
    return function(J::AbstractVector{<:Real})
        length(J) == 4 ||
            throw(ArgumentError("J must have length 4: [J1, J2, J3, J4]"))
        J1, J2, J3, J4 = J

        psi = copy(psi0)    # cheap: copies N small tensors, no SVD
        for _ in 1:n_steps
            gates = _step_gates(J1, J2, J3, J4)
            psi   = apply(gates, psi; maxdim=maxdim, cutoff=cutoff)
            normalize!(psi)
        end
        return staggered_magnetization(psi)
    end
end


# ─────────────────────────────────────────────────────────────────────────────
# Simple one-shot oracle  (convenient for testing / few evaluations)
# ─────────────────────────────────────────────────────────────────────────────

"""
    heisenberg_oracle_mps(J; N, n_steps, t, maxdim, cutoff) -> Float64

One-shot MPS oracle.  Recreates sites/state/operators on every call.
**Prefer `make_heisenberg_oracle` for TCI usage** (many calls).

# Arguments
- `J`: `[J1, J2, J3, J4]`
- `N`, `n_steps`, `t`, `maxdim`, `cutoff`: as in `make_heisenberg_oracle`.
"""
function heisenberg_oracle_mps(J::AbstractVector{<:Real};
                               N::Int,
                               n_steps::Int,
                               t::Real     = 1.0,
                               maxdim::Int = 256,
                               cutoff::Real = 1e-12)
    fq = make_heisenberg_oracle(N; n_steps=n_steps, t=t,
                                   maxdim=maxdim, cutoff=cutoff)
    return fq(J)
end

end # module HeisenbergMPS
