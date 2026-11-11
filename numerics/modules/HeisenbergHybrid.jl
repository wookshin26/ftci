"""
HeisenbergHybrid.jl
===================
Hybrid oracle that automatically selects the fastest backend per chain length:

    N ≤ crossover  →  Yao.jl statevector  (O(2^N), faster for small N)
    N >  crossover →  ITensors MPS-TEBD   (O(N χ³), wins for large N)

Benchmark (K=4, t=1, MPS_MAXDIM=128) found crossover ≈ N=17–18.
Default `crossover = 17` gives Yao for N ≤ 17, MPS for N ≥ 18.

## Exports
- `make_heisenberg_oracle_auto` — factory returning `fq(J)` closure
- `CROSSOVER_N`                 — default crossover value
"""
module HeisenbergHybrid

export make_heisenberg_oracle_auto, CROSSOVER_N

const CROSSOVER_N = 17   # default: Yao ≤ 17, MPS > 17

"""
    make_heisenberg_oracle_auto(N; n_steps, t, maxdim, cutoff, crossover) -> Function

Hybrid oracle factory.  Automatically selects:
- Yao statevector for N ≤ `crossover`  (fast for small N, O(2^N) per call)
- MPS-TEBD for N > `crossover`          (efficient for large N, O(N·χ³) per call)

Returns a closure `fq(J)` where `J = [J1, J2, J3, J4]`.

# Arguments
- `N`         : chain length (number of qubits/spins)
- `n_steps`   : Trotter steps K
- `t`         : total evolution time (default 1.0)
- `maxdim`    : MPS bond-dim cap (only used when MPS backend is selected)
- `cutoff`    : MPS SVD cutoff (only used when MPS backend is selected)
- `crossover` : switch threshold; Yao for N ≤ crossover (default $(CROSSOVER_N))
"""
function make_heisenberg_oracle_auto(N::Int;
                                      n_steps::Int,
                                      t::Real      = 1.0,
                                      maxdim::Int  = 256,
                                      cutoff::Real = 1e-12,
                                      crossover::Int = CROSSOVER_N)
    if N <= crossover
        @info "Hybrid oracle → Yao  (N=$N ≤ crossover=$crossover, backend: statevector)"
        return Main.HeisenbergYao.make_heisenberg_oracle_yao(N;
                    n_steps = n_steps, t = t)
    else
        @info "Hybrid oracle → MPS  (N=$N > crossover=$crossover, backend: TEBD, maxdim=$maxdim)"
        return Main.HeisenbergMPS.make_heisenberg_oracle(N;
                    n_steps = n_steps, t = t,
                    maxdim  = maxdim,  cutoff = cutoff)
    end
end

end # module HeisenbergHybrid
