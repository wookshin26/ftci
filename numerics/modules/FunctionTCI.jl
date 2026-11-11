"""
FunctionTCI.jl
==============
Core algebra for the Fourier-TCI (FTCI) algorithm from the manuscript.

This is the cleaned-up, standalone version of `FunctionTCI` from `Modules.jl`.
It contains everything needed to:
  1. Sample a target function on a Fourier grid (via TCI).
  2. Convert the TT(F) cores to TT(C) (Fourier coefficient tensor train).
  3. Evaluate the approximation at arbitrary continuous inputs.

Public API (via `using .FunctionTCI`):
  - `make_basis_vectors`     – build local Fourier basis functions
  - `gamma_vals`             – compute sampling domain widths
  - `ttF_to_ttC`             – transform TT(F) → TT(C) cores
  - `eval_from_ttC`          – evaluate TT(C) at a continuous point
  - `dense_from_tt`          – materialise the full tensor from TT cores (debug/validation)
  - `CounterWithLog`         – query counter with caching for TCI
  - `counted_unique_with_log`– wrap a function with the above counter
"""
module FunctionTCI

using LinearAlgebra, Random

export make_basis_vectors,
       gamma_vals,
       ttF_to_ttC,
       eval_from_ttC,
       dense_from_tt,
       relerr_grid_2norm,
       CounterWithLog,
       counted_unique_with_log

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Query counter (caching wrapper for the TCI oracle)
# ─────────────────────────────────────────────────────────────────────────────

"""
    CounterWithLog

Mutable struct that wraps a function to:
  - cache results (avoid re-evaluating the same index),
  - count only *new* (cache-miss) evaluations,
  - optionally log every called index.

# Fields
- `n::Int`                          – number of unique function evaluations so far
- `cache::Dict{Tuple,Float64}`      – index → value cache
- `logged_indices::Vector{Vector{Int}}` – all queried indices (in order)
- `verbose::Bool`                   – print each index to stdout if true
"""
mutable struct CounterWithLog
    n::Int
    cache::Dict{Tuple{Vararg{Int}}, Float64}
    logged_indices::Vector{Vector{Int}}
    verbose::Bool
end

"""
    CounterWithLog(; verbose=false) -> CounterWithLog

Construct an empty counter.
"""
CounterWithLog(; verbose::Bool=false) =
    CounterWithLog(0,
                   Dict{Tuple{Vararg{Int}}, Float64}(),
                   Vector{Vector{Int}}(),
                   verbose)

"""
    counted_unique_with_log(f, ctr::CounterWithLog) -> Function

Return a wrapped version of `f(v::Vector{Int})` that:
  - records `v` in `ctr.logged_indices`,
  - returns the cached value if seen before (no increment),
  - evaluates `f(v)`, caches, and increments `ctr.n` on a cache miss.
"""
function counted_unique_with_log(f, ctr::CounterWithLog)
    return function(v::AbstractVector{<:Integer})
        key = Tuple(v)
        push!(ctr.logged_indices, collect(Int, v))
        ctr.verbose && println("TCI called index: ", v)

        if haskey(ctr.cache, key)
            return ctr.cache[key]
        end
        val = f(v)
        ctr.cache[key] = val
        ctr.n += 1
        return val
    end
end


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Frequency / domain helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    gamma_vals(ωss) -> Vector{Float64}

For each frequency set ωs = ωss[k], compute
    γ_k = min(min_j |ω_j|,  min_{j≠l} |ω_j − ω_l|,  min_{j≠l} |ω_j + ω_l|)

which determines the appropriate sampling domain width (Nyquist-like condition).
"""
function gamma_vals(ωss)
    gammas = Float64[]
    for ωs in ωss
        ω = sort(abs.(ωs))
        n = length(ω)
        min1 = minimum(ω)
        min2 = Inf
        min3 = Inf
        for j in 1:n, l in 1:n
            if j != l
                min2 = min(min2, abs(ω[j] - ω[l]))
                min3 = min(min3, abs(ω[j] + ω[l]))
            end
        end
        γ = min(min1, min2, min3)
        γ <= 0 && error("γ ≤ 0 at frequency set $ωs — check that frequencies are nonzero and non-degenerate.")
        push!(gammas, γ)
    end
    return gammas
end


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Local Fourier basis vectors
# ─────────────────────────────────────────────────────────────────────────────

"""
    make_basis_vectors(ωss, phis) -> Vector{Function}

Build a list of local basis functions, one per TCI site k:

    t[k](x) = (1/√(D_k+1)) · [1, cos(ω₁·ϕ_k(x)), sin(ω₁·ϕ_k(x)), …,
                                    cos(ω_{D_k}·ϕ_k(x)), sin(ω_{D_k}·ϕ_k(x))]

where ωss[k] = [ω₁, …, ω_{D_k}] and phis[k] is the pre-processing function ϕ_k.

# Arguments
- `ωss`:  Vector of frequency vectors, length L.
- `phis`: Vector of functions ϕ_k : x → scalar, length L.
          For the Hamiltonian experiment use `phis = [J -> J[k] for k in 1:L]`.

# Returns
Vector of L functions; each takes the full input vector x and returns a
(2D_k+1)-dimensional real vector.
"""
function make_basis_vectors(ωss::Vector{<:AbstractVector}, phis::AbstractVector)
    length(ωss) == length(phis) ||
        throw(ArgumentError("ωss and phis must have the same length"))
    N = length(ωss)

    function single_basis(ωs::AbstractVector{<:Real}, fphi::Function)
        D = length(ωs)
        norm = 1.0 / sqrt(D + 1)
        return x -> begin
            v = Vector{Float64}(undef, 2D + 1)
            v[1] = norm
            phi_val = fphi(x)
            @inbounds for (r, ω) in enumerate(ωs)
                v[2r]   = norm * cos(ω * phi_val)
                v[2r+1] = norm * sin(ω * phi_val)
            end
            return v
        end
    end

    return [single_basis(ωss[p], phis[p]) for p in 1:N]
end


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: TT core manipulation
# ─────────────────────────────────────────────────────────────────────────────

"""
    _build_local_V(ωss, auxs) -> (V_list, V_pinv_list)

Internal helper: for each site k, build the sampling matrix V^{(k)} whose
columns are the local basis vectors evaluated at the auxiliary inputs auxs[k].

- `V^{(k)}` has shape `(2D_k+1, M_k)` where M_k = length(auxs[k]).
- Returns `V_list` and `V_pinv_list` (Moore-Penrose pseudoinverses).
"""
function _build_local_V(ωss::Vector{<:AbstractVector{<:Real}},
                        auxs::Vector{<:AbstractVector{<:Real}})
    length(ωss) == length(auxs) ||
        throw(ArgumentError("ωss and auxs must have the same length"))

    L  = length(ωss)
    Ds = [length(ωs) for ωs in ωss]
    Ms = [length(a) for a in auxs]

    V_list     = Vector{Matrix{Float64}}(undef, L)
    V_pinv_list = Vector{Matrix{Float64}}(undef, L)

    for k in 1:L
        D_k  = Ds[k]
        M_k  = Ms[k]
        ωs_k = ωss[k]
        a_k  = auxs[k]

        norm = 1.0 / sqrt(D_k + 1)
        V_k  = Matrix{Float64}(undef, 2D_k + 1, M_k)

        for j in 1:M_k
            a_j = a_k[j]
            V_k[1, j] = norm
            @inbounds for (r, ω) in enumerate(ωs_k)
                V_k[2r,   j] = norm * cos(ω * a_j)
                V_k[2r+1, j] = norm * sin(ω * a_j)
            end
        end

        if rank(V_k) < min(size(V_k)...)
            @warn "V^{($k)} may not be full-rank. " *
                  "Rank=$(rank(V_k)), expected=$(min(size(V_k)...)). " *
                  "Consider increasing M_k or adjusting auxiliary points."
        end

        V_list[k]     = V_k
        V_pinv_list[k] = pinv(V_k)
    end

    return V_list, V_pinv_list
end


"""
    _apply_W_on_core(G, W) -> Array{T,3}

Mode-2 multiply a TT core `G` of shape `(rL, d, rR)` by matrix `W` of shape
`(d, b)`, producing a new core of shape `(rL, b, rR)`.

This implements the transformation  G̃[a,s,c] = Σ_j G[a,j,c] · W[j,s].
"""
function _apply_W_on_core(G::Array{T,3}, W::AbstractMatrix) where {T}
    rL, d, rR = size(G)
    size(W, 1) == d || throw(DimensionMismatch("W rows ($(size(W,1))) ≠ core phys dim ($d)"))
    b   = size(W, 2)
    out = Array{T}(undef, rL, b, rR)
    @inbounds for a in 1:rL, c in 1:rR
        out[a, :, c] = W' * view(G, a, :, c)
    end
    return out
end


"""
    ttF_to_ttC(cores_F, ωss; auxs) -> Vector{Array{Float64,3}}

Convert the TT(F) cores (from TCI on the Fourier grid) to TT(C) cores
(in Fourier coefficient space), using the pseudoinverse transform

    G̃^{(k)} = G^{(k)} ×₂ (V^{(k)+})ᵀ

where V^{(k)+} is the Moore-Penrose pseudoinverse of the local sampling matrix.

# Arguments
- `cores_F`: TT cores from `TCI.sitetensors(TCI.TensorTrain(tci))`.
             Each core has shape `(r_{k-1}, M_k, r_k)`.
- `ωss`:     Frequency vectors used to build the Fourier basis.
- `auxs`:    Auxiliary sampling grids used in TCI.

# Returns
Vector of TT(C) cores, each with shape `(r_{k-1}, 2D_k+1, r_k)`.
"""
function ttF_to_ttC(cores_F::Vector{Array{T,3}},
                    ωss::Vector{<:AbstractVector};
                    auxs::Vector{<:AbstractVector{<:Real}}) where {T}
    _, V_pinv_list = _build_local_V(ωss, auxs)

    cores_C = Vector{Array{T,3}}(undef, length(cores_F))
    for p in eachindex(cores_F)
        Vp = V_pinv_list[p]
        size(cores_F[p], 2) == size(Vp, 1) ||
            throw(DimensionMismatch(
                "Core $p phys-dim $(size(cores_F[p],2)) ≠ V_pinv rows $(size(Vp,1))"))
        cores_C[p] = _apply_W_on_core(cores_F[p], Vp)
    end
    return cores_C
end


"""
    eval_from_ttC(cores_C, t, x) -> Float64

Evaluate the FTCI approximation at continuous input `x`:

    F̂(x) = Σ_{s1,…,sL} C[s1,…,sL] · t₁(x)[s1] · … · tL(x)[sL]

implemented efficiently as a left-to-right MPS contraction.

# Arguments
- `cores_C`: TT(C) cores, each of shape `(rL, d_k, rR)`.
- `t`:       Basis functions from `make_basis_vectors`, length L.
- `x`:       Continuous input vector.
"""
function eval_from_ttC(cores_C::Vector{<:Array},
                       t::AbstractVector{<:Function},
                       x::AbstractVector)
    L = length(cores_C)
    L == length(t) || throw(ArgumentError("cores_C and t must have the same length"))

    # Site 1: contract into 1×r1 row vector
    G1 = cores_C[1]
    d1 = size(G1, 2)
    r1 = size(G1, 3)
    t1 = t[1](x)
    length(t1) == d1 || throw(DimensionMismatch("Basis dim mismatch at site 1"))

    vrow = zeros(eltype(G1), 1, r1)
    @inbounds for b in 1:d1
        vrow .+= t1[b] .* reshape(view(G1, 1, b, :), 1, r1)
    end

    # Sites 2…L: accumulate (1×r_{p-1}) × (r_{p-1}×r_p)
    for p in 2:L
        Gp = cores_C[p]
        dp = size(Gp, 2)
        rL, _, rR = size(Gp)
        tp = t[p](x)
        length(tp) == dp || throw(DimensionMismatch("Basis dim mismatch at site $p"))

        # Effective matrix Mp[a,c] = Σ_b tp[b] * Gp[a,b,c]
        Mp = zeros(eltype(Gp), rL, rR)
        @inbounds for b in 1:dp
            Mp .+= tp[b] .* view(Gp, :, b, :)
        end
        size(vrow, 2) == size(Mp, 1) ||
            throw(DimensionMismatch("Bond dimension mismatch at site $p"))
        vrow = vrow * Mp   # (1 × rR)
    end

    size(vrow, 2) == 1 ||
        error("Final TT bond dim should be 1, got $(size(vrow,2))")
    return float(vrow[1])
end


"""
    dense_from_tt(cores) -> Array

Materialise the full dense tensor from TT cores (for small grids only).
Useful for computing exact relative errors in validation.

Each core has shape `(r_{k-1}, d_k, r_k)`.
Returns an array of shape `(d_1, d_2, …, d_L)`.
"""
function dense_from_tt(cores::Vector{Array{T,3}}) where {T}
    L = length(cores)
    A = cores[1]   # (1, d1, r1)
    for k in 2:L
        G  = cores[k]          # (r_{k-1}, d_k, r_k)
        sh = size(A)
        m  = prod(sh[1:end-1])
        rk1 = sh[end]
        rk1 == size(G, 1) || throw(DimensionMismatch("Bond mismatch at site $k"))
        A  = reshape(A, m, rk1) * reshape(G, rk1, :)
        A  = reshape(A, (sh[1:end-1]..., size(G, 2), size(G, 3)))
    end
    return dropdims(A; dims=(1, ndims(A)))   # drop singleton bond dims
end


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Grid relative error (validation helper)
# ─────────────────────────────────────────────────────────────────────────────

"""
    relerr_grid_2norm(f_idx, dims, cores_F; nsamples, seed, exact_limit) -> Float64

Estimate the 2-norm relative error ‖F − F̂‖ / ‖F‖ between the true function
on the TCI grid and its TT approximation.

- Computes exactly when `prod(dims) ≤ exact_limit`.
- Otherwise uses Monte-Carlo sampling with `nsamples` random grid points.
"""
function relerr_grid_2norm(f_idx, dims::AbstractVector{<:Int}, cores_F;
                           nsamples::Int  = 10_000,
                           seed::Int      = 0,
                           exact_limit::Int = 100_000)
    total = prod(dims)
    if total ≤ exact_limit
        F = zeros(Float64, dims...)
        for I in Iterators.product((1:d for d in dims)...)
            F[I...] = f_idx(collect(I))
        end
        Fhat = dense_from_tt(cores_F)
        return norm(F .- Fhat) / max(norm(F), eps())
    else
        rng = MersenneTwister(seed)
        function tt_eval(I::Vector{Int})
            v = vec(view(cores_F[1], 1, I[1], :))
            for p in 2:length(cores_F)
                v = Matrix(view(cores_F[p], :, I[p], :))' * v
            end
            return float(v[1])
        end
        num2 = 0.0; den2 = 0.0
        for _ in 1:nsamples
            I = [rand(rng, 1:d) for d in dims]
            y  = f_idx(I)
            yh = tt_eval(I)
            num2 += (y - yh)^2
            den2 += y^2
        end
        return sqrt(num2) / max(sqrt(den2), eps())
    end
end

end # module FunctionTCI
