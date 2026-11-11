module ForrelationFeature
    using Yao, YaoBlocks, LinearAlgebra, Random, Statistics


    # --------------------------
    # φ_S(x) 기본 규정 (논문 예시와 동일한 기본값)
    #   - 단일항: φ_{i}(x) = x_i
    #   - 이중항: φ_{i,j}(x) = (π - x_i)*(π - x_j)
    # 필요시 사용자 정의로 바꿔 넘길 수 있게 함수 인자로 받음.
    # --------------------------

    # 한 번의 diagonal-in-Z 데이터 인코딩층 U_Φ(x)를 CNOT+Rz 분해로 구성
    function feature_diag_layer(n::Int, x::AbstractVector;
                                singles = (xv,i)->xv[i],
                                pairs   = (xv,i,j)->(π - xv[i])*(π - xv[j]),
                                pair_edges = nothing)
        # 기본 pair_edges: 모든 (i<j) 완전그래프 대신 "선택된" 에지 세트 사용 권장
        # 논문은 하드웨어 에지(예: (0,1)만)로도 시연했지만, 일반적으로는 원하는 에지 집합 지정.
        if pair_edges === nothing
            # 기본: 선형 체인 (1--2--3--...--n)
            pair_edges = [(i, i+1) for i in 1:n-1]
        end
        ops = Any[]
        # 1) single-Z phases: e^{i φ_i(x) Z_i} == Rz(-2 φ_i(x))
        for i in 1:n
            θ = singles(x, i)
            push!(ops, put(n, i=>Rz(θ)))
        end
        # 2) ZZ phases: e^{i φ_{ij}(x) Z_i Z_j} == CNOT(i->j); Rz(-2φ); CNOT(i->j)
        for (i,j) in pair_edges
            θ = pairs(x, i, j)
            push!(ops, chain( control(n, i, j=>X),
                            put(n, j=>Rz(θ)),
                            control(n, i, j=>X) ))
        end
        return chain(ops...)
    end

    # L 번의 H 와 U_Φ(x)로 이루어진 전체 피처맵 준비회로 (L,N) vector를 인풋으로 가짐. 
    function feature_map_circuit(n::Int, x::AbstractVector, L::Int=2; kwargs...)
        layers = Any[]
        for l in 1:L
            push!(layers, repeat(n, H))
            push!(layers, feature_diag_layer(n, x; kwargs...))
        end
        return chain(layers...)
    end

    # 하드웨어 효율적 변분 블록 W(θ): [단일회전층 ∘ CZ-체인]을 L번
    # θ는 크기 (L, n, 3) 배열로 가정: (층, 큐빗, {α,β,γ}) for Rz(α) Rx(β) Rz(γ)
    function variational_block(n::Int, θ::Array{<:Real,3}; ent_edges=nothing)
        L = size(θ,1)
        if ent_edges === nothing
            ent_edges = [(i, i+1) for i in 1:n-1]  # 선형 체인 CZ
        end
        layers = Any[]
        for ℓ in 1:L
            # 단일-큐빗 SU(2) 층
            singles = Any[]
            for q in 1:n
                α, β, γ = θ[ℓ, q, 1], θ[ℓ, q, 2], θ[ℓ, q, 3]
                # U3(α,β,γ) ~ Rz(γ) Rx(β) Rz(α)
                push!(singles, put(n, q=>chain(Rz(α), Rx(β), Rz(γ))))
            end
            # CZ 얽힘층
            ents = Any[ control(n, i, j=>Z) for (i,j) in ent_edges ]
            push!(layers, chain(chain(singles...), chain(ents...)))
        end
        return chain(layers...)
    end
# ------------------------------------------------------------
    # 관측자 O 구성 유틸: 디폴트는 전체 parity Z⊗...⊗Z (n=2이면 Z₁Z₂)
    function default_parity_Z(n::Int)
        return repeat(n, Z)   # returns a block, not a matrix
    end

    function zero_state_projector(n::Int)
        # |0⟩⟨0| = (I + Z)/2 for single qubit,
        # so |0...0⟩⟨0...0| = ⊗_{i=1}^n (I + Z)_i / 2^n
        Id = [1.0 0.0; 0.0 1.0]
        terms = [Id + mat(Z) for _ in 1:n]
        P = terms[1]
        for i in 2:n
            P = kron(P, terms[i])
        end
        P /= 2^n
        return P
    end

        # Returns (Z - I)/2 ⊗ I ⊗ ... ⊗ I as an AbstractMatrix for n qubits
    function first_zero_prob(n::Int)
        Id = [1.0 0.0;0.0 1.0]
        op = 0.5 * (mat(Z) + Id)
        full_op = kron(op, [Id for _ in 1:n-1]...)
        return full_op
    end
#--------------------------------------------------------------

    # 전체 회로 U(x)
    function build_circuit(n::Int, x::AbstractVector, L::Int, θ::Array{<:Real,3};
                        singles = (xv,i)->xv[i],
                        pairs   = (xv,i,j)->(π - xv[i])*(π - xv[j]),
                        pair_edges = nothing)
        return chain(feature_map_circuit(n, x, L; singles=singles, pairs=pairs, pair_edges=pair_edges), repeat(n, H))
    end


    function sample_eigenvalues(vals::AbstractVector, probs::AbstractVector, shots::Int, rng)
        probs = probs ./ sum(probs)
        edges = cumsum(probs)
        edges[end] = 1.0                     # guard against rounding
        out = similar(vals, shots)
        for s in 1:shots
            r = rand(rng)
            idx = searchsortedfirst(edges, r)
            out[s] = vals[idx]
        end
        return out
    end
    # 메인: f_q(x, θ; O=...)  —— ⟨0| U† O U |0⟩
    """It takes (L,N) vector as input"""
    function Forrelation_fq(n::Int, x::AbstractVector, L::Int, θ::Array{<:Real,3};
                O::Union{Nothing, AbstractMatrix, AbstractBlock}=nothing,
                pair_edges = nothing,
                singles = (xv,i)->xv[i],
                pairs   = (xv,i,j)->(π - xv[i])*(π - xv[j]),
                shots::Union{Nothing, Int}=nothing,
                rng = Random.default_rng())
    
        O === nothing && (O = zero_state_projector(n))
        U = build_circuit(n, x, L, θ; singles=singles, pairs=pairs,
                        pair_edges=pair_edges)
        ψ0 = zero_state(n)
        ψ  = apply(ψ0, U)

        # prepare observable as both block (for exact) and matrix (for sampling)
        obs_block = O isa AbstractMatrix ? matblock(O) : O
        obs_mat   = O isa AbstractMatrix ? O : mat(O)

        if shots === nothing
            return real(expect(obs_block, ψ))
        end

        eig = eigen(Hermitian(obs_mat))
        ψvec = vec(state(ψ))                      # flatten state amplitudes
        amps = eig.vectors' * ψvec
        probs = abs2.(amps)
        outcomes = sample_eigenvalues(real(eig.values), probs, shots, rng)
        return (mean=mean(outcomes), outcomes=outcomes)
    end


    function feature_diag_layer_nopre(n::Int,  x::AbstractVector;
        pair_edges = nothing)
    #
        if pair_edges === nothing
        # 기본: 선형 체인 (1--2--3--...--n)
            pair_edges = [(i, i+1) for i in 1:n-1]
        end
    
        ops = Any[]
        # 1) single-Z phases: e^{i φ_i(x) Z_i/2} == Rz( φ_i(x))
        for i in 1:n
            θ = x[i]
            push!(ops, put(n, i=>Rz(θ)))
        end
        # 2) ZZ phases: e^{i φ_{ij}(x) Z_i Z_j/2} == CNOT(i->j); Rz(-φ); CNOT(i->j)
        for (k,pair) in enumerate(pair_edges)
            i = pair[1]
            j = pair[2]
            θ = x[n+k]
            push!(ops, chain( control(n, i, j=>X),
                put(n, j=>Rz(θ)),
                control(n, i, j=>X) ))
        end
        return chain(ops...)
    end
    
    # L 번의 H 와 두 번의 U_Φ(x)로 이루어진 전체 피처맵 준비회로
    function feature_map_circuit_nopre(n::Int,  x::AbstractVector, L::Int; kwargs...)
        layers = Any[]
        for l in 1:L
            push!(layers, repeat(n, H))
            push!(layers, feature_diag_layer_nopre(n, x; kwargs...))
        end
        return chain(layers...)
    end

    

end


module FunctionTCI

    using LinearAlgebra, Random


    export build_local_V, make_basis_vectors, gamma_vals,apply_W_on_core,
        ttF_to_ttC,eval_from_ttC,dense_from_tt    

    # ---------- helpers: per-mode basis, core transform, TT expansion ----------


    """
    build_local_V(ωss, Ds; α=2, domain_range=2π, seed=0)

    Build local V matrices and their Kronecker product as described in the algorithm.

    # Arguments
    - `ωss`: Vector of vectors, where `ωss[k]` contains the frequencies {ω^{(k)}_r} for layer k
    - 'auxs' : 'L' Cartesian product of 'M^{(k)}'-tuples (default: nothing (randomly choose)). 
    - `α`: Scaling factor such that M^{(k)} = α * D^{(k)} (default: 2)
    - `domain_range`: Range for random auxiliary inputs (default: 2π)
    - `seed`: Random seed (default: 0)

    # Returns

    - `V_list`: List of individual V^{(k)} matrices
    - `a_list`: List of auxiliary input vectors used for each layer
    """
    function build_local_V(ωss::Vector{<:AbstractVector{<:Real}};
                        auxs = nothing,
                        α::Real = 2,
                        seed::Int = 0)
        
        L = length(ωss) ##Length of the total TN
        Ds = [length(ωs) for ωs in ωss]
        
        
        # Compute M^{(k)} = α * (2D^{(k)}+1) if auxs is nothing, otherwise use lengths of auxs
        if auxs === nothing
            Ms = [Int(round(α * (2*Ds[k]+1))) for k in 1:L]
        else
            @assert length(auxs) == L "Length of auxs must match length of ωss"
            Ms = [length(a_vec) for a_vec in auxs]
        end
        
        # Storage for V^{(k)} matrices and auxiliary inputs
        V_list = Vector{Matrix{Float64}}(undef, L)
        a_list = Vector{Vector{Float64}}(undef, L)

        
        # For each layer k = 1, ..., L
        for k in 1:L
            M_k = Ms[k]
            D_k = Ds[k]
            ωs_k = ωss[k]
            a_list[k] = auxs[k]
            
            
            # Build local feature vectors b^{(k)}(a^j_k) for each j
            # Each vector has dimension 2*D_k + 1: [1, cos(ω₁a), sin(ω₁a), ..., cos(ω_{D_k}a), sin(ω_{D_k}a)]
            V_k = zeros(Float64, 2*D_k + 1, M_k)
            
            normalization = 1.0 / sqrt(D_k + 1)
            
            for j in 1:M_k
                a_j = a_list[k][j]
                # First element: 1 (normalized)
                V_k[1, j] = normalization
                
                # Then pairs: cos(ω_r * a^j_k), sin(ω_r * a^j_k) for each frequency
                for (r_idx, ω_r) in enumerate(ωs_k)
                    cos_idx = 1 + 2*(r_idx - 1) + 1  # 2, 4, 6, ...
                    sin_idx = 1 + 2*(r_idx - 1) + 2  # 3, 5, 7, ...
                    V_k[cos_idx, j] = normalization * cos(ω_r * a_j)
                    V_k[sin_idx, j] = normalization * sin(ω_r * a_j)
                end
            end
            
            V_list[k] = V_k
            
            # Check linear independence (optional warning)
            if rank(V_k) < min(size(V_k)...)
                @warn "V^{(k)} for k=$k may not be full rank. Rank = $(rank(V_k)), expected = $(min(size(V_k)...))"
            end
        end
        
        # Compute pseudoinverse for each V^{(k)} individually
        # Note: We don't compute the full Kronecker product V or V_pinv as they are
        # extremely memory-intensive and not needed. The property (A ⊗ B)^+ = A^+ ⊗ B^+
        # means we can work with individual pseudoinverses.
        V_pinv_list = [pinv(V_k) for V_k in V_list]
        
        # Return only what's needed (V and V_pinv are not computed to save memory)
        return  V_list, V_pinv_list, a_list
    end

    """
    Given a vector of frequencies and vector of pre-processing functions x->ϕ_k(x),
    returns x-> [1 cosw_1ϕ_1(x) sinw_1ϕ_1(x) ... cosw_Nϕ_N(x) sinw_Nϕ_N(x) ]
    """
    function make_basis_vectors(ωss::Vector{<:AbstractVector}, phis::AbstractVector)
        @assert length(ωss) == length(phis) "ωss must be a vector of length of phis"
        N = length(ωss)

        function tp(ωs::AbstractVector{<:Real}, fphi::Function)
            D = length(ωs)
            # Returns a function of x
            return x -> begin
                v = [1.0]
                for ω in ωs
                    push!(v, cos(ω * fphi(x)))
                    push!(v, sin(ω * fphi(x)))
                end
                return (1/sqrt(D+1)).*v
            end
        end

        t = [tp(ωss[p], phis[p]) for p in 1:N]
        return t
    end
    "Need for calculating appropriate domain range"
    function gamma_vals(ωss)
        gammas = Float64[]
        for ω in ωss
            ω = sort(abs.(ω))  # sort absolute values for min computations
            min1 = minimum(ω)  # min_j |ω_j|
            n = length(ω)
            min2 = Inf        # min_{j≠k} |ω_j - ω_k|
            min3 = Inf        # min_{j≠k} |ω_j + ω_k|
            for j in 1:n
                for k in 1:n
                    if j != k
                        min2 = min(min2, abs(ω[j] - ω[k]))
                        min3 = min(min3, abs(ω[j] + ω[k]))
                    end
                end
            end
            γ = min(min1, min2, min3)
            if γ <= 0
                error("γ ≤ 0 encountered! Check that ω_j are nonzero and nondegenerate.")
            end
            push!(gammas, γ)
        end
        return gammas
    end

    "Mode-2 multiply a TT core G (rL, d, rR) by W (d, b) → (rL, b, rR)."
    function apply_W_on_core(G::Array{T,3}, W::AbstractMatrix) where {T}
        rL, d, rR = size(G)
        @assert size(W,1) == d
        b = size(W,2)
        out = Array{T}(undef, rL, b, rR)
        @inbounds for a in 1:rL, c in 1:rR
            out[a, :, c] = view(G, a, :, c)' * W  # transpose to (1,d) for correct matmul
        end
        return out
    end


    "Expand TT cores to a dense tensor of size (d_1,d_2,...,d_N)."
    function dense_from_tt(cores::Vector{Array{T,3}}) where {T}
        M = length(cores)
        A = cores[1]                           # (1, d1, r1)
        for k in 2:M
            G   = cores[k]                     # (r_{k-1}, d_k, r_k)
            sh  = size(A)
            m   = prod(sh[1:end-1])            # (r0*d1*...*d_{k-1})
            rk1 = sh[end]
            @assert rk1 == size(G,1)
            A   = reshape(A, m, rk1) * reshape(G, rk1, :)
            A   = reshape(A, (sh[1:end-1]..., size(G,2), size(G,3)))  # (..., d_k, r_k)
        end
        # drop the first and last singleton TT-bond dims
        A = dropdims(A; dims=(1, ndims(A)))
        return A                                # (d_1,d_2,...,d_N)
    end


    "TT(F) cores → TT(C) cores using per-mode W_p = (V_p^{+})ᵀ."
    #We ignore preprocessing functions in local_basis_vec
    function ttF_to_ttC(cores_F::Vector{Array{T,3}},ωss::Vector{<:AbstractVector}; auxs=nothing) where {T}

        cores_C = Vector{Array{T,3}}(undef, length(cores_F))
        
        _, V_pinv_list, _ = build_local_V(ωss; auxs = auxs)

        for p in eachindex(cores_F)
            Vp = V_pinv_list[p]
            @assert size(cores_F[p],2) == size(Vp,1) "site $p has physical dim $(size(cores_F[p],2)) ≠ size(Vp,2) ($(size(Vp,1)))"
            cores_C[p] = apply_W_on_core(cores_F[p], Vp)  # (rL,2D^(p)+1,rR)
        end
        return cores_C
    end



    """
        eval_from_ttC(cores_C::Vector{<:Array}, t::Vector{<:AbstractVector}, x)

    Evaluate a TT function given TT cores `cores_C` and a list of basis vectors `t` for each site.
    - `cores_C[k]` should have shape (rL, d_k, rR)
    - `t[k]` should be a function that takes x as input. t[k](x) has a dimension d_k (physical dimension 2D^{(k)}+1)

    Returns a scalar value.
    """
    function eval_from_ttC(cores_C::Vector{<:Array}, t::AbstractVector{<:Function}, x::AbstractVector)
        Len = length(cores_C)
        @assert length(t) == Len "Number of basis vectors does not match number of TT cores"
        # Ensure t[k](x) yields a numeric vector
        t1 = t[1](x)
        @assert eltype(t1) <: Real "t[1](x) must return a vector of Real"       
        
        # site 1 -> 1×r1 row 
        d1 = size(cores_C[1], 2)
        r1 = size(cores_C[1], 3)
        t1 = t[1](x)
        @assert length(t1) == d1 "Basis vector at site 1 has wrong dimension"
        vrow = zeros(eltype(cores_C[1]), 1, r1)  # 1×r1
        G1 = cores_C[1]
        
        @inbounds for b in 1:d1
            g = @view G1[1, b, :]                # (r1,)
            vrow .+= t1[b] .* reshape(g, 1, r1)
        end
        # sweep
        for p in 2:Len
            dp = size(cores_C[p], 2)
            tp = t[p](x)
            @assert length(tp) == dp "Basis vector at site $p has wrong dimension"
            Gp = cores_C[p]
            rL, _, rR = size(Gp)
            Mp = zeros(eltype(Gp), rL, rR)       # (r_{p-1}×r_p)
            
            @inbounds for b in 1:dp
                Mp .+= tp[b] .* @view Gp[:, b, :]
            end
            @assert size(vrow,2) == size(Mp,1) "rank mismatch at site $p"
            vrow = vrow * Mp                     # (1×r_p)
        end
        @assert size(vrow,2) == 1 "final TT rank should be 1, got $(size(vrow))"
        return float(vrow[1])
    end
end #module


module VQMLOracle

    using LinearAlgebra, Random, Statistics,Yao, YaoBlocks
    
    # -------- utilities --------
    
    pow2(n) = 1 << n
    
    """Kronecker of a vector of small matrices (left-to-right order)."""
    function kron_all(ms::Vector{<:AbstractMatrix})
        K = Matrix{ComplexF64}(ms[1])
        @inbounds for k in 2:length(ms)
            K = kron(K, Matrix{ComplexF64}(ms[k]))
        end
        return K
    end
    
    """Haar-random unitary of dimension d (QR-based)."""
    function haar_unitary(rng::AbstractRNG, d::Int)
        Z = randn(rng, ComplexF64, d, d) .+ 1im * randn(rng, ComplexF64, d, d)
        Q, R = qr(Z); Q = Matrix(Q)
        Λ = Diagonal((diag(R) ./ abs.(diag(R))) .^ (-1))
        return Q * Λ
    end
    
    """Pauli Z on qubit q (1-based), identity elsewhere, N qubits (matrix form)."""
    function Z_on(q::Int, N::Int)
        obs = put(N, q => z)
        return obs
    end
    
    # -------- oracle builder (Yao-based simulation) --------
    
    """
    make_fq(N, L; W_list=nothing, O=:Z1, seed=0, init=:plus)
    
    Returns (fq, W_list, O_mat, info).
    
    - N: number of qubits
    - L: number of layers
    - W_list: optional Vector of length L, each a 2^N×2^N unitary (matrix).
              If `nothing`, Haar-random unitaries are generated.
    - O: observable or symbol:
         :Z1   => Z on qubit 1
         :Zsum => ∑_{j=1}^N Z_j
         or a full 2^N×2^N Hermitian matrix
    - seed: RNG seed (used only if W_list is nothing)
    - init: :plus (|+⟩^⊗N, default) or :zero (|0⟩^⊗N)
    
    `fq(phis)` expects length(phis) = N*L in layer-major order:
    [φ_{1,1}, …, φ_{1,N}, φ_{2,1}, …, φ_{L,N}]
    
    Internally this uses Yao/YaoBlocks for state evolution and expectation values.
    """
    function make_fq(N::Int, L::Int; W_list=nothing, O=:Z1, seed=0, init=:plus)
        d   = pow2(N)
        rng = MersenneTwister(seed)
    
        # ----- Observable (matrix form first, for compatibility) -----
        O_mat = if O === :Z1
            Z_on(1, N)
        end

        # Hermitize softly (in case user gave a slightly non-Hermitian matrix)
        O_mat = (O_mat + O_mat') / 2
        
    
        # ----- Interleaving unitaries (keep as matrices for return, use matblock for evolution) -----
        W = if W_list === nothing
            [haar_unitary(rng, d) for _ in 1:L]
        else
            length(W_list) == L || error("W_list length != L")
            all(size(W_list[ℓ]) == (d, d) for ℓ in 1:L) || error("W_list has wrong size")
            [Matrix{ComplexF64}(W_list[ℓ]) for ℓ in 1:L]
        end
    
        # Yao blocks corresponding to W_list
        W_blocks = [matblock(W[ℓ]) for ℓ in 1:L]
    
        # ----- Initial state as Yao register -----
        ψ0_reg = if init === :plus
            # |+⟩^⊗N = uniform superposition
            uniform_state(ComplexF64, N)
        elseif init === :zero
            # |0...0⟩
            zero_state(ComplexF64, N)
        else
            error("init must be :plus or :zero")
        end
    
        # ----- Per-layer data encoding: ⊗_j Rz(φ_{ℓ,j}) as a Yao block -----
        function Rz_layer_block(φ::AbstractVector{<:Real})
            length(φ) == N || error("φslice length must be N")
            # chain(N, iterator-of-put-blocks)
            return chain(N, (put(N, j => Rz(φ[j])) for j in 1:N))
        end
    
        # ----- Oracle fq implemented via Yao -----
        function fq(phis::AbstractVector{<:Real})
            length(phis) == N*L || error("length(phis) must be N*L")
            reg = copy(ψ0_reg)
            @inbounds for ℓ in 1:L
                ϕℓ = view(phis, (ℓ-1)*N + 1 : ℓ*N)
                reg |> Rz_layer_block(ϕℓ) |> W_blocks[ℓ]
            end
            # Yao's expect gives ⟨ψ|O|ψ⟩
            val = expect(O_block, reg)
            return real(val)  # should be real for Hermitian O
        end
    
        info = (; N, L, dim = d, seed, init, obs_norm = opnorm(O_mat))
        return fq, W, O_mat, info
    end
    
    """
    make_fq_noisy(N, L; W_list=nothing, O=:Z1, seed=0, init=:plus, delta=1e-4)
    
    Returns (fq_noisy, fq_clean, W_list, O_mat, info).
    
    Same as make_fq, but fq_noisy(phis) returns fq(phis) + η,
    where η ~ Normal(0, sqrt(delta)).
    """
    function make_fq_noisy(
            N::Int,
            L::Int;
            W_list=nothing,
            O=:Z1,
            seed=0,
            init=:plus,
            delta=1e-4,
        )
        fq, W, O_mat, info = make_fq(N, L; W_list=W_list, O=O, seed=seed, init=init)
        rng = MersenneTwister(seed + 123456)  # offset to avoid correlation with circuit RNG
    
        function fq_noisy(phis::AbstractVector{<:Real})
            val   = fq(phis)
            noise = sqrt(delta) * randn(rng)
            return val + noise
        end
    
        return fq_noisy, fq, W, O_mat, info
    end
    
    end # module
    