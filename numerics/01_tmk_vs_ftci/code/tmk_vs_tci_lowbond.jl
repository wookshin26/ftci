"""
Compare FTCI with the truncated trigonometric monomial kernel (TMK).

Default mode is a quick smoke test. Use

    julia --project=Numerics Numerics/tmk_vs_tci_lowbond.jl --full

for the larger sweep.
"""

using Dates, Printf, Random

include(joinpath(@__DIR__, "FunctionTCI.jl"))
include(joinpath(@__DIR__, "TrigonometricMonomialKernel.jl"))

using .FunctionTCI:
    CounterWithLog,
    counted_unique_with_log,
    dense_from_tt,
    eval_from_ttC,
    gamma_vals,
    make_basis_vectors,
    ttF_to_ttC
using .TrigonometricMonomialKernel:
    fit_tmk,
    predict,
    sample_uniform_inputs

import TensorCrossInterpolation as TCI

const OUTDIR = joinpath(@__DIR__, "results")

function feasible_tt_ranks(dims::AbstractVector{<:Integer}, chi::Integer)
    L = length(dims)
    ranks = ones(Int, L + 1)
    for k in 1:(L - 1)
        ranks[k + 1] = min(Int(chi), prod(dims[1:k]), prod(dims[(k + 1):end]))
    end
    return ranks
end

function tt_frobenius_norm(cores)
    env = ones(Float64, 1, 1)
    for G in cores
        _, d, rR = size(G)
        next_env = zeros(Float64, rR, rR)
        @inbounds for s in 1:d
            Gs = view(G, :, s, :)
            next_env .+= transpose(Gs) * env * Gs
        end
        env = next_env
    end
    return sqrt(max(env[1, 1], 0.0))
end

function random_tt_cores(rng::AbstractRNG, dims::AbstractVector{<:Integer}, chi::Integer)
    ranks = feasible_tt_ranks(dims, chi)
    cores = [randn(rng, ranks[k], Int(dims[k]), ranks[k + 1]) for k in eachindex(dims)]
    scale = max(tt_frobenius_norm(cores), eps())^(1 / length(cores))
    foreach(G -> G ./= scale, cores)
    return cores
end

function make_lowbond_target(rng::AbstractRNG, L::Integer, chi::Integer)
    omega_sets = [Float64[1.0] for _ in 1:Int(L)]
    phis = [let k = k; x -> x[k] end for k in 1:Int(L)]
    basis = make_basis_vectors(omega_sets, phis)
    phys_dims = [2 * length(omega_sets[k]) + 1 for k in eachindex(omega_sets)]
    cores = random_tt_cores(rng, phys_dims, chi)
    oracle = x -> eval_from_ttC(cores, basis, x)
    return (; oracle, cores, omega_sets, basis, phys_dims)
end

function local_basis_norms_sq(local_dim::Integer)
    isodd(local_dim) || throw(ArgumentError("local Fourier dimension must be odd"))
    D = (Int(local_dim) - 1) ÷ 2
    D >= 1 || throw(ArgumentError("local Fourier dimension must be at least 3"))
    norms = fill(1.0 / (2 * (D + 1)), Int(local_dim))
    norms[1] = 1.0 / (D + 1)
    return norms
end

function coefficient_weight_diagnostics(cores)
    C = dense_from_tt(cores)
    dims = size(C)
    L = length(dims)
    raw_by_weight = zeros(Float64, L + 1)
    l2_by_weight = zeros(Float64, L + 1)
    local_norms = [local_basis_norms_sq(d) for d in dims]

    for I in CartesianIndices(C)
        idx = Tuple(I)
        weight = 0
        basis_norm_sq = 1.0
        @inbounds for k in 1:L
            component = idx[k]
            component == 1 || (weight += 1)
            basis_norm_sq *= local_norms[k][component]
        end
        c2 = abs2(C[I])
        raw_by_weight[weight + 1] += c2
        l2_by_weight[weight + 1] += c2 * basis_norm_sq
    end

    raw_total = max(sum(raw_by_weight), eps())
    l2_total = max(sum(l2_by_weight), eps())
    raw_cumulative = cumsum(raw_by_weight) ./ raw_total
    l2_cumulative = cumsum(l2_by_weight) ./ l2_total
    best_trunc_error = sqrt.(max.(1 .- l2_cumulative, 0.0))

    return (;
        raw_by_weight,
        l2_by_weight,
        coeff_energy_fraction = raw_cumulative,
        l2_energy_fraction = l2_cumulative,
        best_trunc_rel_l2_error = best_trunc_error,
    )
end

function append_diagnostic_rows!(rows, mode, L::Integer, seed::Integer, cores)
    diagnostics = coefficient_weight_diagnostics(cores)
    for lambda in 0:Int(L)
        push!(rows, (;
            mode,
            L=Int(L),
            seed=Int(seed),
            method="TMK_BEST_TRUNCATION",
            lambda,
            chi=-1,
            samples=0,
            features=length(TrigonometricMonomialKernel.truncated_frequencies(L, lambda)),
            rel_l2_error=diagnostics.best_trunc_rel_l2_error[lambda + 1],
            seconds=0.0,
            bond_dims="",
            pivot_error="",
            coeff_energy_fraction=diagnostics.coeff_energy_fraction[lambda + 1],
            l2_energy_fraction=diagnostics.l2_energy_fraction[lambda + 1],
            best_trunc_rel_l2_error=diagnostics.best_trunc_rel_l2_error[lambda + 1],
        ))
    end
    return rows
end

function fit_ftci(oracle::Function, omega_sets, basis;
                  chi::Integer, tolerance::Real, rng::AbstractRNG, alpha::Integer=2)
    L = length(omega_sets)
    phys_dims = [2 * length(omega_sets[k]) + 1 for k in 1:L]
    dims = max.(phys_dims, Int(alpha) .* phys_dims)
    periods = 2pi ./ gamma_vals(omega_sets)
    auxs = [periods[k] .* rand(rng, dims[k]) for k in 1:L]

    f_idx = id -> oracle([auxs[p][id[p]] for p in 1:L])
    ctr = CounterWithLog(; verbose=false)
    f_cached = counted_unique_with_log(f_idx, ctr)

    Random.seed!(Int(rand(rng, UInt32)))
    elapsed = @elapsed begin
        tci, _, errors = TCI.crossinterpolate2(
            Float64, f_cached, dims;
            pivotsearch=:full,
            tolerance=tolerance,
            maxbonddim=Int(chi),
        )
    end

    cores_F = TCI.sitetensors(TCI.TensorTrain(tci))
    cores_C = ttF_to_ttC(cores_F, omega_sets; auxs)
    return (;
        predict = x -> eval_from_ttC(cores_C, basis, x),
        samples = ctr.n,
        seconds = elapsed,
        bond_dims = TCI.linkdims(tci),
        pivot_error = isempty(errors) ? NaN : last(errors),
    )
end

function relative_l2_error(f::Function, g::Function, xs)
    num2 = 0.0
    den2 = 0.0
    for x in xs
        y = Float64(f(x))
        yh = Float64(g(x))
        num2 += (y - yh)^2
        den2 += y^2
    end
    return sqrt(num2) / max(sqrt(den2), eps())
end

function csv_escape(x)
    s = string(x)
    return occursin(r"[,\n\"]", s) ? "\"" * replace(s, "\"" => "\"\"") * "\"" : s
end

function write_rows(path::AbstractString, rows)
    mkpath(dirname(path))
    header = [
        "mode", "L", "seed", "method", "lambda", "chi", "samples",
        "features", "rel_l2_error", "seconds", "bond_dims", "pivot_error",
        "coeff_energy_fraction", "l2_energy_fraction", "best_trunc_rel_l2_error",
    ]
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join((csv_escape(get(row, Symbol(h), "")) for h in header), ","))
        end
    end
end

function config_from_args(args)
    full = "--full" in args
    if full
        return (;
            mode="full",
            Ls=[4, 6, 8, 10, 12],
            seeds=1:5,
            lambdas=[1, 2, 3],
            tmk_samples=[50, 100, 200, 500, 1_000, 2_000, 5_000],
            chi_target=8,
            chi_ftci=12,
            tci_tol=1e-8,
            ntest=1_000,
        )
    end
    return (;
        mode="quick",
        Ls=[4],
        seeds=1:1,
        lambdas=0:4,
        tmk_samples=[20, 50, 100, 200, 500, 1_000, 2_000, 5_000],
        chi_target=2,
        chi_ftci=4,
        tci_tol=1e-8,
        ntest=300,
    )
end

function run_comparison(cfg)
    rows = NamedTuple[]
    for L in cfg.Ls, seed in cfg.seeds
        rng_target = MersenneTwister(10_000 * L + seed)
        target = make_lowbond_target(rng_target, L, cfg.chi_target)
        append_diagnostic_rows!(rows, cfg.mode, L, seed, target.cores)
        xs_test = sample_uniform_inputs(MersenneTwister(20_000 * L + seed), cfg.ntest, L)

        @printf("\nL=%d seed=%d target_chi=%d\n", L, seed, cfg.chi_target)
        ftci = fit_ftci(
            target.oracle,
            target.omega_sets,
            target.basis;
            chi=cfg.chi_ftci,
            tolerance=cfg.tci_tol,
            rng=MersenneTwister(30_000 * L + seed),
        )
        ftci_error = relative_l2_error(target.oracle, ftci.predict, xs_test)
        @printf("  FTCI: samples=%d rel_l2=%.3e bonds=%s\n",
            ftci.samples, ftci_error, string(ftci.bond_dims))

        push!(rows, (;
            mode=cfg.mode,
            L,
            seed,
            method="FTCI",
            lambda=-1,
            chi=cfg.chi_ftci,
            samples=ftci.samples,
            features=3^L,
            rel_l2_error=ftci_error,
            seconds=ftci.seconds,
            bond_dims=string(ftci.bond_dims),
            pivot_error=ftci.pivot_error,
            coeff_energy_fraction="",
            l2_energy_fraction="",
            best_trunc_rel_l2_error="",
        ))

        for lambda in cfg.lambdas
            lambda > L && continue
            features = length(TrigonometricMonomialKernel.truncated_frequencies(L, lambda))
            for n in cfg.tmk_samples
                rng_tmk = MersenneTwister(40_000 * L + 100 * seed + 10 * lambda + n)
                seconds = @elapsed model = fit_tmk(
                    target.oracle,
                    L;
                    lambda,
                    nsamples=n,
                    rng=rng_tmk,
                )
                tmk_error = relative_l2_error(target.oracle, x -> predict(model, x), xs_test)
                @printf("  TMK Lambda=%d n=%d features=%d rel_l2=%.3e\n",
                    lambda, n, features, tmk_error)
                push!(rows, (;
                    mode=cfg.mode,
                    L,
                    seed,
                    method="TMK",
                    lambda,
                    chi=-1,
                    samples=n,
                    features,
                    rel_l2_error=tmk_error,
                    seconds,
                    bond_dims="",
                    pivot_error="",
                    coeff_energy_fraction="",
                    l2_energy_fraction="",
                    best_trunc_rel_l2_error="",
                ))
            end
        end
    end
    return rows
end

function mean_or_nan(vals)
    isempty(vals) && return NaN
    return sum(vals) / length(vals)
end

function sample_threshold_rows(rows, thresholds)
    Ls = sort(unique(r.L for r in rows if r.method in ("FTCI", "TMK")))
    seeds = sort(unique(r.seed for r in rows if r.method in ("FTCI", "TMK")))
    out = NamedTuple[]
    max_tmk_samples = maximum([r.samples for r in rows if r.method == "TMK"]; init=0)

    for threshold in thresholds, L in Ls
        ftci_vals = Float64[]
        tmk_vals = Float64[]
        for seed in seeds
            ftci = [r for r in rows if r.method == "FTCI" && r.L == L && r.seed == seed]
            if !isempty(ftci) && ftci[1].rel_l2_error <= threshold
                push!(ftci_vals, Float64(ftci[1].samples))
            end

            tmk_candidates = [
                r.samples for r in rows
                if r.method == "TMK" && r.L == L && r.seed == seed && r.rel_l2_error <= threshold
            ]
            isempty(tmk_candidates) || push!(tmk_vals, Float64(minimum(tmk_candidates)))
        end

        total_count = count(seed -> any(r -> r.L == L && r.seed == seed && r.method == "FTCI", rows), seeds)
        push!(out, (;
            threshold,
            L,
            method="FTCI",
            mean_samples=mean_or_nan(ftci_vals),
            reached_count=length(ftci_vals),
            total_count,
            max_tmk_samples,
        ))
        push!(out, (;
            threshold,
            L,
            method="TMK",
            mean_samples=mean_or_nan(tmk_vals),
            reached_count=length(tmk_vals),
            total_count,
            max_tmk_samples,
        ))
    end

    return out
end

function main(args=ARGS)
    cfg = config_from_args(args)
    rows = run_comparison(cfg)
    stamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    outpath = joinpath(OUTDIR, "tmk_vs_tci_lowbond_$(cfg.mode)_$stamp.csv")
    write_rows(outpath, rows)
    println("\nSaved results: $outpath")
    return outpath
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
