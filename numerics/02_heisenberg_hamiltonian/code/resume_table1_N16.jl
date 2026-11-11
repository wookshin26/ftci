# ════════════════════════════════════════════════════════════════════════
#  RESUME Table I — compute ONLY the missing N (N=16), reuse cached N=9–15
# ════════════════════════════════════════════════════════════════════════
#
#  HOW TO USE
#  ----------
#  Paste this into a NEW code cell directly below "## 8. Table I Reproduction".
#  Do NOT re-run the original Table I cell (it would wipe table_results and
#  recompute N=9–15 from scratch).
#
#  ▶ CONFIRMED SITUATION (2026-06-15): the kernel that ran N=9–15 is dead; the
#    current IJulia kernels are fresh, so `table_results` is NOT in memory.
#    This cell will rebuild N=9–15 from the on-disk .jls files, then run N=16.
#
#  BEFORE running this cell, run ONLY these two cells (top to bottom):
#     1) the top config cell (the big ╔═══════ banner block) — defines the
#        constants α, N_LIST, MPS_CUTOFF, YAO_CUTOVER, TCI_TOL, M_VAL, ... .
#     2) the "## 1. Setup" code cell — `include(...)` + `using .Modules`, which
#        defines make_heisenberg_oracle_auto, ttF_to_ttC, make_basis_vectors,
#        eval_from_ttC, gamma_vals, CounterWithLog, counted_unique_with_log,
#        and `import TensorCrossInterpolation as TCI`.
#     Sections 2–8 are NOT needed.
#
#  ⚠ Do NOT use "Run All Above" on this cell — that would re-execute the
#    original Table I cell and recompute N=9–15 from scratch (days of compute).
#
#  Recovered from disk: eps_rel & n_calls for N=9–15 (full fidelity).
#  Lost: time_s & bonds for N=9–15 (the raw .jls don't store them) → set to
#  NaN / empty. Those timings were sleep-inflated anyway; the on-screen output
#  still shows them if you ever want to transcribe them by hand.
#
#  ⚠ RUN_ID: the interrupted run is "20260611_124951". This is hard-pinned in
#  `rid` below so the patch always targets your existing files regardless of
#  whether cell 4 has been re-run.
# ════════════════════════════════════════════════════════════════════════

rid       = "20260611_124951"          # interrupted run's ID — matches results/raw/table1_N*_chi24_*.jls
chi_table = QUICK_MODE ? CHI_MAX : 24

# ── 1) Ensure table_results exists; backfill cached N from disk if needed ──
if !@isdefined(table_results)
    table_results = Dict{Int,NamedTuple}()
end
for Nb in N_LIST
    haskey(table_results, Nb) && continue          # keep full in-memory entry if present
    f = joinpath(DIR_RAW, "table1_N$(Nb)_chi$(chi_table)_$(rid).jls")
    if isfile(f)
        d = deserialize(f)
        table_results[Nb] = (N=d["N"], chi=d["chi"], eps_rel=d["eps_rel"],
                             n_calls=d["n_calls"], time_s=NaN, bonds=Int[])
        @info "recovered cached N=$Nb from disk (time_s/bonds unavailable)"
    end
end

missing_N = [Nb for Nb in N_LIST if !haskey(table_results, Nb)]
println("Resuming Table I (chi=$chi_table) — N still missing: $missing_N\n")

# ── 2) Compute ONLY the missing N (identical body to the original loop) ──
for Nb in missing_N
    println("N = $Nb")
    dt_b   = T_EVOLVE / N_STEPS
    ωss_b  = vcat(
        [(2*dt_b).* collect(1:N_STEPS*(Nb-1)) for _ in 1:3],
        [(2*dt_b).* collect(1:N_STEPS*Nb)])
    dims_b   = [2*length(ωss_b[k])+1 for k in 1:4]
    Ms_b  = α .* dims_b   # M^(k) = α d_k
    rng_b    = MersenneTwister(RNG_SEED)
    gammas_b = gamma_vals(ωss_b)
    avecs_b  = [2π/gammas_b[k] .* rand(rng_b, Ms_b[k]) for k in 1:4]

    fq_b    = make_heisenberg_oracle_auto(Nb; n_steps=N_STEPS, t=T_EVOLVE,
                  maxdim=MPS_MAXDIM, cutoff=MPS_CUTOFF, crossover=YAO_CUTOVER)
    fidx_b  = id -> fq_b([avecs_b[p][id[p]] for p in 1:4])
    ctr_b   = CounterWithLog(; verbose=false)
    fcach_b = counted_unique_with_log(fidx_b, ctr_b)

    t_b = @elapsed begin
        tci_b, _, _ = TCI.crossinterpolate2(Float64, fcach_b, Ms_b;
            tolerance=TCI_TOL, maxbonddim=chi_table)
    end
    cF_b   = TCI.sitetensors(TCI.TensorTrain(tci_b))
    cC_b   = ttF_to_ttC(cF_b, ωss_b; auxs=avecs_b)
    tbas_b = make_basis_vectors(ωss_b, [J -> J[k] for k in 1:4])

    rng_vb = MersenneTwister(RNG_SEED+100+Nb)
    Js_vb  = [2.0 .* rand(rng_vb, 4) .- 1.0 for _ in 1:M_VAL]
    y_tb   = [fq_b(J) for J in Js_vb]
    y_hb   = [eval_from_ttC(cC_b, tbas_b, J) for J in Js_vb]
    ε_b    = sqrt(sum((y_tb.-y_hb).^2)) / max(sqrt(sum(y_tb.^2)), eps())

    table_results[Nb] = (N=Nb, chi=chi_table, eps_rel=ε_b,
                          n_calls=ctr_b.n, time_s=t_b, bonds=TCI.linkdims(tci_b))
    @printf("  eps_rel=%.4f  calls=%d  time=%.1fs\n", ε_b, ctr_b.n, t_b)

    serialize(joinpath(DIR_RAW,
        "table1_N$(Nb)_chi$(chi_table)_$(rid).jls"),
        Dict("N"=>Nb,"chi"=>chi_table,"eps_rel"=>ε_b,"n_calls"=>ctr_b.n))
end

# ── 3) Final summary table + processed file (same format as original cell) ──
println("\n=== Table I (chi=$chi_table) ===")
@printf("%-4s  %-8s  %-14s  %-10s\n","N","eps_rel","Sample #","Time (s)")
println("─"^42)
for Nb in N_LIST
    r = table_results[Nb]
    @printf("%-4d  %-8.4f  %-14d  %-10.1f\n", Nb, r.eps_rel, r.n_calls, r.time_s)
end
serialize(joinpath(DIR_PROCESSED,"table1_chi$(chi_table)_$(rid).jls"), table_results)
println("Saved -> results/processed/table1_chi$(chi_table)_$(rid).jls")
