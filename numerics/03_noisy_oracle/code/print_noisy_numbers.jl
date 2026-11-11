# Prints the numbers quoted in the shot-noise subsection (Sec. VII) from the
# alpha-sweep results. Run from 03_noisy_oracle/: julia --project=../../env code/print_noisy_numbers.jl
using Serialization, Statistics, Printf

as = deserialize(joinpath(@__DIR__, "..", "results", "alpha_sweep_N9_chi12.jls"))

println("── amplification factor ∏‖V+‖ (Fig. oversampling b) ──")
for a in sort(collect(keys(as["amp_curve"])))
    @printf("alpha=%d  amp=%.3e   (orders: %.1f)\n", a, as["amp_curve"][a], log10(as["amp_curve"][a]))
end

AR = as["results"]
println("\n── alpha empirical error (Fig. oversampling a) ──")
for k in sort(collect(keys(AR)); by=x->(x[1], isinf(x[2]) ? Inf : x[2]))
    es = [r["eps_rel"] for r in AR[k]]
    @printf("alpha=%d S=%-8s  eps_rel=%.3e  (n=%d)\n", k[1], string(k[2]), mean(es), length(es))
end

g = mean(r["eps_rel"] for r in AR[(1, 1e2)]) / mean(r["eps_rel"] for r in AR[(2, 1e2)])
@printf("\nalpha=1 error larger than alpha=2 by ~%.1f orders of magnitude (at S=1e2)\n", log10(g))
