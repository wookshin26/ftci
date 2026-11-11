using Dates, Plots

function paper_plot_defaults!()
    default(
        fontfamily = "Computer Modern",
        linewidth = 2.2,
        markersize = 4.5,
        guidefontsize = 12,
        tickfontsize = 10,
        legendfontsize = 9,
        titlefontsize = 12,
        framestyle = :box,
        grid = true,
        dpi = 300,
    )
end

function mean_numeric(values)
    vals = Float64[v for v in values if v isa Number && isfinite(Float64(v))]
    isempty(vals) ? NaN : sum(vals) / length(vals)
end

function mean_row_field(rows, field::Symbol, predicate)
    mean_numeric(get(r, field, NaN) for r in rows if predicate(r))
end

function save_vector_formats(plot_obj, pathbase::AbstractString)
    mkpath(dirname(pathbase))
    pdf_path = pathbase * ".pdf"
    svg_path = pathbase * ".svg"
    savefig(plot_obj, pdf_path)
    savefig(plot_obj, svg_path)
    return (; pdf_path, svg_path)
end

function save_spread_truncation_figure(rows, cfg;
                                       outdir::AbstractString=OUTDIR,
                                       stamp::AbstractString=Dates.format(now(), "yyyymmdd_HHMMSS"))
    paper_plot_defaults!()
    Ls = collect(cfg.Ls)

    p_coeff = plot(
        xlabel = "TMK truncation Lambda",
        ylabel = "coefficient energy kept",
        title = "(a) Low-weight coefficient mass",
        ylims = (0.0, 1.02),
        legend = :bottomright,
        size = (470, 330),
    )

    p_trunc = plot(
        xlabel = "TMK truncation Lambda",
        ylabel = "best relative L2 error",
        title = "(b) Best possible truncation error",
        yscale = :log10,
        legend = :topright,
        size = (470, 330),
    )

    for L in Ls
        lambdas = collect(0:L)
        coeff_kept = [
            mean_row_field(rows, :coeff_energy_fraction,
                r -> r.method == "TMK_BEST_TRUNCATION" && r.L == L && r.lambda == lambda)
            for lambda in lambdas
        ]
        trunc_err = [
            max(mean_row_field(rows, :best_trunc_rel_l2_error,
                r -> r.method == "TMK_BEST_TRUNCATION" && r.L == L && r.lambda == lambda), 1e-16)
            for lambda in lambdas
        ]

        plot!(p_coeff, lambdas, coeff_kept; marker=:circle, label="L=$L")
        plot!(p_trunc, lambdas, trunc_err; marker=:circle, label="L=$L")
    end

    fig = plot(p_coeff, p_trunc; layout=(1, 2), size=(940, 330))
    return save_vector_formats(fig, joinpath(outdir, "spread_truncation_$(cfg.mode)_$stamp"))
end

function write_threshold_summary(path::AbstractString, threshold_rows)
    mkpath(dirname(path))
    header = ["threshold", "L", "method", "mean_samples", "reached_count", "total_count", "max_tmk_samples"]
    open(path, "w") do io
        println(io, join(header, ","))
        for row in threshold_rows
            println(io, join((csv_escape(get(row, Symbol(h), "")) for h in header), ","))
        end
    end
end

function save_threshold_samples_figure(rows, cfg;
                                       thresholds=[1e-1, 1e-2, 1e-3],
                                       outdir::AbstractString=OUTDIR,
                                       stamp::AbstractString=Dates.format(now(), "yyyymmdd_HHMMSS"))
    paper_plot_defaults!()
    summary = sample_threshold_rows(rows, thresholds)
    Ls = collect(cfg.Ls)
    panels = Any[]

    for threshold in thresholds
        p = plot(
            xlabel = "L",
            ylabel = "oracle samples",
            title = "error <= $(threshold)",
            yscale = :log10,
            legend = :topleft,
            size = (330, 320),
        )

        ftci_y = [
            mean_row_field(summary, :mean_samples,
                r -> r.method == "FTCI" && r.L == L && r.threshold == threshold)
            for L in Ls
        ]
        tmk_y = [
            mean_row_field(summary, :mean_samples,
                r -> r.method == "TMK" && r.L == L && r.threshold == threshold)
            for L in Ls
        ]
        plot!(p, Ls, ftci_y; marker=:circle, label="FTCI")
        plot!(p, Ls, tmk_y; marker=:square, label="TMK reached")

        censored_x = Int[]
        censored_y = Float64[]
        for L in Ls
            total = mean_row_field(summary, :total_count,
                r -> r.method == "TMK" && r.L == L && r.threshold == threshold)
            reached = mean_row_field(summary, :reached_count,
                r -> r.method == "TMK" && r.L == L && r.threshold == threshold)
            max_samples = mean_row_field(summary, :max_tmk_samples,
                r -> r.method == "TMK" && r.L == L && r.threshold == threshold)
            if !isnan(total) && reached < total
                push!(censored_x, L)
                push!(censored_y, max(max_samples, 1.0))
            end
        end
        isempty(censored_x) || scatter!(p, censored_x, censored_y;
            marker=:utriangle, markersize=6, color=:red,
            label="TMK not reached by max n")

        push!(panels, p)
    end

    fig = plot(panels...; layout=(1, length(panels)), size=(330 * length(panels), 320))
    paths = save_vector_formats(fig, joinpath(outdir, "threshold_samples_$(cfg.mode)_$stamp"))
    csv_path = joinpath(outdir, "threshold_samples_$(cfg.mode)_$stamp.csv")
    write_threshold_summary(csv_path, summary)
    return (; paths..., csv_path)
end
