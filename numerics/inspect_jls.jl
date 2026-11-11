# inspect_jls.jl — summarize every .jls file in the package.
# Run from the Submission_package directory with the project env:
#   julia --project=env inspect_jls.jl            # all files
#   julia --project=env inspect_jls.jl 05_qcnn    # one subfolder
# Requires the same Julia minor version used to write the files (1.11.x).

using Serialization, Printf

function describe(x, indent=0; maxdepth=2)
    pad = "  "^indent
    if x isa Dict
        println(pad, "Dict with $(length(x)) keys:")
        for (k, v) in sort(collect(x); by=string∘first)
            print(pad, "  [", repr(k), "] => ")
            summarize_val(v, indent + 1; maxdepth=maxdepth - 1)
        end
    elseif x isa AbstractVector
        println(pad, "$(typeof(x)) length=$(length(x))")
        !isempty(x) && maxdepth > 0 && (print(pad, "  first: "); summarize_val(first(x), indent + 1; maxdepth=maxdepth - 1))
    else
        summarize_val(x, indent; maxdepth=maxdepth)
    end
end

function summarize_val(v, indent; maxdepth=1)
    pad = "  "^indent
    if v isa Number || v isa String || v isa Symbol
        println(repr(v))
    elseif v isa AbstractArray
        it = eltype(v) <: Number && !isempty(v) ? @sprintf(" range=[%.3g, %.3g]", minimum(v), maximum(v)) : ""
        println("$(typeof(v)) size=$(size(v))", it)
    elseif v isa Dict && maxdepth > 0
        println()
        describe(v, indent + 1; maxdepth=maxdepth)
    else
        println(typeof(v))
    end
end

root = length(ARGS) >= 1 ? ARGS[1] : "."
files = sort(filter(f -> endswith(f, ".jls"), [joinpath(r, f) for (r, _, fs) in walkdir(root) for f in fs]))
println("Found $(length(files)) .jls files under $(abspath(root))\n")
for f in files
    println("═"^70, "\n", f, "  ($(filesize(f)) bytes)")
    try
        describe(deserialize(f))
    catch e
        println("  !! failed to load: ", sprint(showerror, e)[1:min(end, 200)])
        println("  (usually a Julia-version or missing-type issue; open with the original env)")
    end
end
