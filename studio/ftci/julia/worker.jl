# Deliberately small, dependency-free line protocol around the actual TCI engine.
# Python owns the target oracle, so user functions never become Julia source.
using TensorCrossInterpolation
using Random
const TCI = TensorCrossInterpolation

println("READY")
flush(stdout)

function main()
    dims = parse.(Int, split(readline(stdin), ','))
    tolerance = parse(Float64, readline(stdin))
    maxbond = parse(Int, readline(stdin))
    seed = parse(Int, readline(stdin))
    initialpivot = parse.(Int, split(readline(stdin), ','))
    Random.seed!(seed)

    function oracle(indices)
        println("Q\t", join(indices, ','))
        flush(stdout)
        response = readline(stdin)
        startswith(response, "ERR\t") && error(response[5:end])
        value = parse(Float64, response)
        isfinite(value) || error("Oracle returned a non-finite value")
        return value
    end

    tci, ranks, errors = TCI.crossinterpolate2(
        Float64, oracle, dims, [initialpivot];
        tolerance=tolerance, maxbonddim=maxbond, pivotsearch=:full,
        maxiter=40, verbosity=0,
    )
    cores = TCI.sitetensors(TCI.TensorTrain(tci))
    for core in cores
        println("CORE\t", join(size(core), ','), '\t', join(vec(core), ','))
    end
    println("ERRORS\t", join(errors, ','))
    println("RANKS\t", join(ranks, ','))
    println("DONE")
    flush(stdout)
end

try
    main()
catch exception
    println("ERROR\t", replace(sprint(showerror, exception), '\n' => ' '))
    flush(stdout)
    exit(1)
end
