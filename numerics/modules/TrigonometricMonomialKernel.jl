"""
TrigonometricMonomialKernel.jl
==============================
Truncated trigonometric monomial kernel (TMK) learner for functions
`f : [-pi, pi]^d -> R`.

The implementation follows the kernel estimator used in
`Efficient_learning_for.pdf`:

    kappa_Lambda(x, x') = sum_{omega in C(Lambda)}
        2^||omega||_0 Phi_omega(x) Phi_omega(x')

with `omega_i in {0, ±1}` and `||omega||_0 <= Lambda`.
For exact oracle labels this is a direct Monte-Carlo Fourier coefficient
estimator; noisy/shadow labels can be passed in the same way.
"""
module TrigonometricMonomialKernel

using Random

export TMKModel,
       truncated_frequencies,
       sample_uniform_inputs,
       fit_tmk,
       predict,
       relative_l2_error

struct TMKModel
    frequencies::Vector{Vector{Int8}}
    coeffs::Vector{Float64}
    lambda::Int
    nsamples::Int
    d::Int
end

function truncated_frequencies(d::Integer, lambda::Integer)
    d < 1 && throw(ArgumentError("d must be positive"))
    lambda < 0 && throw(ArgumentError("lambda must be non-negative"))

    d_int = Int(d)
    lambda_int = min(Int(lambda), d_int)
    freqs = Vector{Vector{Int8}}()
    push!(freqs, zeros(Int8, d_int))

    support = Int[]

    function append_signed_support!(weight::Int)
        for mask in 0:(1 << weight)-1
            omega = zeros(Int8, d_int)
            @inbounds for j in 1:weight
                omega[support[j]] = ((mask >> (j - 1)) & 1) == 1 ? Int8(1) : Int8(-1)
            end
            push!(freqs, omega)
        end
    end

    function choose_support!(start::Int, left::Int, weight::Int)
        if left == 0
            append_signed_support!(weight)
            return
        end
        last_start = d_int - left + 1
        for i in start:last_start
            push!(support, i)
            choose_support!(i + 1, left - 1, weight)
            pop!(support)
        end
    end

    for weight in 1:lambda_int
        empty!(support)
        choose_support!(1, weight, weight)
    end

    return freqs
end

function sample_uniform_inputs(rng::AbstractRNG, n::Integer, d::Integer;
                               lower::Real=-pi, upper::Real=pi)
    n < 1 && throw(ArgumentError("n must be positive"))
    d < 1 && throw(ArgumentError("d must be positive"))
    width = Float64(upper - lower)
    return [Float64(lower) .+ width .* rand(rng, Int(d)) for _ in 1:Int(n)]
end

function _feature_value(omega::AbstractVector{Int8}, x::AbstractVector{<:Real})
    length(omega) == length(x) || throw(DimensionMismatch("omega and x lengths differ"))
    value = 1.0
    @inbounds for i in eachindex(omega)
        oi = omega[i]
        if oi == 1
            value *= cos(x[i])
        elseif oi == -1
            value *= sin(x[i])
        end
    end
    return value
end

function _feature_matrix(xs::AbstractVector, frequencies::AbstractVector)
    n = length(xs)
    m = length(frequencies)
    Phi = Matrix{Float64}(undef, n, m)
    @inbounds for i in 1:n, j in 1:m
        Phi[i, j] = _feature_value(frequencies[j], xs[i])
    end
    return Phi
end

function fit_tmk(xs::AbstractVector, y::AbstractVector{<:Real}; lambda::Integer)
    length(xs) == length(y) || throw(DimensionMismatch("xs and y lengths differ"))
    isempty(xs) && throw(ArgumentError("training set cannot be empty"))

    d = length(xs[1])
    all(x -> length(x) == d, xs) || throw(DimensionMismatch("all inputs must have the same length"))

    frequencies = truncated_frequencies(d, lambda)
    Phi = _feature_matrix(xs, frequencies)
    weights = [2.0 ^ count(!=(Int8(0)), omega) for omega in frequencies]
    coeffs = weights .* (transpose(Phi) * Float64.(y)) ./ length(y)
    return TMKModel(frequencies, coeffs, min(Int(lambda), d), length(y), d)
end

function fit_tmk(f::Function, d::Integer; lambda::Integer, nsamples::Integer,
                 rng::AbstractRNG=MersenneTwister(0),
                 lower::Real=-pi, upper::Real=pi)
    xs = sample_uniform_inputs(rng, nsamples, d; lower, upper)
    y = [Float64(f(x)) for x in xs]
    return fit_tmk(xs, y; lambda)
end

function predict(model::TMKModel, x::AbstractVector{<:Real})
    length(x) == model.d || throw(DimensionMismatch("input length does not match model dimension"))
    s = 0.0
    @inbounds for j in eachindex(model.frequencies)
        s += model.coeffs[j] * _feature_value(model.frequencies[j], x)
    end
    return s
end

function relative_l2_error(f::Function, model::TMKModel, xs::AbstractVector)
    num2 = 0.0
    den2 = 0.0
    for x in xs
        y = Float64(f(x))
        yh = predict(model, x)
        num2 += (y - yh)^2
        den2 += y^2
    end
    return sqrt(num2) / max(sqrt(den2), eps())
end

end # module TrigonometricMonomialKernel
