using Distributions
using Random
using StatsBase
using Statistics
# using JuMP
# using Ipopt
# using HypothesisTests

import Distributions.logpdf, Distributions.mean, Distributions.median, Random.rand, Distributions.params, Distributions.pdf
import Distributions.cdf, Distributions.quantile, Distributions.var, Distributions.skewness, Distributions.kurtosis

struct JohnsonSU{T<:Real} <: ContinuousUnivariateDistribution
    γ::T
    ξ::T
    δ::T
    λ::T
    JohnsonSU{T}(γ::T, ξ::T, δ::T, λ::T) where {T<:Real} = new{T}(γ, ξ, δ, λ)
end

function JohnsonSU(γ::T, ξ::T, δ::T, λ::T; check_args::Bool=true) where {T <: Real}
    Distributions.@check_args JohnsonSU (δ, δ >= zero(δ))
    Distributions.@check_args JohnsonSU (λ, λ >= zero(λ))
    return JohnsonSU{T}(γ, ξ, δ, λ)
end

#### TODO: Outer constructors handling passing Ints.

####

Distributions.@distr_support JohnsonSU -Inf Inf

params(d::JohnsonSU) = (d.γ, d.ξ, d.δ, d.λ)

rand(rng::AbstractRNG, d::JohnsonSU{T}) where {T} = d.λ*sinh( (randn(rng, float(T)) - d.γ) / d.δ) + d.ξ
    
mean(d::JohnsonSU) = d.ξ - d.λ * exp( (d.δ^-2)/2) * sinh(d.γ/d.δ)
median(d::JohnsonSU) = d.ξ - d.λ * sinh(d.γ/d.δ)
ω(d::JohnsonSU) = exp(d.δ^(-2))
Ω(d::JohnsonSU) = d.γ/d.δ
var(d::JohnsonSU) = ((d.λ)^2)/2.0 * (ω(d)-1.0) * (ω(d)*cosh(2.0*Ω(d))+1.0)

function skewness(d::JohnsonSU)
    _ω = ω(d)
    _Ω = Ω(d)
    top = (d.λ^3) * sqrt(_ω) * (_ω-1)^2
    top = top * (_ω*(_ω+2)sinh(3*_Ω) + 3*sinh(_Ω))
    bottom = 4 * sqrt(var(d))^3
    -top/bottom
end

function kurtosis(d::JohnsonSU)
    _Ω = Ω(d)
    _ω = ω(d)
    A = _ω^2 * (_ω^4 + 2*_ω^3 + 3*_ω^2 - 3) * cosh(4*_Ω)
    B = 4*_ω^2 * (_ω+2) * cosh(2*_Ω)
    C = 3* (2*_ω + 1)
    top = d.λ^4 * (_ω - 1)^2 * (A+B+C)
    bottom = 8 * var(d)^2
    top/bottom - 3
end

const s2p = sqrt(2*π)


function logpdf(d::JohnsonSU, x::Real) 
    p1 = log(d.δ / (d.λ * s2p))
    p2 = -log(sqrt(1 + ( (x - d.ξ) / d.λ)^2 ))
    p3 = -0.5 * (d.γ + d.δ*asinh((x - d.ξ) / d.λ))^2
    p1+p2+p3
end

pdf(d::JohnsonSU, x::Real) = exp(logpdf(d,x))
cdf(d::JohnsonSU, x::Real) = cdf(Normal(), d.γ + d.δ*asinh((x - d.ξ) / d.λ))
quantile(d::JohnsonSU, x::Real) = d.λ*sinh( (quantile(Normal(),x) - d.γ) / d.δ) + d.ξ

# d = JohnsonSU(1.0,1.1,2.0,1.5)

# function myll(γ, ξ, δ, λ)
#     n = size(x,1)
#     d = JohnsonSU(γ, ξ, δ, λ)
#     ll = sum(logpdf.(d,x))
#     return ll
# end

# diffs = fill(-999.0, (100,4))

# Threads.@threads for i in 1:100
#     global x
#     rng = MersenneTwister(i)
#     x = rand(rng,d,100000)

#     #MLE Optimization problem
#         mle = Model(Ipopt.Optimizer)
#         set_silent(mle)

#         @variable(mle, γ, start = 0.0)
#         @variable(mle, ξ, start = 0.0)
#         @variable(mle, δ >= 0.0, start = 1.0)
#         @variable(mle, λ >= 0.0, start = 1.0)

#         register(mle,:ll,4,myll;autodiff=true)

#         @NLobjective(
#             mle,
#             Max,
#             ll(γ, ξ, δ, λ)
#         )
#     ##########################

#     try
#         optimize!(mle)

#         diffs[i,1] = value(γ) - 1.0
#         diffs[i,2] = value(ξ) - 1.1
#         diffs[i,3] = value(δ) - 2.0
#         diffs[i,4] = value(λ) - 1.5
#     catch
#     end

# end

# diffs = diffs[diffs[:,1] .!= -999.0,:]

# OneSampleTTest(diffs[:,1],0.0)
# OneSampleTTest(diffs[:,2],0.0)
# OneSampleTTest(diffs[:,3],0.0)
# OneSampleTTest(diffs[:,4],0.0)
