using Distributions
using Random
using StatsBase
using Statistics
using SpecialFunctions
using Roots
using QuadGK

import Distributions.logpdf, Distributions.mean, Distributions.median, Random.rand, Distributions.params, Distributions.pdf
import Distributions.cdf, Distributions.quantile, Distributions.var, Distributions.skewness, Distributions.kurtosis

struct SkewNormal{T<:Real} <: ContinuousUnivariateDistribution
    ξ::T
    ω::T
    α::T
    SkewNormal{T}(ξ::T, ω::T, α::T) where {T<:Real} = new{T}(ξ, ω, α)
end

function SkewNormal(ξ::T, ω::T, α::T; check_args::Bool=true) where {T <: Real}
    Distributions.@check_args SkewNormal (ω, ω >= zero(ω))
    return SkewNormal{T}(ξ, ω, α)
end

#### TODO: Outer constructors handling passing Ints.

####

Distributions.@distr_support SkewNormal -Inf Inf

params(d::SkewNormal) = (d.ξ, d.ω, d.α)

const p2 = 2*π
const ipi2 = 2/π

function OwensT(h,a)
    f(x) = (exp(-0.5 * h^2 * (1+x^2)))/(1+x^2)
    integ = quadgk(f,0,a)[1]
    return integ/p2
end

pdf(d::SkewNormal, x::Real) = 2/d.ω * pdf(Normal(),(x-d.ξ)/d.ω) * cdf(Normal(),d.α * (x-d.ξ)/d.ω)
cdf(d::SkewNormal, x::Real) = cdf(Normal(),(x-d.ξ)/d.ω) - 2*OwensT((x-d.ξ)/d.ω,d.α)
quantile(d::SkewNormal, u::Real) = find_zero(x->cdf(d,x)-u,0.0)

function logpdf(d::SkewNormal, x::Real) 
    log(pdf(d,x))
end

rand(rng::AbstractRNG, d::SkewNormal{T}) where {T} = quantile.(d,randn(rng, float(T)))

_δ(d::SkewNormal) = d.α / (sqrt(1+d.α^2))
mean(d::SkewNormal) = d.ξ + d.ω*_δ(d)*sqrt(ipi2)
# median(d::SkewNormal) = d.ξ - d.λ * sinh(d.γ/d.δ)
var(d::SkewNormal) = d.ω^2 * (1 - (ipi2 * _δ(d)))

function skewness(d::SkewNormal)
    first = 2 - ipi2
    second = (_δ(d)*sqrt(ipi2))^3
    third = sqrt(1 - ipi2*_δ(d)^2)^3
    return first * second / third
end

function kurtosis(d::SkewNormal)
    first = p2 - 6
    second = (_δ(d)*sqrt(ipi2))^4
    third = (1 - ipi2*_δ(d)^2)^2
    return first * second / third
end

function cdf(d::NormalInverseGaussian,x::Real)
    function f(_x)
        out = pdf(d,_x)
        isnan(out) ? 0.0 : out
    end
    integ = quadgk(f,-10,x)[1]
end

function quantile(d::NormalInverseGaussian,u::Real)
    st = quantile(Normal(mean(d),std(d)),u)
    # st = 0.0
    try
        return find_zero(x->cdf(d,x)-u,st)
    catch e
        try
            return find_zero(x->cdf(d,x)-u+1e-6,st)
        catch
            return NaN
        end
    end
    
end