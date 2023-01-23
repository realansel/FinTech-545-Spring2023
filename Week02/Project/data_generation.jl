using Distributions
using StatsBase
using DataFrames
using CSV
using Plots
using JuMP
using Ipopt
using SpecialFunctions
using HypothesisTests

#Problem 1
# Sample a normal distribution. both skew and kurtosis = 0
samples = 1000
skews = Vector{Float64}(undef,samples)
kurts = Vector{Float64}(undef,samples)
for i in 1:samples
    r = randn(10)
    skews[i] = skewness(r)
    kurts[i] = kurtosis(r)
end

println(OneSampleTTest(skews,0.0))
# Test summary:
#     outcome with 95% confidence: fail to reject h_0
#     two-sided p-value:           0.2937

println(OneSampleTTest(kurts,0.0))
# Test summary:
#     outcome with 95% confidence: reject h_0
#     two-sided p-value:           <1e-07

#Kurtosis is biased in small samples.  Skewness is harder to prove.

#Test another way
function first4Moments(sample)

    n = size(sample,1)

    #mean
    μ_hat = sum(sample)/n

    #remove the mean from the sample
    sim_corrected = sample .- μ_hat
    cm2 = sim_corrected'*sim_corrected/n

    #variance
    σ2_hat = sim_corrected'*sim_corrected/(n-1)

    #skew
    skew_hat = sum(sim_corrected.^3)/n/sqrt(cm2*cm2*cm2)

    #kurtosis
    kurt_hat = sum(sim_corrected.^4)/n/cm2^2

    excessKurt_hat = kurt_hat - 3

    return μ_hat, σ2_hat, skew_hat, excessKurt_hat
end

# lots of samples and compare to a biased estimate
samples = 1000000
skews = Vector{Float64}(undef,samples)
for i in 1:samples
    r = randn(10)
    skews[i] = skewness(r) - first4Moments(r)[3]
end

mx, mn = extrema(skews)
#(-1.7763568394002505e-15, 1.7763568394002505e-15)
#within floating point error -- likely skew is biased but unable to prove 
#statistically

#problem 2
n = Normal(0,1)
t = TDist(5)
mv = MvNormal([0.0,0.0], [[1.0, .5] [.5, 1.0]])
sim = rand(mv,100)
x = sim[1,:]
y = quantile.(t,cdf(n,sim[2,:]))
# df = DataFrame(:x => x, :y=>y)
# CSV.write("Project/problem2.csv",df)

prob2 = CSV.read("Project/problem2.csv",DataFrame)

Y = prob2.y
X = [ones(100) prob2.x]

B = inv(X'*X)*X'*Y
e = Y - X*B
println("Kurtosis of Error: $(kurtosis(e))")
println("Skewness of Error $(skewness(e))")
# Kurtosis of Error: 3.193101000956875
# Skewness of Error -0.2672665855287958
# Error is too kurtotic to be normally distributed

function normal_ll(s, b...)
    n = size(Y,1)
    beta = collect(b)
    xm = Y - X*beta
    s2 = s*s
    ll = -n/2 * log(s2 * 2 * π) - xm'*xm/(2*s2)
    return ll
end

function __T_loglikelihood(mu,s,nu,x)
    n = size(x,1)
    np12 = (nu + 1.0)/2.0

    mess = loggamma(np12) - loggamma(nu/2.0) - log(sqrt(π*nu)*s)
    xm = ((x .- mu)./s).^2 * (1/nu) .+ 1
    innerSum = sum(log.(xm))
    ll = n*mess - np12*innerSum
    return ll
end

function t_ll(s,nu, b...)
    td = TDist(nu)
    beta = collect(b)
    xm = (Y - X*beta)

    ll = __T_loglikelihood(0.0,s,nu,xm)
    return ll
end

mle = Model(Ipopt.Optimizer)
set_silent(mle)

@variable(mle, beta[i=1:2],start=0)
@variable(mle, σ >= 0.0, start = 1.0)
@variable(mle, ν >= 0.0, start = 10.0)

register(mle,:normLL,3,normal_ll;autodiff=true)

register(mle,:tLL,4,t_ll;autodiff=true)

@NLobjective(
    mle,
    Max,
    normLL(σ,beta...)
)
optimize!(mle)
normal_beta = value.(beta)
normal_s = value(σ)
normalLL = normal_ll(normal_s,normal_beta...)
nAIC = 6 - 2*normalLL
println("Normal Betas: ", normal_beta)
println("Normal S: ", normal_s)
println("Normal LL:", normalLL)
println("Normal AIC: ",nAIC)



mle = Model(Ipopt.Optimizer)
set_silent(mle)

@variable(mle, beta[i=1:2],start=0)
@variable(mle, σ >= 0.0, start = 1.0)
@variable(mle, 100 >= ν >= 3.0, start = 10.0)

register(mle,:tLL,4,t_ll;autodiff=true)

@NLobjective(
    mle,
    Max,
    tLL(σ,ν, beta...)
)

optimize!(mle)
t_beta = value.(beta)
t_s = value(σ)
t_nu = value(ν)
tLL = t_ll(t_s, t_nu, t_beta...)
tAIC = 8 - 2*tLL
println("T Betas: ", t_beta)
println("T S: ", t_s)
println("T df: ", t_nu)
println("T LL:", tLL)
println("T AIC: ", tAIC)


#Problem 3

using PyCall
using Conda
# ENV["Python"] = ""
# using Pkg
# Pkg.build("python")
# Conda.add("statsmodels")
sm = pyimport("statsmodels.api")

#AR1
#y_t = 1.0 + 0.5*y_t-1 + e, e ~ N(0,0.1)
n = 1000
burn_in = 50
y = Vector{Float64}(undef,n)

yt_last = [1.0,1.0]
d = Normal(0,0.1)
e = rand(d,n+burn_in)

for i in 1:(n+burn_in)
    global yt_last
    y_t = 1.0 + 0.5*yt_last[1] + .25*yt_last[2] + e[i]
    yt_last[2] = yt_last[1]
    yt_last[1] = y_t
    if i > burn_in
        y[i-burn_in] = y_t
    end
end

# Python call for ARIMA fitting and simulation
m = sm.tsa.arima.ARIMA(y, order=(2, 0, 0))
res = m.fit()
println(res.summary())

r = fill(0.0,(1000,2))
sim = [res.simulate(1,measurement_shocks=rand(Normal(0,1),1),anchor="end",pretransformed_measurement_shocks=false)[1] for i in 1:1000]
