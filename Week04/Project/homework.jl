using DataFrames
using Plots
using Distributions
using CSV
using Dates
using LoopVectorization
using LinearAlgebra
using JuMP
using Ipopt
using Random
using StatsBase
using StatsPlots
using StateSpaceModels

include("../return_calculate.jl")
include("simulate_pca.jl")
include("ar1_sim.jl")


#Calculation of Risk Metrics
function VaR(a; alpha=0.05)
    x = sort(a)
    nup = convert(Int64,ceil(size(a,1)*alpha))
    ndn = convert(Int64,floor(size(a,1)*alpha))
    v = 0.5*(x[nup]+x[ndn])

    return -v
end


#problem 1
P0 = 100
σ = .1
rdist = Normal(0,σ)
simR = rand(rdist,1000000)

#brownian motion
#P1 = P0 + r
#P1 ~ N(P0,σ^2)
P1 = P0 .+ simR
println("Expect (μ,σ,skew,kurt)=($P0,$σ,0,0)")
println("($(mean(P1)),$(std(P1)),$(skewness(P1)),$(kurtosis(P1)))")


#Arithmetic returns
#P1 ~ N(P0,P0^2σ^2)
P1 = P0 .* (1 .+ simR)
println("Expect (μ,σ,skew,kurt)=($P0,$(σ*P0),0,0)")
println("($(mean(P1)),$(std(P1)),$(skewness(P1)),$(kurtosis(P1)))")

#Geometric brownian motion
#P1 ~ LN(P0,σ^2)
#E(P1) = exp(ln(P0) + (σ^2)/2)
#V(P1) = (exp(σ^2-1)*exp(2*ln(P0)+σ^2)
#Skew(P1) = (exp(σ^2+2)*sqrt(exp(σ^2)-1)
#Kurt(P1) = exp(4*σ^2) + 2*exp(3*σ^2) + 3*exp(2*σ^2) - 6

P1 = P0 * exp.(simR)
println("Expect (μ,σ,skew,kurt)=(
        $(exp(log(P0) + (σ^2)/2)),
        $(sqrt((exp(σ^2)-1)*exp(2*log(P0)+σ^2))),
        $((exp(σ^2)+2)*sqrt(exp(σ^2)-1)),
        $(exp(4*σ^2) + 2*exp(3*σ^2) + 3*exp(2*σ^2) - 6)")
println("(
        $(mean(P1)),
        $(std(P1)),
        $(skewness(P1)),
        $(kurtosis(P1)))")

#problem 2
# prices = CSV.read("Project/meta.csv",DataFrame)
# prices = CSV.read("DailyPrices.csv",DataFrame)
prices = CSV.read("C:/Users/dpazzula/OneDrive - dumac.duke.edu/Documents/FinTech-545-Spring2023/Week04/DailyPrices.csv",DataFrame)
returns = return_calculate(prices;dateColumn="Date")
# l = size(returns,1)
meta = returns[!,"META"]
# ometa = returns[(l-59):l,"META"]

meta = meta .- mean(meta)
s = std(meta)
d1 = Normal(0,s)
VaR1 = -quantile(d1,0.05)


s2 = ewCovar(reshape(meta,(length(meta),1)),0.94)
d2 = Normal(0,sqrt(s2[1]))
VaR2 = -quantile(d2,0.05)


#Fit each stock return to a T distribution
m, s, nu, d3 = fit_general_t(meta)
VaR3 = -quantile(d3,0.05)

#Historic VaR
VaR4 = VaR(meta)

#AR(1) VaR
ar1 = SARIMA(meta,order=(1,0,0),include_mean=true)

StateSpaceModels.fit!(ar1)
ar_sim =  ar1_simulation(meta,ar1.results.coef_table,randn(1000000))

VaR5 = VaR(ar_sim)

current = prices[size(prices,1),"META"]
println("Normal VaR  : \$$(current*VaR1) - $(100*VaR1)%")
println("ewNormal VaR: \$$(current*VaR2) - $(100*VaR2)%")
println("T Dist Var  : \$$(current*VaR3) - $(100*VaR3)%")
println("AR(1) VaR   : \$$(current*VaR5) - $(100*VaR5)%")
println("Historic VaR: \$$(current*VaR4) - $(100*VaR4)%")

# Normal VaR  : $11.671831158552395 - 6.560156967533285%
# ewNormal VaR: $16.259265461678925 - 9.138526093846895%
# T Dist Var  : $10.244594405783998 - 5.757978029129722%
# AR(1) VaR   : $11.66124497216526 - 6.55420699760622%
# Historic VaR: $10.388530174816395 - 5.838877180708287%



# hVaR = VaR(ometa)
# println("Historic Out of Sample VaR: \$$(current*hVaR) - $(100*hVaR)%")

# density(ometa,label="Out of Sample")
# p1 = density!(rand(d1,100000), label="Normal Distribution")

# density(ometa,label="Out of Sample")
# p2 = density!(rand(d2,100000), label="Normal Distribution ewVar")

# density(ometa,label="Out of Sample")
# p3 = density!(rand(d3,1000), label="T Distribution")

# density(ometa,label="Out of Sample")
# p4 = density!(meta, label="Historic Distribution")

# plot(
#     p1, p2, p3, p4,
#     layout=(2,2),
#     size=(1080,920),
#     title="Fitted vs Out of Sample"
# )

#Problem 3
portfolio = CSV.read("../Week04/Project/portfolio.csv",DataFrame)
# prices = CSV.read("DailyPrices.csv",DataFrame)
# returns = return_calculate(prices,dateColumn="Date")

covar = ewCovar(Matrix(returns[!,portfolio.Stock]),0.94)

current = prices[size(prices,1),portfolio.Stock]

nSim = 10000
sim = simulate_pca(covar,10000)
simReturns = DataFrame(sim,portfolio.Stock)

iterations = DataFrame(:iteration=>[x for x in 1:nSim])

# function pricing()
    values = crossjoin(portfolio,iterations)
    nVals = size(values,1)
    currentValue = Vector{Float64}(undef,nVals)
    simulatedValue = Vector{Float64}(undef,nVals)
    pnl = Vector{Float64}(undef,nVals)
    @inbounds begin
        Threads.@threads for i in 1:nVals
            @fastmath begin
                price = current[values.Stock[i]]
                currentValue[i] = values.Holding[i] * price
                simulatedValue[i] = values.Holding[i] * price*(1.0+simReturns[values.iteration[i],values.Stock[i]])
                pnl[i] = simulatedValue[i] - currentValue[i]
            end
        end
        values[!,:currentValue] = currentValue
        values[!,:simulatedValue] = simulatedValue
        values[!,:pnl] = pnl
        values
    end
# end

# st = Dates.now()
# values = pricing()
# el = Dates.now() - st



#Portfolio Level Metrics
#First sum by portfolio, iteration

gdf = groupby(values,[:Portfolio, :iteration])
portfolioValues = combine(gdf,
    :currentValue => sum => :currentValue,
    :pnl => sum => :pnl
)

gdf = groupby(portfolioValues,:Portfolio)
portfolioRisk = combine(gdf, 
    :currentValue => (x-> first(x,1)) => :currentValue,
    :pnl => (x -> VaR(x,alpha=0.05)) => :VaR95
)

#Total Metrics
gdf = groupby(values,:iteration)
#aggregate to totals per simulation iteration
totalValues = combine(gdf,
    :currentValue => sum => :currentValue,
    :pnl => sum => :pnl
)

totalRisk = combine(totalValues,
    :currentValue => (x-> first(x,1)) => :currentValue,
    :pnl => (x -> VaR(x,alpha=0.05)) => :VaR95
)
totalRisk[!,:Portfolio] = ["Total"]

VaRReport = vcat(portfolioRisk,totalRisk)

println(VaRReport)

# 4×3 DataFrame
#  Row │ Portfolio  currentValue  VaR95    
#      │ String     Float64       Float64  
# ─────┼───────────────────────────────────
#    1 │ A             2.9995e5    5627.94
#    2 │ B             2.94386e5   4416.36
#    3 │ C             2.70043e5   3699.24
#    4 │ Total         8.64378e5  13502.70