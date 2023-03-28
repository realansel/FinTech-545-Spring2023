using CSV
using DataFrames
using Dates
using Roots
using Distributions
using Plots
using Random
using StateSpaceModels
using Distributions
using StatsBase
using LinearAlgebra
using LoopVectorization

include("../../library/gbsm.jl")
include("../../library/RiskStats.jl")
include("../../library/return_calculate.jl")

#Problem #1
currentPrice = 165
currentDate=Date("03/03/2023",dateformat"mm/dd/yyyy")
rf = 0.0425
dy = 0.0053
DaysYear = 365

expirationDate = Date("03/17/2023",dateformat"mm/dd/yyyy")
ttm = (expirationDate - currentDate).value/DaysYear

strike = 165
iv = [i/100 for i in 10:2:80]
#gbsm(call::Bool, underlying, strike, ttm, rf, b, ivol)
call_vals = [v.value for v in gbsm.(true,currentPrice,strike,ttm,rf,rf-dy,iv)]
put_vals  = [v.value for v in gbsm.(false,currentPrice,strike,ttm,rf,rf-dy,iv)]

plot(
    plot(call_vals,iv, label="Call Values"),
    plot(put_vals,iv, label="Put Values",linecolor=:red),
    layout=(1,2)
)


#Problem #2
currentPrice = 151.03
options = CSV.read("Project/AAPL_Options.csv",DataFrame)

options[!,:Expiration] = Date.(options.Expiration,dateformat"mm/dd/yyyy")


n = length(options.Expiration)

#list comprehension for TTM
options[!,:ttm] = [(options.Expiration[i] - currentDate).value / DaysYear for i in 1:n]

#gbsm(call::Bool, underlying, strike, ttm, rf, b, ivol)
iv = [find_zero(x->gbsm(options.Type[i]=="Call",currentPrice,options.Strike[i],options.ttm[i],rf,rf-dy,x).value-options[i,"Last Price"],.2) for i in 1:n]
options[!,:ivol] = iv
options[!,:gbsm] = [v.value for v in gbsm.(options.Type.=="Call",currentPrice,options.Strike,options.ttm,rf,rf-dy,options.ivol)]


calls = options.Type .== "Call"
puts = [!calls[i] for i in 1:n]

plot(options.Strike[calls],options.ivol[calls],label="Call Implied Vol",title="Implied Volatilities")
plot!(options.Strike[puts],options.ivol[puts],label="Put Implied Vol",linecolor=:red)
vline!([currentPrice],label="Current Price",linestyle=:dash,linecolor=:purple)


#problem 3

currentS=151.03
returns = return_calculate(CSV.read("Project/DailyPrices.csv",DataFrame)[!,[:Date,:AAPL]],method="LOG",dateColumn="Date")[!,:AAPL]
returns = returns .- mean(returns)
sd = std(returns)
current_dt = Date(2023,3,3)

portfolio = CSV.read("Project/problem3.csv", DataFrame)

#Convert Expiration Date for Options to Date object
portfolio[!,:ExpirationDate] = [
    portfolio.Type[i] == "Option" ? Date(portfolio.ExpirationDate[i],dateformat"mm/dd/yyyy") : missing
    for i in 1:size(portfolio,1) ]

# Calculate implied Vol
portfolio[!, :ImpVol] = [
    portfolio.Type[i] == "Option" ?
    find_zero(x->gbsm(portfolio.OptionType[i]=="Call",
                        currentS,
                        portfolio.Strike[i],
                        (portfolio.ExpirationDate[i]-current_dt).value/365,
                        rf,rf-dy,x).value
                -portfolio.CurrentPrice[i],.2)    : missing     
    for i in 1:size(portfolio,1)
]

#Simulate Returns
nSim = 10000
fwdT = 10

#Fit the AR(1) model
ar1 = SARIMA(returns,order=(1,0,0),include_mean=true)
StateSpaceModels.fit!(ar1)
print_results(ar1)

function ar1_simulation(y,coef_table,innovations; ahead=1)
    m = coef_table.coef[findfirst(r->r == "mean",coef_table.names)]
    a1 = coef_table.coef[findfirst(r->r == "ar_L1",coef_table.names)]
    s = sqrt(coef_table.coef[findfirst(r->r == "sigma2_η",coef_table.names)])

    l = length(y)
    n = convert(Int64,length(innovations)/ahead)

    out = fill(0.0,(ahead,n))

    y_last = y[l] - m
    for i in 1:n
        yl = y_last
        next = 0.0
        for j in 1:ahead
            next = a1*yl + s*innovations[(i-1)*ahead + j]
            yl = next
            out[j,i] = next
        end
    end

    out = out .+ m
    return out
end

#simulate nSim paths fwdT days ahead.
arSim = ar1_simulation(returns,ar1.results.coef_table,randn(fwdT*nSim),ahead=fwdT)

# Sum returns since these are log returns and convert to final prices
simReturns = sum.(eachcol(arSim))
simPrices = currentS .* exp.(simReturns)


iteration = [i for i in 1:nSim]
values = crossjoin(portfolio, DataFrame(:iteration=>iteration))
nVals = size(values,1)

#Set the forward ttm
values[!,:fwd_ttm] = [
    values.Type[i] == "Option" ? (values.ExpirationDate[i]-current_dt-Day(fwdT)).value/365 : missing
    for i in 1:nVals
]

#Calculate values of each position
simulatedValue = Vector{Float64}(undef,nVals)
currentValue = Vector{Float64}(undef,nVals)
pnl = Vector{Float64}(undef,nVals)
for i in 1:nVals
    simprice = simPrices[values.iteration[i]]
    currentValue[i] = values.Holding[i]*values.CurrentPrice[i]
    if values.Type[i] == "Option"
        simulatedValue[i] = values.Holding[i]*gbsm(values.OptionType[i]=="Call",simprice,values.Strike[i],values.fwd_ttm[i],rf,rf-dy,values.ImpVol[i]).value
    elseif values.Type[i] == "Stock"
        simulatedValue[i] = values.Holding[i]*simprice
    end
    pnl[i] = simulatedValue[i] - currentValue[i]
end

values[!,:simulatedValue] = simulatedValue
values[!,:pnl] = pnl
values[!,:currentValue] = currentValue


risk = aggRisk(values,[:Portfolio])
# 9×14 DataFrame
#  Row │ Portfolio     currentValue  VaR95     ES95      VaR99     ES99      Standard_Dev  min        max       mean        VaR95_Pct  VaR99_Pct   ES95_Pct    ES99_Pct   
#      │ String15      Float64       Float64   Float64   Float64   Float64   Float64       Float64    Float64   Float64     Float64    Float64     Float64     Float64    
# ─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#    1 │ Straddle             11.65   1.37751   1.38716   1.39119   1.39148       3.85855   -1.39162  34.3681    1.5666     0.118242    0.119415    0.11907     0.11944
#    2 │ SynLong               1.95  16.0563   20.0026   22.5418   25.6814       10.2293   -37.8521   44.062     0.07706    8.234      11.5599     10.2577     13.1699
#    3 │ CallSpread            4.59   3.86897   4.17726   4.37661   4.46646       2.52177   -4.5874    5.35245  -0.0761469  0.842914    0.953511    0.910078    0.973084
#    4 │ PutSpread             3.01   2.64529   2.81039   2.90321   2.94738       2.22515   -3.00737   6.89661   0.284136   0.878832    0.964522    0.933683    0.979195
#    5 │ Stock               151.03  15.8152   19.7382   22.2612   25.3908       10.2412   -37.5483   44.4116    0.280639   0.104715    0.147396    0.13069     0.168118
#    6 │ Call                  6.8    6.01319   6.35759   6.57743   6.67213       6.13689   -6.79739  39.215     0.821832   0.884292    0.967269    0.93494     0.981196
#    7 │ Put                   4.85   4.387     4.60048   4.71941   4.77403       4.70124   -4.84693  31.0547    0.744772   0.904536    0.973074    0.948552    0.984337
#    8 │ CoveredCall         146.98  12.0093   15.8103   18.2595   21.3657        5.90912  -33.4985    7.42136  -0.660811   0.0817074   0.124231    0.107567    0.145365
#    9 │ ProtectedPut        154.04   8.02894   8.67429   9.09554   9.30692       7.24164   -9.6229   41.4026    0.912551   0.0521224   0.0590466   0.0563119   0.0604188

# CSV.write("problem3_risk.csv",risk)