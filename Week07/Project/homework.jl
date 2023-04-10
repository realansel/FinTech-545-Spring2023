using BenchmarkTools
using Distributions
using Random
using StatsBase
using Roots
using QuadGK
using DataFrames
using Plots
using LinearAlgebra
using JuMP
using Ipopt
using Dates
using ForwardDiff
using FiniteDiff
using CSV
using LoopVectorization

include("../../library/return_calculate.jl")
include("../../library/gbsm.jl")
include("../../library/bt_american.jl")
include("../../library/RiskStats.jl")
include("../../library/simulate.jl")
include("../../library/fitted_model.jl")


s = 151.03
x = 165
ttm = (Date(2022,4,15)-Date(2022,3,13)).value/365
rf = 0.0425
b = 0.0053

#Calculate the GBSM Values.  Return Struct has all values
bsm_call = gbsm(true,s,x,ttm,rf,rf-b,.2,includeGreeks=true)
bsm_put = gbsm(false,s,x,ttm,rf,rf-b,.2,includeGreeks=true)

outTable = DataFrame(
    :Valuation => ["GBSM","GBSM"],
    :Type => ["Call", "Put"],
    :Method => ["Closed Form","Closed Form"],
    :Delta => [bsm_call.delta, bsm_put.delta],
    :Gamma => [bsm_call.gamma, bsm_put.gamma],
    :Vega => [bsm_call.vega, bsm_put.vega],
    :Theta => [bsm_call.theta, bsm_put.theta],
    :Rho => [missing, missing],
    :CarryRho => [bsm_call.cRho, bsm_put.cRho]
)

#Differential Library call the calculate the gradient
_x = [s,x,ttm,rf,rf-b,.2]
f(_x) = gbsm(true,_x...).value
call_grad = ForwardDiff.gradient(f,_x)

f(_x) = gbsm(false,_x...).value
put_grad = ForwardDiff.gradient(f,_x)

#Derivative of Delta = Gamma
f(_x) = gbsm(true,_x...;includeGreeks=true).delta
call_gamma = ForwardDiff.gradient(f,_x)[1]
f(_x) = gbsm(false,_x...;includeGreeks=true).delta
put_gamma = ForwardDiff.gradient(f,_x)[1]

outTable = vcat(outTable,
    DataFrame(
        :Valuation => ["GBSM","GBSM"],
        :Type => ["Call", "Put"],
        :Method => ["Numeric","Numeric"],
        :Delta => [call_grad[1], put_grad[1]],
        :Gamma => [call_gamma, put_gamma],
        :Vega => [call_grad[6], put_grad[6]],
        :Theta => [-call_grad[3], -put_grad[3]],
        :Rho => [call_grad[4], put_grad[4]],
        :CarryRho => [call_grad[5], put_grad[5]]
    )
)

# bt_american(call::Bool, underlying,strike,ttm,rf,divAmts::Vector{Float64},divTimes::Vector{Int64},ivol,N)
divDate = Date(2022,04,11)
divDays = (divDate - Date(2022,3,13)).value
ttmDays = ttm*365
NPoints = convert(Int64,ttmDays*3)
divPoint = divDays*3
divAmt = 0.88

#Values
am_call = bt_american(true, s,x,ttm,rf,[divAmt],[divPoint],.2,NPoints)
am_put = bt_american(false, s,x,ttm,rf,[divAmt],[divPoint],.2,NPoints)

_x = [s,x,ttm,rf,.2]
function f(_x)
    _in = collect(_x)
    bt_american(true, _in[1],_in[2],_in[3],_in[4],[divAmt],[divPoint],_in[5],NPoints)
end
call_grad = FiniteDiff.finite_difference_gradient(f,_x)
δ = 1 #Need to play with the offset value to get a good derivative.  EXTRA 0.5 point if they do this
call_gamma = (bt_american(true, s+δ,x,ttm,rf,[divAmt],[divPoint],.2,NPoints)+bt_american(true, s-δ,x,ttm,rf,[divAmt],[divPoint],.2,NPoints)-2*am_call)/(δ^2)
δ = 1e-6
call_div = (bt_american(true, s,x,ttm,rf,[divAmt+δ],[divPoint],.2,NPoints)-am_call)/(δ)


function f(_x)
    _in = collect(_x)
    bt_american(false, _in[1],_in[2],_in[3],_in[4],[divAmt],[divPoint],_in[5],NPoints)
end
put_grad = FiniteDiff.finite_difference_gradient(f,_x)
δ = 10
put_gamma = (bt_american(false, s+δ,x,ttm,rf,[divAmt],[divPoint],.2,NPoints)+bt_american(false, s-δ,x,ttm,rf,[divAmt],[divPoint],.2,NPoints)-2*am_call)/(δ^2)
δ = 1e-6
put_div = (bt_american(false, s,x,ttm,rf,[divAmt+δ],[divPoint],.2,NPoints)-am_put)/(δ)

outTable = vcat(outTable,
    DataFrame(
        :Valuation => ["BT","BT"],
        :Type => ["Call", "Put"],
        :Method => ["Numeric","Numeric"],
        :Delta => [call_grad[1], put_grad[1]],
        :Gamma => [call_gamma, put_gamma],
        :Vega => [call_grad[5], put_grad[5]],
        :Theta => [-call_grad[3], -put_grad[3]],
        :Rho => [call_grad[4], put_grad[4]],
        :CarryRho => [missing, missing]
    )
)

sort!(outTable,[:Type, :Valuation, :Method])
println(outTable)
println("Call Derivative wrt Dividend: $call_div")
println("Put  Derivative wrt Dividend: $put_div")

# 6×9 DataFrame
#  Row │ Valuation  Type    Method       Delta       Gamma      Vega     Theta      Rho              CarryRho      
#      │ String     String  String       Float64     Float64    Float64  Float64    Float64?         Float64?      
# ─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────
#    1 │ BT         Call    Numeric       0.0749314  0.0186027  6.32562  -7.43533         0.933496   missing       
#    2 │ GBSM       Call    Closed Form   0.0829713  0.0168229  6.93871  -8.12652   missing                1.13295
#    3 │ GBSM       Call    Numeric       0.0829713  0.0168229  6.93871  -8.12652        -0.0303599        1.13295
#    4 │ BT         Put     Numeric      -0.936203   0.301074   5.6463   -0.391418      -12.4527     missing       
#    5 │ GBSM       Put     Closed Form  -0.91655    0.0168229  6.93871  -1.94099   missing              -12.5153
#    6 │ GBSM       Put     Numeric      -0.91655    0.0168229  6.93871  -1.94099        -1.24273        -12.5153
# Call Derivative wrt Dividend: -0.02162170542607811
# Put  Derivative wrt Dividend: 0.9393956741376996



#Problem #2
portfolio = CSV.read("../Week07/Project/problem2.csv",DataFrame)
currentDate = Date(2023,3,3)
divDate = Date(2023,3,15)
divAmt =1.00
currentS=151.03
mult = 5
daysDiv = (divDate - currentDate).value


prices = CSV.read("../Week07/Project/DailyPrices.csv",DataFrame)[!,[:Date, :AAPL]]
returns = return_calculate(prices,dateColumn="Date")[!,:AAPL]
returns = returns .- mean(returns)
sd = std(returns)

portfolio[!,:ExpirationDate] = [
    portfolio.Type[i] == "Option" ? Date(portfolio.ExpirationDate[i],dateformat"mm/dd/yyyy") : missing
    for i in 1:size(portfolio,1) ]

#Implied Vols
# bt_american(call::Bool, underlying,strike,ttm,rf,divAmts::Vector{Float64},divTimes::Vector{Int64},ivol,N)
portfolio[!, :ImpVol] = [
    portfolio.Type[i] == "Option" ?
    find_zero(x->bt_american(portfolio.OptionType[i]=="Call",
                        currentS,
                        portfolio.Strike[i],
                        (portfolio.ExpirationDate[i]-currentDate).value/365,
                        rf,
                        [divAmt],[daysDiv*mult],x,convert(Int64,(portfolio.ExpirationDate[i]-currentDate).value*mult))
                -portfolio.CurrentPrice[i],.2)    : missing     
    for i in 1:size(portfolio,1)
]

#Delta function for BT American
function bt_delta(call::Bool, underlying,strike,ttm,rf,divAmts::Vector{Float64},divTimes::Vector{Int64},ivol,N)

    f(_x) = bt_american(call::Bool, _x,strike,ttm,rf,divAmts::Vector{Float64},divTimes::Vector{Int64},ivol,N)
    FiniteDiff.finite_difference_derivative(f, underlying)
end

#Position Level Deltas needed for DN VaR
portfolio[!, :Delta] = [
    portfolio.Type[i] == "Option" ?  (
            bt_delta(portfolio.OptionType[i]=="Call",
                currentS, 
                portfolio.Strike[i], 
                (portfolio.ExpirationDate[i]-currentDate).value/365, 
                rf, 
                [divAmt],[daysDiv*mult],
                portfolio.ImpVol[i],convert(Int64,(portfolio.ExpirationDate[i]-currentDate).value*mult))*portfolio.Holding[i]*currentS    
    ) : portfolio.Holding[i] * currentS    
    for i in 1:size(portfolio,1)
]

#Simulate Returns
nSim = 10000
fwdT = 10
_simReturns = rand(Normal(0,sd),nSim*fwdT)

#collect 10 day returns
simPrices = Vector{Float64}(undef,nSim)
for i in 1:nSim
    r = 1.0
    for j in 1:fwdT
        r *= (1+_simReturns[fwdT*(i-1)+j])
    end
    simPrices[i] = currentS*r
end

iteration = [i for i in 1:nSim]
values = crossjoin(portfolio, DataFrame(:iteration=>iteration))
nVals = size(values,1)

#Precalculate the fwd TTM
values[!,:fwd_ttm] = [
    values.Type[i] == "Option" ? (values.ExpirationDate[i]-currentDate-Day(fwdT)).value/365 : missing
    for i in 1:nVals
]

#Valuation
simulatedValue = Vector{Float64}(undef,nVals)
currentValue = Vector{Float64}(undef,nVals)
pnl = Vector{Float64}(undef,nVals)
Threads.@threads for i in 1:nVals
    simprice = simPrices[values.iteration[i]]
    currentValue[i] = values.Holding[i]*values.CurrentPrice[i]
    if values.Type[i] == "Option"
        simulatedValue[i] = values.Holding[i]*bt_american(values.OptionType[i]=="Call",
                                                simprice,
                                                values.Strike[i],
                                                values.fwd_ttm[i],
                                                rf,
                                                [divAmt],[(daysDiv-fwdT)*mult],
                                                values.ImpVol[i],
                                                convert(Int64,values.fwd_ttm[i]*mult*365)
                                            )
    elseif values.Type[i] == "Stock"
        simulatedValue[i] = values.Holding[i]*simprice
    end
    pnl[i] = simulatedValue[i] - currentValue[i]
end

values[!,:simulatedValue] = simulatedValue
values[!,:pnl] = pnl
values[!,:currentValue] = currentValue



#Calculate Simulated Risk Values
risk = aggRisk(values,[:Portfolio])

#Calculate the Portfolio Deltas
gdf = groupby(portfolio, [:Portfolio])
portfolioDelta = combine(gdf,
    :Delta => sum => :PortfolioDelta    
)

#Delta Normal VaR is just the Portfolio Delta * quantile * current Underlying Price
portfolioDelta[!,:DN_VaR] = abs.(quantile(Normal(0,sd),.05)*sqrt(10)*portfolioDelta.PortfolioDelta)
portfolioDelta[!,:DN_ES] = abs.((sqrt(10)*sd*pdf(Normal(0,1),quantile(Normal(0,1),.05))/.05)*portfolioDelta.PortfolioDelta)

leftjoin!(risk,portfolioDelta[!,[:Portfolio, :DN_VaR, :DN_ES]],on=:Portfolio)

println(risk)

# 9×16 DataFrame
#  Row │ Portfolio     currentValue  VaR95     ES95      VaR99     ES99      Standard_Dev  min        max       mean        VaR95_Pct  VaR99_Pct   ES95_Pct    ES99_Pct    DN_VaR    DN_ES
#      │ String15      Float64       Float64   Float64   Float64   Float64   Float64       Float64    Float64   Float64     Float64    Float64     Float64     Float64     Float64?  Float64?
# ─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#    1 │ Straddle             11.65   1.30259   1.31911   1.32968   1.33147       4.05448   -1.33269  33.071     1.85414    0.11181     0.114136    0.113228    0.114289    1.38613   1.73826
#    2 │ SynLong               1.95  18.0294   22.2631   25.0626   28.0019       10.839    -37.8686   42.7683   -0.524121   9.24583    12.8526     11.417      14.36       17.7467   22.2551
#    3 │ CallSpread            4.59   3.90547   4.19721   4.38468   4.46553       2.51623   -4.58222   5.40767  -0.179533   0.850865    0.955267    0.914425    0.972881    4.93711   6.19133
#    4 │ PutSpread             3.01   2.75741   2.87879   2.95581   2.98091       2.51964   -3.0089    6.98765   0.481184   0.916083    0.981996    0.95641     0.990337    4.43382   5.56019
#    5 │ Stock               151.03  16.8072   20.7755   23.3908   26.2453       10.7264   -35.9916   43.6547    0.0376162  0.111284    0.154875    0.137559    0.173776   17.6271   22.1051
#    6 │ Call                  6.8    6.05804   6.38167   6.58671   6.67171       6.17394   -6.79218  37.9197    0.665008   0.890888    0.968634    0.938481    0.981134    9.56643  11.9967
#    7 │ Put                   4.85   4.52432   4.68241   4.7821    4.81358       5.37064   -4.84863  31.0764    1.18913    0.932849    0.986001    0.965445    0.992491    8.1803   10.2584
#    8 │ CoveredCall         146.98  12.9806   16.8365   19.3853   22.2191        6.37775  -31.9422    7.98391  -0.802279   0.0883157   0.131891    0.11455     0.151171   10.7278   13.4531
#    9 │ ProtectedPut        154.04   7.49743   7.83499   8.03186   8.06011       7.35185   -8.074    40.6453    1.00952    0.048672    0.0521414   0.0508634   0.0523248  12.0608   15.1247


###Problem 3 ###
#Read All Data
ff3 = CSV.read("../Week07/Project/F-F_Research_Data_Factors_daily.CSV", DataFrame)
mom = CSV.read("../Week07/Project/F-F_Momentum_Factor_daily.CSV",DataFrame)
prices = CSV.read("../Week07/Project/DailyPrices.csv",DataFrame)
returns = return_calculate(prices,dateColumn="Date")
rf = 0.0425

# Join the FF3 data with the Momentum Data
ffData = innerjoin(ff3,mom,on=:Date)
rename!(ffData, names(ffData)[size(ffData,2)] => :Mom)
ffData[!,names(ffData)[2:size(ffData,2)]] = Matrix(ffData[!,names(ffData)[2:size(ffData,2)]]) ./ 100
ffData[!,:Date] = Date.(string.(ffData.Date),dateformat"yyyymmdd")

returns[!,:Date] = Date.(returns.Date,dateformat"mm/dd/yyyy")

# Our 20 stocks
stocks = [:AAPL, :META, :UNH, :MA, :MSFT, :NVDA, :HD, :PFE, :AMZN, Symbol("BRK-B"), :PG, :XOM, :TSLA, :JPM, :V, :DIS, :GOOGL, :JNJ, :BAC, :CSCO]

# Data set of all stock returns and FF3+1 returns
to_reg = innerjoin(returns[!,vcat(:Date,stocks)], ffData, on=:Date)

println("Max RF value is: $(max(to_reg.RF...))")
#since the value is always 0, no need to difference the stock returns.

xnames = [Symbol("Mkt-RF"), :SMB, :HML, :Mom]

#OLS Regression for all Stocks
X = hcat(fill(1.0,size(to_reg,1)),Matrix(to_reg[!,xnames]))
Y = Matrix(to_reg[!,stocks])

Betas = (inv(X'*X)*X'*Y)'

#Calculate the means of the last 10 years of factor returns
#adding the 0.0 at the front to 0 out the fitted alpha in the next step
means = vcat(0.0,mean.(eachcol(ffData[ffData.Date .>= Date(2013,02,09),xnames])))

#Discrete Returns, convert to Log Returns and scale to 1 year
stockMeans =log.(1 .+ Betas*means)*255 .+ rf
covar = cov(log.(1.0 .+ Y))*255

#optimize.  Directly find the max SR portfolio.  Can also do this like in the notes and
#   build the Efficient Frontier

function sr(w...)
    _w = collect(w)
    m = _w'*stockMeans - rf
    s = sqrt(_w'*covar*_w)
    return (m/s)
end

n = length(stocks)

m = Model(Ipopt.Optimizer)
# set_silent(m)
# Weights with boundry at 0
@variable(m, w[i=1:n] >= 0,start=1/n)
register(m,:sr,n,sr; autodiff = true)
@NLobjective(m,Max, sr(w...))
@constraint(m, sum(w)==1.0)
optimize!(m)

w = round.(value.(w),digits=4)

OptWeights = DataFrame(:Stock=>String.(stocks), :Weight => w, :Er => stockMeans)
println(OptWeights)
println("Expected Retrun = $(stockMeans'*w)")
println("Expected Vol = $(sqrt(w'*covar*w))")
println("Expected SR = $(sr(w...)) ")

# 20×3 DataFrame
#  Row │ Stock   Weight   Er       
#      │ String  Float64  Float64  
# ─────┼───────────────────────────
#    1 │ AAPL     0.0525  0.204519
#    2 │ META    -0.0     0.180179
#    3 │ UNH      0.3092  0.158593
#    4 │ MA       0.0908  0.178517
#    5 │ MSFT     0.0458  0.196068
#    6 │ NVDA     0.0061  0.275016
#    7 │ HD      -0.0     0.145135
#    8 │ PFE      0.0335  0.13479
#    9 │ AMZN    -0.0     0.196912
#   10 │ BRK-B   -0.0     0.147398
#   11 │ PG       0.1496  0.128636
#   12 │ XOM      0.184   0.164509
#   13 │ TSLA    -0.0     0.213734
#   14 │ JPM     -0.0     0.154053
#   15 │ V       -0.0     0.161979
#   16 │ DIS     -0.0     0.152448
#   17 │ GOOGL   -0.0     0.179894
#   18 │ JNJ     -0.0     0.100215
#   19 │ BAC      0.0007  0.166333
#   20 │ CSCO     0.1279  0.169715
# Expected Retrun = 0.16249327524718193
# Expected Vol = 0.20370334317040237
# Expected SR = 0.5890589392379529