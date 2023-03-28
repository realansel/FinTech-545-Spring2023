
function VaR(a; alpha=0.05)
    x = sort(a)
    nup = convert(Int64,ceil(size(a,1)*alpha))
    ndn = convert(Int64,floor(size(a,1)*alpha))
    v = 0.5*(x[nup]+x[ndn])

    return -v
end

function VaR(d::UnivariateDistribution; alpha=0.05)
    -quantile(d,alpha)
end

function ES(a; alpha=0.05)
    #ReWrite VaR code so that we are not forcing a double sort on a
    x = sort(a)
    nup = convert(Int64,ceil(size(a,1)*alpha))
    ndn = convert(Int64,floor(size(a,1)*alpha))
    v = 0.5*(x[nup]+x[ndn])
    
    es = mean(x[x.<=v])
    return -es
end

function ES(d::UnivariateDistribution; alpha=0.05)
    v = VaR(d;alpha=alpha)
    f(x) = x*pdf(d,x)
    st = quantile(d,1e-12)
    return -quadgk(f,st,-v)[1]/alpha
end


#Calculation of Risk Metrics
function aggRisk(df,aggLevel::Vector{Symbol})
    gdf = []
    if !isempty(aggLevel)
        gdf = groupby(df,vcat(aggLevel,[:iteration]))

        agg = combine(gdf,
            :currentValue => sum => :currentValue,
            :simulatedValue => sum => :simulatedValue,
            :pnl => sum => :pnl
        )
        
        gdf = groupby(agg,aggLevel)
    else
        gdf = groupby(df,[:iteration])

        gdf = combine(gdf,
            :currentValue => sum => :currentValue,
            :simulatedValue => sum => :simulatedValue,
            :pnl => sum => :pnl
        )
    end

    risk = combine(gdf, 
        :currentValue => (x-> first(x,1)) => :currentValue,
        :pnl => (x -> VaR(x,alpha=0.05)) => :VaR95,
        :pnl => (x -> ES(x,alpha=0.05)) => :ES95,
        :pnl => (x -> VaR(x,alpha=0.01)) => :VaR99,
        :pnl => (x -> ES(x,alpha=0.01)) => :ES99,
        :pnl => std => :Standard_Dev,
        :pnl => (x -> [extrema(x)]) => [:min, :max],
        :pnl => mean => :mean
    )
    risk[!,:VaR95_Pct] =  risk.VaR95 ./ risk.currentValue
    risk[!,:VaR99_Pct] =  risk.VaR99 ./ risk.currentValue
    risk[!,:ES95_Pct] =  risk.ES95 ./ risk.currentValue
    risk[!,:ES99_Pct] =  risk.ES99 ./ risk.currentValue
    return risk
end