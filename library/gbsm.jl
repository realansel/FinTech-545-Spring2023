###
#Generalize Black Scholes Merton
# rf = b       -- Black Scholes 1973
# b = rf - q   -- Merton 1973 stock model where q is the continous dividend yield
# b = 0        -- Black 1976 futures option model
# b,r = 0      -- Asay 1982 margined futures option model
# b = rf - rff -- Garman and Kohlhagen 1983 currency option model where rff is the risk free
#                 rate of the foreign currency
###

struct GBSM 
    value
    delta
    gamma
    vega
    theta
    rho
    cRho
end

function gbsm(call::Bool, underlying, strike, ttm, rf, b, ivol; includeGreeks=false)
    d1 = (log(underlying/strike) + (b+ivol^2/2)*ttm)/(ivol*sqrt(ttm))
    d2 = d1 - ivol*sqrt(ttm)

    delta = 0
    gamma = 0
    vega  = 0
    theta = 0
    rho = 0
    cRho = 0

    if call
        delta = exp((b-rf)*ttm) * cdf(Normal(),d1)
        value = underlying * delta - strike*exp(-rf*ttm)*cdf(Normal(),d2)
    else
        delta = exp((b-rf)*ttm)*(cdf(Normal(),d1)-1)
        value = strike*exp(-rf*ttm)*cdf(Normal(),-d2) - underlying*exp((b-rf)*ttm)*cdf(Normal(),-d1)
    end

    if includeGreeks
        gamma = pdf(Normal(),d1)*exp((b-rf)*ttm)/(underlying*ivol*sqrt(ttm))
        vega = underlying*exp((b-rf)*ttm)*pdf(Normal(),d1)*sqrt(ttm)
        if call
            theta = - underlying*exp((b-rf)*ttm)*pdf(Normal(),d1)*ivol / (2*sqrt(ttm)) - 
                      (b-rf)*underlying*exp((b-rf)*ttm)*cdf(Normal(),d1) -
                      rf*strike*exp(-rf*ttm)*cdf(Normal(),d2)

            rho = ttm*strike*exp(-rf*ttm)*cdf(Normal(),d2)
            cRho = ttm*underlying*exp((b-rf)*ttm)*cdf(Normal(),d1)
        else
            theta = - underlying*exp((b-rf)*ttm)*pdf(Normal(),d1)*ivol / (2*sqrt(ttm)) + 
                      (b-rf)*underlying*exp((b-rf)*ttm)*cdf(Normal(),-d1) +
                      rf*strike*exp(-rf*ttm)*cdf(Normal(),-d2)

            rho = -ttm*strike*exp(-rf*ttm)*cdf(Normal(),-d2)
            cRho = - ttm*underlying*exp((b-rf)*ttm)*cdf(Normal(),-d1)
        end
    end

    return GBSM(value,delta,gamma,vega,theta,rho,cRho)
end