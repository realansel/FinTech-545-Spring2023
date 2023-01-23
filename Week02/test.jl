using Distributions
using SpecialFunctions

function __T_loglikelihood(mu,s,nu,x)
    n = size(x,1)
    np12 = (nu + 1.0)/2.0

    mess = loggamma(np12) - loggamma(nu/2.0) - log(sqrt(π*nu)*s)
    xm = ((x .- mu)./s).^2 * (1.0/nu) .+ 1
    innerSum = sum(log.(xm))
    ll = n*mess - np12*innerSum
    return ll
end

x = [i for i in -10:.001:10]

df = 10
mu=.5
s=2.0
td = TDist(df)

dist_ll = loglikelihood.(td,(x .- mu)/s)
my_ll = __T_loglikelihood.(0,1,df,x)
diff = dist_ll - my_ll

sse = diff'*diff
println(sse)
