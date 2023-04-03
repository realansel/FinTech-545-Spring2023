
function bt_american(call::Bool, underlying,strike,ttm,rf,b,ivol,N)

    dt = ttm/N
    u = exp(ivol*sqrt(dt))
    d = 1/u
    pu = (exp(b*dt)-d)/(u-d)
    pd = 1.0-pu
    df = exp(-rf*dt)
    z = call ? 1 : -1

    nNodeFunc(n) = convert(Int64,(n+1)*(n+2)/2 )
    idxFunc(i,j) = nNodeFunc(j-1)+i+1
    nNodes = nNodeFunc(N)

    optionValues = Vector{Float64}(undef,nNodes)

    for j in N:-1:0
        for i in j:-1:0
            idx = idxFunc(i,j)
            price = underlying*u^i*d^(j-i)
            optionValues[idx] = max(0,z*(price-strike))
            
            if j < N
                optionValues[idx] = max(optionValues[idx], df*(pu*optionValues[idxFunc(i+1,j+1)] + pd*optionValues[idxFunc(i,j+1)])  )
            end
        end
    end

    return optionValues[1]
end


# divAmts and divTimes are vectors
# divTimes is the time of the dividends in relation to the grid of j ∈ 0:N 
function bt_american(call::Bool, underlying,strike,ttm,rf,divAmts::Vector{Float64},divTimes::Vector{Int64},ivol,N)

    # println("Call:  divAmts:$divAmts ")
    # println("      divTimes:$divTimes ")
    # println("             N:$N ")
    # println("           ttm:$ttm ")
    # println("    underlying:$underlying ")

    #if there are no dividends or the first dividend is outside out grid, return the standard bt_american value
    if isempty(divAmts) || isempty(divTimes)
        return bt_american(call, underlying,strike,ttm,rf,rf,ivol,N)
    elseif divTimes[1] > N
        return bt_american(call, underlying,strike,ttm,rf,rf,ivol,N)
    end

    dt = ttm/N
    u = exp(ivol*sqrt(dt))
    d = 1/u
    pu = (exp(rf*dt)-d)/(u-d)
    pd = 1.0-pu
    df = exp(-rf*dt)
    z = call ? 1 : -1

    nNodeFunc(n) = convert(Int64,(n+1)*(n+2)/2 )
    idxFunc(i,j) = nNodeFunc(j-1)+i+1
    nDiv = size(divTimes,1)
    nNodes = nNodeFunc(divTimes[1])

    optionValues = Vector{Float64}(undef,nNodes)

    for j in divTimes[1]:-1:0
        for i in j:-1:0
            idx = idxFunc(i,j)
            price = underlying*u^i*d^(j-i)        
            
            if j < divTimes[1]
                #times before the dividend working backward induction
                optionValues[idx] = max(0,z*(price-strike))
                optionValues[idx] = max(optionValues[idx], df*(pu*optionValues[idxFunc(i+1,j+1)] + pd*optionValues[idxFunc(i,j+1)])  )
            else
                #time of the dividend
               valNoExercise = bt_american(call, price-divAmts[1], strike, ttm-divTimes[1]*dt, rf, divAmts[2:nDiv], divTimes[2:nDiv] .- divTimes[1], ivol, N-divTimes[1])
               valExercise =  max(0,z*(price-strike))
               optionValues[idx] = max(valNoExercise,valExercise)
            end
        end
    end

    return optionValues[1]
end