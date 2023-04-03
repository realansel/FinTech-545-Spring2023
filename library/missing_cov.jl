
#calculate either the covariance or correlation function when there are missing values
function missing_cov(x; skipMiss=true, fun=cov)
    n,m = size(x)
    nMiss = count.(ismissing, eachcol(x))
    #nothing missing, just calculate it.
    if 0==sum(nMiss)
        return fun(x)
    end

    idxMissing = Set.(findall.(ismissing,eachcol(x)))
    
    if skipMiss
        #Skipping Missing, get all the rows which have values and calculate the covariance
        rows = Set([i for i in 1:n])
        for c in 1:m
            for rm in idxMissing[c]
                delete!(rows,rm)
            end
        end
        rows = sort(collect(rows))
        return fun(x[rows,:])
    else
        #Pairwise, for each cell, calculate the covariance.
        out = Array{Float64,2}(undef,m,m)
        for i in 1:m
            for j in 1:i
                rows = Set([i for i in 1:n]) 
                for c in (i,j)
                    for rm in idxMissing[c]
                        delete!(rows,rm)
                    end
                end
                rows = sort(collect(rows))
                out[i,j] = fun(x[rows,[i,j]])[1,2]
                if i!=j
                    out[j,i] = out[i,j]
                end
            end
        end
        return out
    end
end