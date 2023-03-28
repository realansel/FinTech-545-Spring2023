
#Function to calculate expoentially weighted covariance.  
function ewCovar(x,λ)
    m,n = size(x)
    w = Vector{Float64}(undef,m)

    #Remove the mean from the series
    xm = mean.(eachcol(x))
    for j in 1:n
        x[:,j] = x[:,j] .- xm[j]
    end

    #Calculate weight.  Realize we are going from oldest to newest
    for i in 1:m
        w[i] = (1-λ)*λ^(m-i)
    end
    #normalize weights to 1
    w = w ./ sum(w)

    #covariance[i,j] = (w # x)' * x  where # is elementwise multiplication.
    return (w .* x)' * x
end

function expW(m,λ)
    w = Vector{Float64}(undef,m)
    for i in 1:m
        w[i] = (1-λ)*λ^(m-i)
    end
    #normalize weights to 1
    w = w ./ sum(w)
    return w
end