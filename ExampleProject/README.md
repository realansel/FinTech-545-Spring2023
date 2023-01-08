#Example Project

Run code from the local project
```
julia --project example.jl
```

If you get an error on missing libraries, install the packages
Start julia in the current library
```
julia --project
```

Then add the packages

```julia
using Pkg
Pkg.add("StatsBase")
Pkg.add("Printf")
```

