# function to adapt KAZR spectrum data for Voodoo's input
function adapt_KAZR_data4voodoo(spec::Dict; NORMALIZE=true, var::Symbol=:Znn, cln_time::Vector{DateTime}=[], TIME_STEP=30)

    # 1) ++++
    # Correcting spectral reflectivity for noise level:
    spec[:Znn], spec[:SNR] = if haskey(spec, :Nspec_ave)
        ARMtools.Extract_Spectra_NL(spec[:η_hh]; p=Int64(spec[:Nspec_ave]))
    else
        ARMtools.Extract_Spectra_NL(spec[:η_hh])
    end

    # 2) ++++
    # making time steps everey 30 seconds:
    cln_time = if isempty(cln_time)
        let Δt = Second(TIME_STEP)
            round(spec[:time][1], Δt):Δt:round(spec[:time][end], Δt)
        end
    end
        
    # spectrum time indexes to match cln_time:
    thr_ts = minimum(diff(spec[:time]))
    idx_ts = [findfirst(abs.(x .- spec[:time]) .< 2thr_ts) for x ∈ cln_time ]  |> x->filter(!isnothing,x)
    # checking whether "nothing" is found in idx_ts (when nearest time is less than 2*thr)
    idx_ts .|> isnothing |> any && error("Time steps larger than twice the threshold!!")

    # 3) ++++
    # selecting the 6 spectra around time step:
    ii_6spec = map(idx_ts) do j
        let n_ts = length(spec[:time])
            x_ini = ifelse( 3 ≤ j ≤ n_ts-5, j-2, min(j, n_ts-5))
            range(x_ini, length=6)
        end
    end
    ii_6spec = hcat(ii_6spec...);  # (6 x n_ts)

    # dimensions for output file:
    ##n_samples = map(1:6) do i_6p
    ##    local n = spec[:spect_mask][:, ii_6spec[i_6p, :]]
    ##    length(n[n .≥ 0])
    ##end |> maximum
    n_rg = 250 #length(spec[:height])
    
    n_samples = filter(≥(0), spec[:spect_mask][1:n_rg, idx_ts]) |> length
        
    n_vel = length(spec[:vel_nn])
    
    n_ts = length(spec[:time])

    # 4) ++++ output variables:
    # defining output array: features & masked
    # masked = @. ifelse(spec[:spect_mask][1:n_rg, idx_ts] ≥ 0, true, false)    

    # according to voodoo predictor: (n_samples, n_ch=1, n_ts=6, n_vel)
    features = fill(NaN32, (n_samples, 1, 6, n_vel))

    # helper variable to fill data into features: (n_vel, n_samples, n_ch, n_ts)
    fuzzydat = PermutedDimsArray(features, (4, 1, 2, 3))

    fuzzydat = let k = ii_6spec[3, :]
        local garbage = view(spec[:spec_mask][1:n_rg, k]) .+ 1
        fill(spec[var][:, garbage], (n_vel, n_samples, 1, 6))
    end
    
    for i_6p ∈ (1, 2, 4, 5, 6)
        garbage = map(spec[:spect_mask][1:n_rg, ii_6spec[i_6p, :] ]) do x   #i_6p
            x .≥ 0 ? x+1 : missing
        end;

        let var_rg = skipmissing(garbage) |> collect
            local n_subsample = min(length(var_rg), n_samples)
            fuzzydat[:, 1:n_subsample, 1, i_6p] = spec[var][:, var_rg[1:n_subsample]]
        end
    end

    # converting features array into normalized array [0, 1]
    NORMALIZE && (features = η₀₁(features))
    
    return features, idx_ts
end
# ---/PermutedDimsArray(features, (2, 3, 4, 1))

# *******************************************************************
# Function to normalize the spectrum
"""
Function to normalize spectrum data.
   > η = η₀₁(Znn)
with default normalization min and max limits -100 and -55, respectively.
or
   > η = η₀₁(Znn, η0 = -90, η1 = -50)
for other min/max limit values e.g. -90 and -50.
All values outside the range min to max are set-up to 0 and 1. 

"""
function η₀₁(η::Array{Float32, 4}; η0 =-100, η1 = -50)
    H_out = @. (η - η0)/(η1 - η0)
    @. H_out[H_out > 1] = 1f0
    @. H_out[H_out < 0] = 0f0
    
    return H_out
end
# end of function

# Alternative function to normalize based on (min, max) of given dimension:
# Input Arrray must have (n_samples, 1, 6, n_vel)
function Norm_dim(X::Array{Float32, 4}; dims=1)
    X0, X1 = extrema(X, dims=4)
    return η₀₁(X, η0=X0, η1=X1)
end
# ----/

