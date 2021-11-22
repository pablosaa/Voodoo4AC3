# function to adapt KAZR spectrum data for Voodoo's input
function adapt_KAZR_data4voodoo(spec::Dict; NORMALIZE=true)

    # making time steps everey 30 seconds:
    TIME_STEP = 30
    cln_time = round(spec[:time][1], Second(TIME_STEP)):Second(TIME_STEP):round(spec[:time][end], Second(TIME_STEP))

    # spectra time indexes to match cln_time:
    thr_ts = minimum(diff(spec[:time]))
    idx_ts = [findfirst(abs.(x .- spec[:time]) .< 2thr_ts) for x ∈ cln_time ]  |> x->filter(!isnothing,x)
    # checking whether "nothing" is found in idx_ts (when nearest time is less than 2*thr)
    idx_ts .|> isnothing |> any && error("Time steps larger than twice the threshold!!")
    
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
    
    n_samples = let n = spec[:spect_mask][1:n_rg, idx_ts]
        length(n[n .≥ 0]) |> maximum
    end
        
    n_vel = length(spec[:vel_nn])
    
    n_ts = length(spec[:time])

    # defining output array: features & masked
    masked = @. ifelse(spec[:spect_mask][1:n_rg, idx_ts] ≥ 0, true, false)
    

    # according to voodoo predictor: (n_samples, n_ch=1, n_ts=6, n_vel)
    features = fill(NaN32, (n_samples, 1, 6, n_vel))

    # helper variable to fill data into features
    fuzzydat = PermutedDimsArray(features, (4, 1, 2, 3))

    # correcting spectral reflectivity for noise level:
    Znn, SNR = ARMtools.Extract_Spectra_NL(spec[:η_hh]; p=Int64(spec[:Nspec_ave]))

    for i_6p ∈ (1:6)
        garbage = map(spec[:spect_mask][1:n_rg, ii_6spec[i_6p, :] ]) do x #i_6p
            x .≥ 0 ? x+1 : missing
        end;

        let var = skipmissing(garbage) |> collect
            local n_subsample = min(length(var), n_samples)
            fuzzydat[:, 1:n_subsample, 1, i_6p] = Znn[:, var[1:n_subsample]]
        end
    end

    # converting features array into normalized array [0, 1]
    NORMALIZE && (features = η₀₁(features))
    
    return features, masked, idx_ts, Znn, SNR
end
# ---/PermutedDimsArray(features, (2, 3, 4, 1))

# *******************************************************************
# Function to normilize the spectrum
function η₀₁(η::Array{Float32, 4}; η0 =-100, η1 = -55)
    return @. (η - η0)/(η1 - η0)
end
# end of function
# ----/

