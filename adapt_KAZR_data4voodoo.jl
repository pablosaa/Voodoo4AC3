# function to adapt KAZR spectrum data for Voodoo's input
function adapt_KAZR_data4voodoo(spec::Dict; NORMALIZE::Bool=true,
                                var::Symbol=:Znn,
                                cln_time::Vector{DateTime} = Vector{DateTime}(undef,0),
                                TIME_STEP::Int = 30,
                                Δs::Int = 1,
                                MaxHkm::Float64 = 8.0)

    ## 1) ++++
    # Correcting spectral reflectivity for noise level:
    spec[:Znn], spec[:SNR] = if haskey(spec, :Nspec_ave)
        ARMtools.Extract_Spectra_NL(spec[:η_hh]; p=Int64(spec[:Nspec_ave]))
    else
        ARMtools.Extract_Spectra_NL(spec[:η_hh])
    end

    ## 2) ++++
    # if cloudnet time not provided then make time steps every 30 seconds:
    cln_time = if isempty(cln_time)
        let Δt = Dates.Second(TIME_STEP)
            round(spec[:time][1], Δt):Δt:round(spec[:time][end], Δt)
        end
    else
        cln_time;
    end
        
    # spectrum time indexes to match cln_time:
    thr_ts = minimum(diff(spec[:time]))
    idx_ts = [findfirst(abs.(x .- spec[:time]) .< 2thr_ts) for x ∈ cln_time ]  |> x->filter(!isnothing,x)
    # checking whether "nothing" is found in idx_ts (when nearest time is less than 2*thr)
    idx_ts .|> isnothing |> any && error("Time steps larger than twice the threshold!!")

    ## 3) ++++
    # Definition of dimension for output variables:
    # * Height dimension:
    n_rg = findlast(≤(MaxHkm), 1f-3spec[:height]); #250

    # * Time dimesnion:
    n_ts = length(idx_ts);

    # * Number of samples:
    #n_samples = filter(≥(0), spec[:spect_mask][1:n_rg, idx_ts]) |> length
    n_samples = length(spec[:spect_mask][1:n_rg, idx_ts])

    # * Number of spectral dimension:
    n_vel = length(spec[:vel_nn])

    ## 4) ++++
    # Creating the feature output array:
    # according to voodoo predictor: (n_samples, n_ch=1, n_ts=6, n_vel)
    features = fill(NaN, (n_samples, 1, 6, n_vel))

    # helper variable to fill data into features: (n_vel, n_samples, n_ch, n_ts)
    fuzzydat = PermutedDimsArray(features, (4, 1, 2, 3))

    ## 5) ++++
    # Filling feature array with data:
    n_ts = length(spec[:time])

    foreach(enumerate(range(-2Δs, step=Δs, length=6))) do (k, j)
        idx_in = @. min(n_ts, max(1, (idx_ts + j)))
        
        dat_in = spec[:spect_mask][1:n_rg, idx_in] .≥ 0
        tmp_spec = spec[:spect_mask][1:n_rg, idx_in][dat_in] .+ 1
        tmp_dat = dat_in[:]
        fuzzydat[:, tmp_dat, 1, k] = spec[var][:, tmp_spec]
        
    end
    
    # 6) ++++ output variables:
    # defining output array: masked
    masked = @. ifelse(spec[:spect_mask][1:n_rg, idx_ts] ≥ 0, true, false)

    # 7) ++++ Optional Normalization:
    # converting features array into normalized array [0, 1]
    NORMALIZE && (features = voodoo.η₀₁(features))
    
    return features, masked, idx_ts
end
# ---/PermutedDimsArray(features, (2, 3, 4, 1))


# ----/

