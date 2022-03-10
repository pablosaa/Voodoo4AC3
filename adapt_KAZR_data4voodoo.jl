# function to adapt KAZR spectrum data for Voodoo's input
function adapt_KAZR_data4voodoo(spec::Dict; NORMALIZE::Bool=true,
                                var::Symbol=:Znn,
                                cln_time::Vector{DateTime} = Vector{DateTime}(undef,0),
                                TIME_STEP::Int = 30,
                                Δs::Int = 1)

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
    n_rg = 250

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
    k = 0
    foreach(range(-2Δs, step=Δs, length=6)) do j
        k += 1 
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
    NORMALIZE && (features = η₀₁(features))
    
    return features, masked, idx_ts
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
function η₀₁(η::Array{<:AbstractFloat, 4}; ηlim::Tuple{Int, Int} = (-100, -50))
    η0 = ηlim[1]
    η1 = ηlim[2]
    H_out = @. (η - η0)/(η1 - η0)
    @. H_out = min(1, H_out) # H_out[H_out > 1] = 1f0
    @. H_out = max(0, H_out) #[H_out < 0] = 0f0
    # removing NaNs:
    @. H_out[isnan(H_out)] = 0
    return H_out
end
# end of function

# Alternative function to normalize based on (min, max) of given dimension:
# Input Arrray must have (n_samples, 1, 6, n_vel)
function Norm_dim(X::Array{<:AbstractFloat, 4}; dims=1)
    X0, X1 = extrema(X, dims=4)
    return η₀₁(X, η0=X0, η1=X1)
end
# ----/

