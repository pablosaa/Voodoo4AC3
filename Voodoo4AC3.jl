# File containing module for using Voodoo with ARM radars:
#

module voodoo

using PyCall
using TOML
using Dates
using ARMtools

# *******************************************************************
function MakePrediction(X_in::Array{<:AbstractFloat, 4};
                        sizeout::NTuple{2, Int} = (0,0))
    
    # loading python torch package:
    torch = pyimport("torch");

    # including path of Voodoo's Library into python path:
    pushfirst!(PyVector(pyimport("sys")."path"), joinpath(@__DIR__, "Voodoo") );

    # Loading Voodoo Library Torch model:
    TM = pyimport("libVoodoo.TorchModel");

    # ---
    # Defining Voodoo model input parameters:
    model_setup_file = "Voodoo/VnetSettings-1.toml"
    trained_model = "Voodoo/Vnet0x60de1687-fnX-gpu0-VN.pt"
    NCLASSES = 3
    USEDEVICE = "cpu"

    torch_settings = let
        helper_settings = TOML.parsefile(model_setup_file)["pytorch"]
        Dict(Symbol(k) => v for (k, v) in helper_settings);
    end
    torch_settings[:dev] = USEDEVICE;

    # setting numbers of threads used in CPU:
    # torch.set_num_threads(int)

    # ---
    # Converting Array into Torch array:
    Xₜ = torch.Tensor(X_in);

    model = TM.VoodooNet(Xₜ.shape, NCLASSES; torch_settings...) ;

    model.load_state_dict(torch.load(trained_model,
                                     map_location=model.device)["state_dict"])

    # Performing Prediction:
    prediction = model.predict(Xₜ, batch_size = 256);

    # Get model prediction converted to Julia array (via numpy):
    X_out = prediction.to(USEDEVICE).numpy();

    predict_var = try
        # Final output array with predictions (range x time):
        reshape(X_out, (sizeout..., NCLASSES));
    catch
        X_out
    end

    return predict_var
end

# ***********************************************************************
# Function to normalize the spectrum
"""
Function to normalize spectrum data.
   > η = η₀₁(Znn)
with default normalization min and max limits -100 and -40, respectively.
or
   > η = η₀₁(Znn, η0 = -90, η1 = -50)
for other min/max limit values e.g. -90 and -50.
All values outside the range min to max are set-up to 0 and 1. 

"""
function η₀₁(η::Array{<:AbstractFloat, 4}; ηlim::NTuple{2, Int} = (-100, -40))
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
# ----/

# Alternative function to normalize based on (min, max) of given dimension:
# Input Arrray must have (n_samples, 1, 6, n_vel)
function Norm_dim(X::Array{<:AbstractFloat, 4}; dims=1)
    X0, X1 = extrema(X, dims=4)
    return η₀₁(X, η0=X0, η1=X1)
end
# ----/

"""
Function to convert ARM radar spectrum data to be suitable as input
for Voodoo predictor. This function perform the following tasks:
1. correct for spectrum noise,
2. determine the time series for the output,
3. create the feature output array with dimension compliant to Vooodoo
4. (optional) normilize output array to [0,1]

USAGE:
julia> feature, masked, idx_time = adapt_RadarData(spec)
julia> feature, masked, idx_time = adapt_RadarData(spec, cln_time=cloudnet_time)
julia> feature, masked, idx_time = adapt_RadarData(spec, var=:SNR, Δs=3)

INPUT:
* spec::Dict() -> Variable with spectrum data as returned from ARMtools.jl
* NORMALIZE::Bool -> flag to normalize [0,1] the output. Default false
* var::Symbol -> spectrum variable to use, either :Znn (default) or :SNR
* cln_time::Vector -> DateTime vector with values to consider as centered
* Δs::Int -> step for filling the 6 spectra, default 1
* MaxHkm::Float -> maximum altitude to consider in km, default 8km
* TIME_STEP::Int -> if cln_time is not provided, a time vector is created 

OUTPUT:
* feature::Array{T}(num_samples, 1, 6, n_dopplervelocity) -> data array adapted
* masked::Array{Bool}(num_time, num_heights) -> true for considered spectrum
* idx_time::Vector{Int} -> indexes of the spec[:time] considered in output
"""
function adapt_RadarData(spec::Dict;
                         NORMALIZE::Bool=false,
                         var::Symbol=:Znn,
                         cln_time::Vector{DateTime} = Vector{DateTime}(undef,0),
                         TIME_STEP::Int = 30,
                         Δs::Int = 5,
                         MaxHkm::Float64 = 8.0,
                         adjustDoppler = false)
    
    ## 1) ++++
    # Correcting spectral reflectivity for noise level:
    spec[:Znn], spec[:SNR] = if haskey(spec, :Nspec_ave)
        ARMtools.Extract_Spectra_NL(spec[:η_hh]; p=Int64(spec[:Nspec_ave]))
    else
        ARMtools.Extract_Spectra_NL(spec[:η_hh])
    end
    # ------------------------

    ## 2) ++++
    # if cloudnet time not provided then make time steps every TIME_STEP seconds:
    cln_time = if isempty(cln_time)
        let Δt = Dates.Second(TIME_STEP)
            round(spec[:time][1], Δt):Δt:round(spec[:time][end], Δt)
        end
    else
        cln_time;
    end

    # input spectrum spec[:time] indexes to match cln_time time series:    
    idx_ts = let thr_ts = diff(spec[:time]) |> minimum
        local tmp = [findfirst(abs.(x .- spec[:time]) .< 2thr_ts) for x ∈ cln_time]
        filter(!isnothing, tmp)
    end
    # checking whether "nothing" is found in idx_ts (when nearest time is less than 2*thr)
    idx_ts .|> isnothing |> any && error("Time steps larger than twice the threshold!!")
    # -------------------------------
    ## 3.1) ++++
    # Definition of dimension for output variables:
    # * Height dimension:
    n_rg = findlast(≤(MaxHkm), 1f-3spec[:height]); #250

    # * Number of samples:
    n_samples = filter(≥(0), spec[:spect_mask][1:n_rg, idx_ts]) |> length
    #n_samples = length(spec[:spect_mask][1:n_rg, idx_ts])

    # * Number of Doppler spectral bins dimension (voodoo only support 256):
    n_vel = 256  # length(spec[:vel_nn])

    ## 3.1) ++++
    # Creating the features output array:
    # according to voodoo predictor, should be (n_samples, n_ch=1, n_ts=6, n_vel)
    features = fill(NaN32, (n_samples, 1, 6, n_vel))

    # helper variable to fill data into features: (n_vel, n_samples, n_ch, n_ts)
    fuzzydat = PermutedDimsArray(features, (4, 1, 2, 3))

    ## 3.2) ++++
    ## 3.2.1) ++++
    idx_rad, idx_voo, Vn_voo = index4DopplerDims(spec[:vel_nn], adjust = adjustDoppler)

    ## 3.2.2) ++++
    # Filling feature array with 6 consecutive spectra centered at cln_time:
    n_ts = length(spec[:time])

    foreach(idx_ts) do k
        len_in = findall(≥(0), spec[:spect_mask][1:n_rg, k])
        i0 = findfirst(isnan, fuzzydat[128,:,1,3])  #(i-1)*len_in + 1
        i1 = i0 + length(len_in) -1

        δts = k .+ range(-2Δs, step=Δs, length=6) |> x-> min.(n_ts, max.(1, x))

        foreach(enumerate(δts)) do (j, its)
            dat_in = spec[:spect_mask][len_in, its] .+ 1
            # dummy (256, len_in)
            dummy = hcat(map(x->x≠0 ? spec[var][idx_rad, x] : fill(NaN32, length(idx_rad)), dat_in)...)
            fuzzydat[idx_voo, i0:i1, 1, j] = dummy
        end
    end

    ##foreach(enumerate(range(-2Δs, step=Δs, length=6))) do (k, δts)
    ##    idx_in = @. min(n_ts, max(1, (idx_ts + δts)))
    ##    
    ##    dat_in = spec[:spect_mask][1:n_rg, idx_in] .≥ 0
    ##    tmp_spec = spec[:spect_mask][1:n_rg, idx_in][dat_in] .+ 1
    ##    tmp_dat = dat_in[:]
    ##    fuzzydat[idx_voo, tmp_dat, 1, k] = spec[var][idx_rad, tmp_spec]
    ##    
    ##end

    # 3.3) ++++ creating output variables:
    # defining output array: masked
    masked = @. ifelse(spec[:spect_mask][1:n_rg, idx_ts] ≥ 0, true, false)

    # 4) ++++ Optional Normalization:
    # converting features array into normalized array [0, 1]
    NORMALIZE && (features = voodoo.η₀₁(features))

    # 5) ++++ Creating output Dictionary:
    Xout = Dict(:feature=>features,
                :masked => masked,
                :idx_ts => idx_ts)
    
    # 5.1) Checking if optional adjustment of Doppler spectrum is performed:
    adjustDoppler && merge(Xout, Dict(:νₙ => Vn_voo))
    
    return Xout #features, masked, idx_ts
    
end
# ----/

# ***************************************************
# Function to adjust Doppler velocity into dimension
"""
Function to estimate the indexes corresponding to the Doppler spectrum
velocity array.
The adjusment is done based on the LIMRAD 94 radar Doppler spectrum with
a Nyquist velocity of 9.0 ms⁻¹ and 256 spectral bins.
USAGE:
julia> idx1, idx2 = index4DopplerDims(DopplerVel)
will return the indexes of DopplerVel[idx1] to be assigned to the
Feature array under indexes idx2.
or
julia> idx1, idx2 = index4DopplerDims(DopplerVel, adjust=true)
returns the indexes idx1 corresponding to sub-range of Feature array indexes
that match the ±Nyquist velocity of DopplerVel within ±νₙ of Feature array.

In both cases the assignation it has to be doen as:

julia> Feature[:, 1,1,idx2] = spec[:Znn][idx1, :]

"""
function index4DopplerDims(Vdp::Vector{<:AbstractFloat};
                           υₙ = 9.0, Nₙ=256, adjust=false)

    V_limrad = adjust ? ARMtools.DopplerVelocityVector(υₙ, Nₙ) : Vdp
    ΔNy = extrema(Vdp)
    NNy = length(Vdp)
    idx = adjust ? [findfirst(≥(x), V_limrad) for x ∈ ΔNy] : [1, Nₙ]
    Nidx = diff(idx) |> first
    Nidx += 1

    idxVdp = range(1, stop = NNy, length=Nidx) .|> x->round(Int, x)
    idxVnn = range(idx[1], stop=idx[2] , length=Nidx) .|> x->round(Int, x)

    return idxVdp, idxVnn, V_limrad
end
# ----/

end  # end of module
# ----!
# end of script
