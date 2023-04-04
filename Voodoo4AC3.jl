# File containing module for using Voodoo with ARM radars:
#

module voodoo

using PyCall
using TOML
using Dates
using ARMtools

# *******************************************************************
function MakePrediction(X_in::Dict; masked = [])
    predict_var = Dict(k=> MakePrediction(v, masked=masked) for (k,v) ∈ X_in[:features])

    return predict_var
end
function MakePrediction(X_in::Array{<:AbstractFloat, 4};
        masked = [])
    
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

    # Final output array with predictions (range x time):
    predict_var = let xy = size(masked)
        if NCLASSES*sum(masked[:]) == (prod∘size)(X_out)
                index_mask = reshape(cumsum(masked[:]), xy)
	        AA = fill(NaN32, (xy..., NCLASSES))
	        for i ∈ (1:xy[1])
		        for j ∈ (1:xy[2])
			        !masked[i,j] && continue
			        AA[i,j,:] = X_out[index_mask[i,j], :]
		        end
	        end
	        AA
        else
                X_out
        end
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
function η₀₁(η::Dict; ηlim=Dict())
    thelims = Dict(k=>ifelse(haskey(ηlim,k), ηlim[k], () ) for (k,v) ∈ η)
    H_out = Dict(k=>η₀₁(v; ηlim=thelims[k]) for (k,v) ∈ η)
    return H_out
end
function η₀₁(η::Array{<:AbstractFloat, 4}; ηlim=())
    (η0, η1) = if typeof(ηlim) <: NTuple{2, Real}
        sort([ηlim...])
    elseif typeof(ηlim) <: Real
        (0, ηlim)
    elseif isempty(ηlim)
        extrema(η)
    else
        @error "ηlim needs to be (η₀, η₁) or η₁ or ()"
    end
    
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
julia> feature, masked, idx_time = adapt_RadarData(spec, var=(Znn=(-100, -50), SNR=40), Δs=3)

INPUT:
* spec::Dict() -> Variable with spectrum data as returned from ARMtools.jl
* var::NamedTuple (Znn=(min, max), SNR=max) -> spectrum variable with normalization ranges, (default) Znn=(-90,-20) & SNR=60
* cln_time::Vector -> DateTime vector with values to consider as centered
* Δs::Int -> step for filling the 6 spectra, default 5
* LimHm::Tuple{Real, Real} -> (min, max) altitude to consider in m, default (0, 8000m)
* AdjustDoppler::Bool -> true if Doppler spectrum adjusted to 256 Vooodoo default.
* Normalize::Bool -> flag to normalize [0,1] the output. Default false

OUTPUT:
Function returns a Dictionary with following keys:
* features::Dict(:Znn, :SNR) with Array{T}(num_samples, 1, 6, n_dopplervelocity) -> data array adapted
* masked::Array{Bool}(num_time, num_heights) -> true for considered spectrum
* idx_time::Vector{Int} -> indexes of the spec[:time] considered in output
* idxrng::Vector{Int} -> indexes of spec[:height] considered in output
"""
function adapt_RadarData(spec::Dict;
                  var = :Znn,     #NamedTuple{(:Znn, :SNR), Tuple{Tuple{Int, Int}, Int}}=(Znn=(-91, -20), SNR=60),
                         cln_time::Vector{DateTime} = Vector{DateTime}(undef,0),
                         Δs::Int = 5,
                         LimHm::Tuple{Real, Real} = (NaN32, NaN32),
                         AdjustDoppler = false,
                         Normalize::Dict=Dict())
                         
    
    ## 0) +++
    # Checking initialized variables:
    var = if typeof(var)<:Symbol && haskey(spec, var)
        [var]
    elseif typeof(var)<:Vector{Symbol}
        [v for v in var if haskey(spec,v)]
    else
        @warn "Optional variable var needs to be Symbol or Vector{Symbol}"
        [:Znn]
    end

    #if ~(!isempty(var[:Znn]) && (typeof(var[:Znn]) <: Tuple{Real, Real} ))
    #    @warn "input variable var[:Znn] needs to be type Tuple(Real, Real). Ignored!"
    #    var[:Znn] =()
    #else
    #    Normalize = true
    #end
    #if haskey(var, :SNR) && !(typeof(var[:SNR]) <:Real )
    #    @warn "input variable var[:SNR] need to be ::Real. Ignored!"
    #    delete!(var, :SNR)
    #else
    #    Normalize = true
    #end

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
    # indexes containing Cloudnet data corresponding to the spectra time:
    clnet_it = ifelse(isempty(cln_time), : , @. spec[:time][1] ≤ cln_time ≤ spec[:time][end])

    # input spectrum spec[:time] indexes to match cln_time time series:
    idx_ts=:;
    
    if !isempty(cln_time)
        idx_ts = let thr_ts = diff(spec[:time]) |> x->min(x..., Millisecond(2500) )
            tmp = [findfirst(abs.(x .- spec[:time]) .< 2thr_ts) for x ∈ cln_time[clnet_it]]
            filter(!isnothing, tmp)
        end
    end
    # checking whether "nothing" is found in idx_ts (when nearest time is less than 2*thr)
    idx_ts .|> isnothing |> any && error("Time steps larger than twice the threshold!!")
    n_ts = length(spec[:time][idx_ts]);

    # -------------------------------
    ## 3.1) ++++
    # Definition of dimension for output variables:
    # * Height dimension:
    idxrng = ifelse(map(isnan, LimHm) |> any, : ,  @. LimHm[1] ≤ spec[:height] ≤ LimHm[2])
    n_rg = length(spec[:height][idxrng]) #findlast(idxrng) #≤(MaxHkm), 1f-3spec[:height]); #250

    # * Number of samples:
    n_samples = filter(≥(0), spec[:spect_mask][idxrng, idx_ts]) |> length
    #n_samples = length(spec[:spect_mask][1:n_rg, idx_ts])

    # * Number of Doppler spectral bins dimension (voodoo only support 256):
    n_vel = AdjustDoppler ? 256 : length(spec[:vel_nn])

    ## 3.1) ++++
    # Creating the features output array:
    # according to voodoo predictor, should be (n_samples, n_ch=1, n_ts=6, n_vel)
    features = Dict(x=>fill(NaN32, (n_samples, 1, 6, n_vel)) for x in var )
        
    ## 3.2) ++++
    idx_rad, idx_voo, Vn_voo = index4DopplerDims(spec[:vel_nn],
                                                 adjust = AdjustDoppler)

    ## 3.3) ++++
    # Filling feature array with 6 consecutive spectra centered at cln_time:
    NN = lastindex(spec[:time])

    foreach(var) do kk
        # helper variable to fill data into features: (n_vel, n_samples, n_ch, n_ts)
        fuzzydat = PermutedDimsArray(features[kk], (4, 1, 2, 3)) 

        let i0 = 1
            time_iter = ifelse(typeof(idx_ts) <: Colon, 1:n_ts , idx_ts)
            foreach(time_iter) do k
                len_in = findall(≥(0), spec[:spect_mask][idxrng, k])
                iall = range(i0, length=length(len_in) )
                
                δts = k .+ range(-2Δs, step=Δs, length=6) |> x-> min.(NN, max.(1, x))
                foreach(enumerate(δts)) do (j, its)
                    dat_in = spec[:spect_mask][idxrng, its][len_in] .+ 1
                    for (i, x) ∈ zip(iall, dat_in)
                        x<1 && continue
                        fuzzydat[idx_voo, i, 1, j] = spec[kk][idx_rad, x]
                    end
                end
                i0 += length(len_in) 
           end
        end
    end

    # 3.3) ++++ creating output variables:
    # defining output array: masked
    masked = @. ifelse(spec[:spect_mask][idxrng, idx_ts] ≥ 0, true, false)

    # 4) ++++ Optional Normalization:
    # converting features array into normalized array [0, 1]
    !isempty(Normalize) && foreach(pairs(Normalize)) do (k,v)
        haskey(features, k) && (features[k] = voodoo.η₀₁(features[k], ηlim = v))
    end
    
    # 5) ++++ Creating output Dictionary:
    Xout = Dict(:features =>features,
                :masked => masked,
                :idx_ts => idx_ts,
                :limrng => LimHm,
                :clnet_it => clnet_it)
    
    # 5.1) Checking if optional adjustment of Doppler spectrum is performed:
    AdjustDoppler && merge(Xout, Dict(:νₙ => Vn_voo))
    
    return Xout
    
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
    Nₙ = adjust ? Nₙ : length(Vdp)
    idx = adjust ? [findfirst(≥(x), V_limrad) for x ∈ ΔNy] : [1, Nₙ]
    Nidx = diff(idx) |> first
    Nidx += 1

    idxVdp = range(1, stop = NNy, length=Nidx) .|> x->round(Int, x)
    idxVnn = range(idx[1], stop=idx[2] , length=Nidx) .|> x->round(Int, x)

    return idxVdp, idxVnn, V_limrad
end
# ----/

# **********************************************************
"""
Function to store the features Array as Zarr file for VOODOO
training and predictions.
"""
function to_zarr(Xin::Dict, clnet::Dict, zfilen::String)

    # Loading necessary packages:
    xr = pyimport("xarray")
    da = pyimport("dask.array")

    # Defining size for Zarr file dimentions:
    sizedims = size(Xin[:features])
    n_samples, _, n_spec, n_velocity = sizedims;
    n_rg, n_ts = size(Xin[:masked])

    # Vector{Bool} with indexes of considered spectrum time in cloudnet:
    clnet_it = view(Xin[:clnet_it], :)

    # Vector{Bool} with indexes of considered heights: 
    rng_it = let H = Xin[:limrng] .+ clnet[:alt]
        H[1] .≤ clnet[:height] .≤ H[2]
    end
    
    n_rg != sum(rng_it) && (@error "$n_rg, $(sum(rng_it)) of heigths not compatible!")
    
    ## Converting Julia variables to dask:
    ## ***** For N-D variales ***********
    FeaturesArray = let tmp = dropdims(Xin[:features]; dims=2)
        features = PermutedDimsArray(tmp, (1,3,2))
        da.from_array(features, chunks = (floor(n_samples/4), floor(n_velocity/4), 2));
    end
    
    Xin[:targets] = let idx_dat = findall(==(1), Xin[:masked])
        clnet[:CLASSIFY][rng_it, clnet_it][idx_dat]
    end

    
    # Unix epoch seconds from start of day:
    ts = datetime2unix.(clnet[:time][clnet_it]);

    # *** coordinates info:
    infoND_coor = Dict(
        :dt => clnet[:time][clnet_it],
        :nchannels => collect(Int64, 0:5),
        :nsamples => collect(Int64, range(0, stop=n_samples-1)),
        :nvelocity => collect(Int64, range(0, stop=n_velocity-1)),
        :rg => clnet[:height][rng_it],
        :ts => ts,
    );

    # *** variables info:
    clnetVAR2dask(A) = da.from_array(PermutedDimsArray(eltype(A)<:Bool ? A : Int32.(A), (2,1)) )

    infoND_vars = Dict(
        :class => ([:ts, :rg], clnetVAR2dask(clnet[:CLASSIFY][1:n_rg, clnet_it]) ),
        :features => ([:nsamples, :nvelocity, :nchannels], FeaturesArray),
        :masked => ([:ts, :rg], clnetVAR2dask(.!Xin[:masked]) ),
        :status => ([:ts, :rg], clnetVAR2dask(clnet[:DETECTST][1:n_rg, clnet_it]) ),
        :targets => ([:nsamples], da.from_array(Float32.(Xin[:targets])) ),
    );

    # *** Attributes info:
    infoND_att = Dict(
        :dt_unit       => "date",
        :dt_unit_long  => "Datetime format",
        :nchannels_unit=> "Number of stacked spectra",
        :nsamples_unit => "Number of samples",
        :nvelocity_unit=> "Number of velocity bins",
        :rg_unit       => "m",
        :rg_unit_long  => "Meter",
        :ts_unit       => "sec",
        :ts_unit_long  => "Unix time, seconds since Jan 1. 1970",
    );

    # *** Creating N-D XARRAY Dataset:
    ND_dataset = xr.Dataset(
        data_vars = infoND_vars,
        coords = infoND_coor,
        attrs = infoND_att,
    );

    # *** storing Dataset to zarr file:
    try
        ND_dataset.to_zarr(store = zfilen, mode="w")
        return true
    catch
        @warn "Cannot create Zarr $zfilen"
        return false
    end
end
# ----/

end  # end of module

module voodoo2cloudnet
using NCDatasets
using CloudnetTools

function voodoo2categorize(Xpre::Matrix, cate_file::String; modelname="mykakes")
        NCDataset(cate_file, "a") do nc
	cate=nc[:category_bits];
        foreach(eachindex(Xpre)) do idx
            Xpre[idx]>0 && (cate[idx] |= eltype(cate)(1) )
	end
        nc.attrib["postprocessor"] = "Voodoo_v2.0, Modelname: $(modelname)"
        end
end

end  # end of module voodoo2cloudnet
# ----!
# end of script
