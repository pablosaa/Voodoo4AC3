#!/opt/julia-1.7.2/bin/julia

# script to run the Voodoo predictor with fixed parameters

using ARMtools
using Dates
using Printf
using Zarr
using CloudnetTools

include("Voodoo4AC3.jl")

# defining date and time to analyze:
yy = 2020 #2019
mm = 04 #01
dd = 15 #01

	
# defining Site information:
SITE = "arctic-mosaic" # or "utqiagvik-nsa"
PRODUCT = "MWACR"  #"KAZR" # or "KAZR"

# defining Directory paths:
const BASE_PATH = joinpath(homedir(), "LIM/data") #remsens")
const PATH_DATA = joinpath(BASE_PATH, SITE) #"LIM/remsens/utqiagvik-nsa/KAZR")
const PROD_TYPE = "$(PRODUCT)/SPECCOPOL"
const OUTDAT_PATH = joinpath(pwd(), "data", @sprintf("%02d/%02d/%02d", yy, mm, dd))

# Limits for normalization based on variable:
spec_params = Dict(:Znn=>(-100, -30), :SNR=>(0, 45));
spec_var = :Znn

# Reading CloudNet data input:
const CLNET_PATH = joinpath(homedir(), "LIM/data/CloudNet/arctic-mosaic/TROPOS")
const clnet_file = ARMtools.getFilePattern(CLNET_PATH, "categorize", yy, mm, dd);
clnet = CloudnetTools.readCLNFile(clnet_file);


# running over hours:
for hh in (18:23) #04
    spec_file = ARMtools.getFilePattern(PATH_DATA, PROD_TYPE , yy, mm, dd; hh=hh)
    !isfile(spec_file) && (warn("spectrum data does not exist!"); )

    spec = ARMtools.readSPECCOPOL(spec_file);

    # indexes vector containing Cloudnet data corresponding to the spectra time vector:
    clnet_it = @. spec[:time][1] ≤ clnet[:time] ≤ spec[:time][end];


    XdB, masked, i_ts = voodoo.adapt_RadarData(spec,
                                               cln_time=clnet[:time][clnet_it],
                                               var=spec_var,
                                               Δs=3);


    # Normalization of spectral data:
    X = voodoo.η₀₁(XdB[:,:,:,1:2:end], ηlim = spec_params[spec_var]);

    # Calling the predictor:
    sizedims = size(masked)
    predict_var = voodoo.MakePrediction(X, sizeout=sizedims);

    # Saving the output predicted array
    zfilen = let fn = basename(spec_file)
        replace(fn, "M1.a1."=>".voodoo_$(spec_var).",
                ".nc"=>".zarr") |> x-> joinpath(OUTDAT_PATH, x)
    end
    Zcompressor = Zarr.BloscCompressor(cname="zstd", clevel=3, shuffle=true)
    
    if isfile(zfilen)
	append!(Zₚᵣₑ, predict_var[:,:,2])
    else
	Zₚᵣₑ = zcreate(eltype(predict_var),
                       sizedims...,
                       chunks = sizedims,
		       compressor = Zcompressor,
                       path = zfilen)
	Zₚᵣₑ[:,:] = predict_var[:,:,2];
    end
    
end # end over hour

# end of file
