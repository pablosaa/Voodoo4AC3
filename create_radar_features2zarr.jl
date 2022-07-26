#!/opt/julia-1.7.2/bin/julia

# script to create Zarr files with features for Voodoo training purposes:

using ARMtools
using Dates
using Printf
using CloudnetTools

include("Voodoo4AC3.jl")

# defining date and time to analyze:
years = (2020) #2019
months = (04) #01
days = (15) #01

!isempty(ARGS) && foreach(ARGS) do in
    ex = Meta.parse(in)
    eval(ex)
end
	
# defining Site information:
SITE = "arctic-mosaic" # or "utqiagvik-nsa"
PRODUCT = "KAZR/GE" #"MWACR"  # or "KAZR"

# defining Directory paths:
const BASE_PATH = joinpath(homedir(), "LIM/data") #remsens")
const PATH_DATA = joinpath(BASE_PATH, SITE) #"LIM/remsens/utqiagvik-nsa/KAZR")
const PROD_TYPE = "$(PRODUCT)/SPECCOPOL"
const CLNET_PATH = joinpath(homedir(), "LIM/data/CloudNet/arctic-mosaic/TROPOS")

# Limits for normalization based on variable:
spec_params = Dict(:Znn=>(-100, -10), :SNR=>(0, 70));
spec_var = :Znn

for yy ∈ years
    for mm ∈ months
        for dd ∈ days

            # DATE dependent parameters:
            local OUTDAT_PATH = joinpath(pwd(), "data", @sprintf("%02d/%02d/%02d", yy, mm, dd))

            # Reading CloudNet data input:
            local clnet_file = ARMtools.getFilePattern(CLNET_PATH, "categorize", yy, mm, dd);
            clnet = CloudnetTools.readCLNFile(clnet_file);

            # Reading all available files for the day:
            #spec_file = ARMtools.getFilePattern(PATH_DATA, PROD_TYPE , yy, mm, dd; hh=6)
            files_of_day = ARMtools.getFilePattern(PATH_DATA, PROD_TYPE , yy, mm, dd; hh=6)

            # running over hours:
            #for spec_file ∈ files_of_day #hh in (18:23) #04
            let spec_file=files_of_day
                !isfile(spec_file) && (warn("spectrum data does not exist!"); )

                spec = ARMtools.readSPECCOPOL(spec_file);

                # indexes vector containing Cloudnet data corresponding to the spectra time vector:
                #clnet_it = @. spec[:time][1] ≤ clnet[:time] ≤ spec[:time][end];

                # XdB is Dict with adapted feature, mask and index of spec time to cloudnet:
                XdB = voodoo.adapt_RadarData(spec,
                                             cln_time=clnet[:time],
                                             var=spec_var,
                                             MaxHkm=10.0,
                                             Δs=3,
                                             norm_params=spec_params);

                # Creating output zarr file name:
                zfilen = let fn = basename(spec_file)
                    replace(fn, "M1.a1."=>".voodoo_$(spec_var).",
                            ".nc"=>".zarr") |> x-> joinpath(OUTDAT_PATH, x)
                end

                voodoo.to_zarr(XdB, clnet, zfilen)

            end # end over hour
        end
    end
end


# Normalization of spectral data:
    #X = voodoo.η₀₁(XdB[:,:,:,1:2:end], ηlim = spec_params[spec_var]);
    #X = voodoo.η₀₁(XdB[:feature], ηlim = spec_params[spec_var]);

 #   # Defining size for Zarr file dimentions:
 #   sizedims = size(XdB[:feature])
 #   n_samples, _, n_spec, n_velocity = sizedims;
 #   n_rg, n_ts = size(XdB[:masked])
 #   ##predict_var = voodoo.MakePrediction(X, sizeout=sizedims);
 #
 #   # *******************************************************
 #   # Saving the data as zarr file:
 #
 #   ## TRYING with PyCall
 #   using PyCall
 #   xr = pyimport("xarray")
 #   da = pyimport("dask.array")
 #
 #   ## Converting Julia variables to dask:
 #   ## ***** For N-D variales ***********
 #   FeaturesArray = let tmp = dropdims(X; dims=2)
 #       features = PermutedDimsArray(tmp, (1,3,2))
 #       da.from_array(features, chunks = (floor(n_samples/4), floor(n_velocity/4), 2));
 #   end
 #   XdB[:targets] = let idx_dat = findall(==(1), XdB[:masked])
 #       clnet[:CLASSIFY][1:n_rg, clnet_it][idx_dat]
 #   end
 #
 #   # seconds from start of day:
 #   ts = datetime2unix.(clnet[:time][clnet_it]);
 #
 #   # *** coordinates info:
 #   infoND_coor = Dict(
 #       :dt => clnet[:time][clnet_it],
 #       :nchannels => collect(Int64, 0:5),
 #       :nsamples => collect(Int64, range(0, stop=n_samples-1)),
 #       :nvelocity => collect(Int64, range(0, stop=n_velocity-1)),
 #       :rg => clnet[:height][1:n_rg], #[3:3+n_rg-1],
 #       :ts => ts,
 #   );
##,Dimensions:    (ts: 120, rg: 292, dt: 120, nsamples: 15216, nvelocity: 256,
##,                nchannels: 6)
##,Coordinates:
##,  * dt         (dt) datetime64[ns] 2021-02-05T09:00:14.999771 ... 2021-02-05T...
##,  * nchannels  (nchannels) int64 0 1 2 3 4 5
##,  * nsamples   (nsamples) int64 0 1 2 3 4 5 ... 15211 15212 15213 15214 15215
##,  * nvelocity  (nvelocity) int64 0 1 2 3 4 5 6 7 ... 249 250 251 252 253 254 255
##,  * rg         (rg) float32 119.2 149.1 178.9 ... 1.188e+04 1.192e+04 1.196e+04
##,  * ts         (ts) float64 1.613e+09 1.613e+09 ... 1.613e+09 1.613e+09


# *** variables info:
#convertCLNET2dask(A) = da.from_array(PermutedDimsArray(eltype(A)<:Bool ? A : Int32.(A), (2,1)) )
#
#infoND_vars = Dict(
#    :class => ([:ts, :rg], convertCLNET2dask(clnet[:CLASSIFY][1:n_rg, clnet_it]) ),
#    :features => ([:nsamples, :nvelocity, :nchannels], FeaturesArray),
#    :masked => ([:ts, :rg], convertCLNET2dask(.!XdB[:masked]) ),
#    :status => ([:ts, :rg], convertCLNET2dask(clnet[:DETECTST][1:n_rg, clnet_it]) ),
#    :targets => ([:nsamples], da.from_array(Float32.(XdB[:targets])) ),
#);

##,Data variables:
##,    class      (ts, rg) int32 dask.array<chunksize=(120, 292), meta=np.ndarray>
##,    features   (nsamples, nvelocity, nchannels) float32 dask.array<chunksize=(1902, 64, 2), meta=np.ndarray>
##,    masked     (ts, rg) bool dask.array<chunksize=(120, 292), meta=np.ndarray>
##,    status     (ts, rg) int32 dask.array<chunksize=(120, 292), meta=np.ndarray>
##,    targets    (nsamples) float32 dask.array<chunksize=(15216,), meta=np.ndarray>

# *** Attributes info:
#infoND_att = Dict(
#    :dt_unit       => "date",
#    :dt_unit_long  => "Datetime format",
#    :nchannels_unit=> "Number of stacked spectra",
#    :nsamples_unit => "Number of samples",
#    :nvelocity_unit=> "Number of velocity bins",
#    :rg_unit       => "m",
#    :rg_unit_long  => "Meter",
#    :ts_unit       => "sec",
#    :ts_unit_long  => "Unix time, seconds since Jan 1. 1970",
#);

##,Attributes:
##,    dt_unit:         date
##,    dt_unit_long:    Datetime format
##,    nchannels_unit:  Number of stacked spectra
##,    nsamples_unit:   Number of samples
##,    nvelocity_unit:  Number of velocity bins
##,    rg_unit:         m
##,    rg_unit_long:    Meter
##,    ts_unit:         sec
##,    ts_unit_long:    Unix time, seconds since Jan 1. 1979
##

# Creating N-D XARRAY Dataset:
#ND_dataset = xr.Dataset(
#    data_vars = infoND_vars,
#    coords = infoND_coor,
#    attrs = infoND_att,
#);

    
#ND_dataset.to_zarr(store = zfilen, mode="w")

    # **** old stuff:
#Zcompressor = Zarr.BloscCompressor(cname="zstd", clevel=3, shuffle=true)
    
#    if isfile(zfilen)
#	append!(Zₚᵣₑ, predict_var[:,:,2])
#    else
#Zₚᵣₑ = zcreate(eltype(X),
#               sizedims...,
#               chunks = FeaturesArray.chunksize,
#	       compressor = Zcompressor,
#               path = zfilen,
#               storagetype = DirectoryStore,
#               name = "features")
#Zₚᵣₑ[:,:,:,:] = FeaturesArray #predict_var[:,:,2];
#end
    


# end of file
