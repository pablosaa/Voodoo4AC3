# script to convert the KAZR spectrum to Zarr file compatible with Voodoo

# Including ARM toolbox to read data:
include(joinpath(homedir(), "LIM/repos/ARMtools.jl/src/ARMtools.jl"))
include(joinpath(homedir(), "LIM/repos/CloudnetTools.jl/src/CloudnetTools.jl"))

# Defining constants:
ARM_SITE = "utqiagvik-nsa"
PATH = joinpath(homedir(), "LIM/remsens", ARM_SITE)
RADARPRO = "KAZR/ARSCL"
SPECTPRO = "KAZR/SPECCOPOL"
CLNET = joinpath(homedir(), "LIM/data/CloudNet/output")
# Date and hour to be read (spectrum data comes in hourly data files)

yy = 2019;
mm = 1;
dd = 27;
hh = 03;

# reading CloudNet categorize data file:
clnfile = ARMtools.getFilePattern(CLNET, "CEIL10m", yy, mm, dd; submonth=true)
cln = CloudNet.readCLNFile(clnfile);

# reading Radar Data
#rafile = ARMtools.getFilePattern(PATH, RADARPRO, yy, mm, dd)
#kazr = ARMtools.getKAZRData(rafile);

# reading Spectrum Data
spfile = ARMtools.getFilePattern(PATH, SPECTPRO, yy, mm, dd; hh)
#spfile = "/home/pablo/LIM/data/utqiagvik-nsa/KAZR/SPECCOPOL/nsakazrspeccmaskgecopolC1.a0.20190127.010010.nc";
spec = ARMtools.readSPECCOPOL(spfile);
nt, nh = size(spec[:spect_mask]);

## MATCHING the CloudNet time vector with the Spectrum time vector:
ii = #select indexes in cln[:time] corresponding to spec[:time];

## TRYING with PyCall
using PyCall, Dates
xr = pyimport("xarray")
da = pyimport("dask.array")

## *** To fill data with KAZR SPEC ****
# For coordinates:
my_coor = Dict(
    :dt => dt, #    (dt) int64 0 29999542 59999085 ... 
    :nbits => collect(Int64, 0:5), #   (nbits) int64 0 1 2 3 4 5
    :nchannels => collect(Int64, 0:5), # (nchannels) int64 0 1 2 3 4 5
    :npol => collect(Int64, 0:1), #      (npol) int64 0 1
    :nsamples => collect(Int64, range(0, stop=nsam-1)),  #  (nsamples) int64 0 1 2 3 4 5 ... 11855 11856 11857 11858 11859
    :nvelocity => collect(Int64, range(0, stop=nvel-1)), #  (nvelocity) int64 0 1 2 3 4 5 6 ... 250 251 252 253 254 255
    :rg => spec[:height], #    (rg) float32 119.247086 149.05884 ... 11924.608 11964.357
    :ts => spec[:time],
);

# For Variables:
my_vars = Dict(
    :features => ([:nsamples, :nvelocity, :nchannels, :npol], FeaturesArray), #    (nsamples, nvelocity, nchannels, npol) float32 dask.array<shape=(11860, 256, 6, 2), chunksize=(2965, 64, 2, 1)>
    :masked => ([:ts, :rg], MastekArray), # bool dask.array<shape=(120, 292), chunksize=(120, 292)>
    :multitargets => ([:nsamples, :nbits], MultiTargets), # float32 dask.array<shape=(11860, 6), chunksize=(5930, 6)>
    :targets => (:nsamples, TargetsVec),
);

# For Attributes:
my_att = Dict(
    :dt_unit       => "date",
    :dt_unit_long  => "Datetime format",
    :nbits_unit    => "Number of Cloudnet category bits",
    :nchannels_unit=> "Number of stacked spectra",
    :npol_unit     => "Number of polarizations",
    :nsamples_unit => "Number of samples",
    :nvelocity_unit=> "Number of velocity bins",
    :rg_unit       => "m",
    :rg_unit_long  => "Meter",
    :ts_unit       => "sec",
    :ts_unit_long  => "Unix time, seconds since Jan 1. 1979",
);

# Creating XARRAY Dataset:
VariableND = xr.Dataset(
    data_vars = my_vars,
    coords = my_coor,
    attrs = my_att,
);

# ØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØ
# the following doesn't work!!
# Storing as Zarr
VariablesND.to_zarr(store=filename_zarr)

p = "data/kazrspec.zarr";

Z = zcreate(eltype(spec[:spect_mask]), nt, nh, path=p, name="spec_mask");

Z[:,:] = spec[:spect_mask]);
# ØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØ
# the following just as example:
temperature = 15 .+ 8 .* randn(2, 2, 3)
precipitation = 10 .* rand(2, 2, 3)
lon = [-99.83 -99.32; -99.79 -99.23]
lat = [42.25 42.21; 42.63 42.59]
thetime = DateTime(2014,09,06):Hour(1):DateTime(2014,09,06,2,0,0);
reference_time = DateTime(2014,09,05)

ds = xr.Dataset(
data_vars=Dict(
        :temperature=>(["x", "y", "time"], temperature),
        :precipitation=>(["x", "y", "time"], precipitation),
),
coords=Dict(
        :lon=>(["x", "y"], lon),
        :lat=>(["x", "y"], lat),
        :time=>thetime,
        :reference_time=>reference_time,
),
attrs=Dict(:description=>"Weather related data."),
)

ds = xr.Dataset(
data_vars=Dict(
        :temperature=>  temperature,
        :precipitation=>precipitation,
),
coords=Dict(
        :lon=>lon,
        :lat=>lat,
        :time=>thetime,
),
attrs=Dict(:description=>"Weather related data."),
)
# ØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØ


# end of script
