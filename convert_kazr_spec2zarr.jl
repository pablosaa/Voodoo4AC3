# script to convert the KAZR spectrum to Zarr file compatible with Voodoo

# Including ARM toolbox to read data:
include(joinpath(homedir(), "LIM/repos/ARMtools.jl/src/ARMtools.jl"))
include(joinpath(homedir(), "LIM/repos/CloudnetTools.jl/src/CloudnetTools.jl"))

# Defining constants:
ARM_SITE = "utqiagvik-nsa"
PATH = joinpath(homedir(), "LIM/data", ARM_SITE)
RADARPRO = "KAZR/ARSCL"
SPECTPRO = "KAZR/SPECCOPOL"
CLNET = joinpath(homedir(), "LIM/data/CloudNet/output")
# Date and hour to be read (spectrum data comes in hourly data files)

yy = 2019;
mm = 1;
dd = 27;
hh = 06;

# reading CloudNet categorize data file:
clnfile = ARMtools.getFilePattern(CLNET, "CEIL10m", yy, mm, dd; submonth=true)
cln = CloudNet.readCLNFile(clnfile);

# reading Radar Data
#rafile = ARMtools.getFilePattern(PATH, RADARPRO, yy, mm, dd)
#kazr = ARMtools.getKAZRData(rafile);

# reading Spectrum Data
spfile = ARMtools.getFilePattern(PATH, SPECTPRO, yy, mm, dd; hh)
spec = ARMtools.readSPECCOPOL(spfile);

## MATCHING the CloudNet time vector with the Spectrum time vector:
#select indexes ii from cln[:time] within the span of spec[:time];
using Dates
ii = filter(i-> spec[:time][1] ≤ i ≤ spec[:time][end], cln[:time]);
# select the spec[:time] corresponding to exact cln[:time] values:
thr_ts = minimum(diff(spec[:time]));

idx_ts = map(x->findfirst((abs.(x .- spec[:time])) .< thr_ts), ii);

# The spectrum time corresponding to cloudnet is then spec[:time][idx_ts]

# The 6 latest spectrum for every time stamp are:
ii_6spect = map(j -> range(j<5 ? 1 : j-5, length=6), idx_ts);

# The 6 spectra for the time stamp are retrieved as:
# > idx_hgt = 100;  # this is the height index
# > idx_tit = 20;
# iteration over i the 6 spectra correspondint to spec[:time][idx_ts]:
# > idx_alt = 1 .+ filter(x->x ≥ 0, spec[:spect_mask][idx_hgt, ii_6spect[idx_tit]]);

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Creating Matrix for Voodoo:
# * feature[nsamples, nvelocity, nchannel, npol]
# * tagets[nsamples, nvelocity, nchannel, npol]
# * masked[ts, rg]
#
n_rg = 292; # n_rg,_ = size(cln[:Z])
n_ts = length(idx_ts);
masked = Array{Bool}(undef, n_rg, n_ts) .= false;

# filling masked array with (true/false) values:
@. masked[!isnan(cln[:Z][1:n_rg, idx_ts])] = true;

# filling the features array with spectrum data:
n_samples = sum(masked);
n_velocity = length(spec[:vel_nn]);
n_channels = 6;
n_pol = 2;
features = Array{Float32}(undef, n_samples, n_velocity, n_channels, n_pol) .= NaN32;

i_sample = 0;
for i_rg ∈ (1:n_rg)
    for i_ts ∈ (1:n_ts)
        if masked[i_rg, i_ts]
            global i_sample += 1
            idx_alt = 1 .+ filter(x->x ≥ 0, spec[:spect_mask][3+i_rg, ii_6spect[i_ts]])
            n_alt = length(idx_alt)
            features[i_sample, :, 1:n_alt, 1] = spec[:η_hh][:, idx_alt];
        end
    end
end

# Estimating time variables:
dt = Dates.value.(ii .- ii[1]);

## TRYING with PyCall
using PyCall
xr = pyimport("xarray")
da = pyimport("dask.array")

## Converting Julia variables to dask:
FeaturesArray = da.from_array(features,
                              chunks = (floor(n_samples/4),
                                        floor(n_velocity/4),
                                        2, 1));
MaskedArray = da.from_array(masked, chunks=(n_rg, n_ts));
MultiTargets = da.from_array(Array{Float32}(undef, n_samples, n_channels),
                             chunks = (floor(n_samples/2), n_channels))
TargetsVec = da.from_array(Array{Int32}(undef, n_samples), chunks=(n_samples));

## *** To fill data with KAZR SPEC ****
# For coordinates:
#    (dt) int64 0 29999542 59999085 ...
#   (nbits) int64 0 1 2 3 4 5
# (nchannels) int64 0 1 2 3 4 5
#      (npol) int64 0 1
#  (nsamples) int64 0 1 2 3 4 5 ... 11855 11856 11857 11858 11859
#  (nvelocity) int64 0 1 2 3 4 5 6 ... 250 251 252 253 254 255
#    (rg) float32 119.247086 149.05884 ... 11924.608 11964.357
my_coor = Dict(
    :dt => dt,
    :nbits => collect(Int64, 0:5),
    :nchannels => collect(Int64, 0:5),
    :npol => collect(Int64, 0:1),
    :nsamples => collect(Int64, range(0, stop=n_samples-1)),
    :nvelocity => collect(Int64, range(0, stop=n_velocity-1)),
    :rg => spec[:height][3:3+n_rg-1],
    :ts => spec[:time][idx_ts],
);

# For Variables:
# (nsamples, nvelocity, nchannels, npol) float32 dask.array<shape=(11860, 256, 6, 2), chunksize=(2965, 64, 2, 1)>
# bool dask.array<shape=(120, 292), chunksize=(120, 292)>
# float32 dask.array<shape=(11860, 6), chunksize=(5930, 6)>
my_vars = Dict(
    :features => ([:nsamples, :nvelocity, :nchannels, :npol], FeaturesArray),
    :masked => ([:rg, :ts], MaskedArray),
    :multitargets => ([:nsamples, :nbits], MultiTargets),
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

filename_zarr = "data/kazrspec.zarr";
VariableND.to_zarr(store=filename_zarr)


# ØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØ
# the following doesn't work!!
# Storing as Zarr

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
