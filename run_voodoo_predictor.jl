# script to run voodoo_predictor over a given day:

using ARMtools
using Dates
using ImageFiltering

include("Voodoo4AC3.jl")

# defining input variables:
const yy, mm, dd = 2019, 11, 18;
const SITE = "arctic-mosaic";
const PRODUCT = "KAZR/GE";

# initializing PATHs:
const BASE_PATH = "/projekt2/remsens/data_new/site-campaign";
const PATH_DATA = joinpath(BASE_PATH, SITE);

# reading spectrum data file for the day:

spec = let fname = ARMtools.getFilePattern(PATH_DATA, "$(PRODUCT)/SPECCOPOL", yy, mm, dd)
    !isfile(fname) && error("Spectrum files for given Date do not exist")
    ARMtools.readSPECCOPOL(fname)
end

# defining spectrum variables to adapt: e.g. SNR, Znn
spec_var = [:Znn, :SNR];
spec_params = Dict(:Znn=>(-90, -20), :SNR=>80);

XdB = voodoo.adapt_RadarData(spec; var=spec_var, AdjustDoppler=true);

X = Dict(:features=>voodoo.η₀₁(XdB[:features], ηlim = spec_params));

predict_var = voodoo.MakePrediction(X; masked=XdB[:masked]);

# re-sampling the full grid to output grid:
TIME_STEP = Second(30);
cln_time  = round(spec[:time][1],TIME_STEP):TIME_STEP:round(spec[:time][end],TIME_STEP);

HEIGHT_STEP = 30; # meters
cln_height = floor(spec[:height][1]):HEIGHT_STEP:ceil(spec[:height][end]);

cln_predic = fill(NaN32, length(cln_height), length(cln_time));

jdx = [abs.(h .- spec[:height]) |> argmin for h ∈ cln_height]
idx = argmin(abs.(t .- spec[:time]))

for h ∈ cln_height
    
    for t ∈ cln_time
        

    end
end

# end of script.
