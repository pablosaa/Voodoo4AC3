# script to run voodoo_predictor over a given day:

using ARMtools
using Dates
using ImageFiltering
using Statistics
using CloudnetTools

include("Voodoo4AC3.jl")

# aux function:
function vec2range(V)
       δ = diff(V) |> median
       d = Int(δ)
       return range(V[1], step=d, stop=V[end])
end

# defining input variables:
const yy, mm, dd = 2019, 11, 18;
const SITE = "arctic-mosaic";
const PRODUCT = "KAZR/GE";

# initializing PATHs:
const BASE_PATH = "/projekt2/remsens/data_new/site-campaign";
const PATH_DATA = joinpath(BASE_PATH, SITE);

# reading spectrum data file for the day:

spec = let fname = ARMtools.getFilePattern(PATH_DATA, "$(PRODUCT)/SPECCOPOL", yy, mm, dd, hh=16)
    !isfile(fname) && error("Spectrum files for given Date do not exist")
    ARMtools.readSPECCOPOL(fname)
end

# defining spectrum variables to adapt: e.g. SNR, Znn
spec_var = [:Znn, :SNR];
spec_params = Dict(:Znn=>(-90, -20), :SNR=>70);

XdB = voodoo.adapt_RadarData(spec; var=spec_var); #, AdjustDoppler=true);

X = Dict(:features=>voodoo.η₀₁(XdB[:features], ηlim = spec_params));

predict_var = voodoo.MakePrediction(X; masked=XdB[:masked]);

# reading Cloudnet Categorization file:
#cate_file = ARMtools.getFilePattern("/home/psgarfias/LIM/data","Voodoo", yy, mm, dd, file_ext="categorize.nc");
cate_file = "/home/psgarfias/LIM/data/Voodoo/$(yy)$(mm)$(dd)_testbed_categorize.nc";
cln = CloudnetTools.readCLNFile(cate_file);

# re-sampling the full grid to output grid:
TIME_STEP = Second(30);
cln_time  = cln[:time]; #round(spec[:time][1],TIME_STEP):TIME_STEP:round(spec[:time][end],TIME_STEP);
nn_time = length(cln_time);

HEIGHT_STEP = 30; # meters
cln_height = cln[:height] #floor(spec[:height][1]):HEIGHT_STEP:ceil(spec[:height][end]);
nn_height = length(cln_height);

cln_predic = Dict(k=>fill(NaN32, nn_height, nn_time) for k ∈ spec_var);

jdx = [abs.(h .- spec[:height]) |> argmin for h ∈ cln_height];
idx = [argmin(abs.(t .- spec[:time])) for t ∈ cln_time];

#iijj = (vec2range(jdx), vec2range(idx));

#TT = let ker2d=ImageFiltering.Kernel.gaussian((5,5))
#    Dict(k=> let garbage = imfilter(var[:,:,2], ker2d)
#             garbage[jdx, idx]
#         end
#         for (k, var) ∈ predict_var)
#end



foreach(spec_var) do V
    cln_predic[V] = mapwindow(median!, predict_var[V][:,:,2], (5,5) )
end

# Generating cloudnet categorization file:
let Xpre=sqrt.(cln_predic[:Znn].*cln_predic[:SNR])[jdx, idx]
    #voodoo2cloudnet.voodoo2categorize(Xpre, cate_file)
end


#for (j,h) ∈ enumerate(jdx)
 #   for (i,t) ∈ enumerate(idx)
  #      foreach(spec_var) do V
   #         cln_predic[V][j,i] = predict_var[V][h,t,2]
    #    end
  #  end
#end

# end of script.
