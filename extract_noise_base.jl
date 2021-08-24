# script to extract the Spectrum noise level

using Statistics
include(joinpath(homedir(), "LIM/repos/ARMtools.jl/src/ARMtools.jl"))

DATA_PATH = joinpath(homedir(), "LIM/remsens/utqiagvik-nsa/KAZR/SPECCOPOL/")
spec_file= joinpath(DATA_PATH, "2019/nsakazrspeccmaskgecopolC1.a0.20190126.100004.cdf");

# reading spectrum:
spec=ARMtools.readSPECCOPOL(spec_file);

# Spectrum Index to show:
i = 291
idx_alt = findall(spec[:spect_mask][:,i] .≥ 0);
idx= 1 .+ spec[:spect_mask][idx_alt, i];

using Plots
Plots.plot(spec[:vel_nn], spec[:height][idx_alt], spec[:η_hh][:,idx]',st=:heatmap, color=:berlin, title="$(spec[:time][i])")

# for specific Altitude and time span:
i_hgt = 490
tmp = findfirst(isapprox.(spec[:height][idx_alt], i_hgt, atol=15.0));
# index corresponding to height i_hgt [m]
idx_hgt = 1 .+ spec[:spect_mask][idx_alt[tmp], i];

η = spec[:η_hh][:, idx_hgt]
minmax_peak = extrema(η)
mean_peak = mean(η)

std_peak = std(η) + mean_peak
p1 = plot(spec[:vel_nn],  η, title="$i_hgt [m] at $(spec[:time][i])", ylim=minmax_peak);
plot!(spec[:vel_nn][[1, end]], repeat([mean_peak],2))
plot!(spec[:vel_nn][[1, end]], repeat([std_peak],2))

# smoothing the noise base:
using ImageFiltering
ker = ImageFiltering.Kernel.gaussian((3,))

new_η = imfilter(η, ker)
new_η[η .> std_peak] = η[η .> std_peak]
p2 = plot(spec[:vel_nn],  new_η, laber="filter", ylim=minmax_peak);

plot(p1, p2, layout=(1,2))
# end of script
