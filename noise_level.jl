### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ d758115c-c216-4900-a9d1-393851b5af7a
let
	import Pkg
	Pkg.add(path="/home/pablo/LIM/repos/ARMtools.jl")
	import ARMtools
end

# ╔═╡ fb5569e0-e35c-44ba-82a0-b31a9d99103e
using Plots

# ╔═╡ b9851d02-cc5d-4a35-90ed-213eea344692
using PlutoUI

# ╔═╡ f329f96b-b09c-4568-b86e-be145e338bfc
using Statistics

# ╔═╡ 32adbc2c-0472-11ec-087f-331c556a12c5
md"""
# Spectra Noise Level
"""

# ╔═╡ 9fb8fbce-0637-42d9-87b8-ee00bfe93b2c
spec_file="/home/pablo/LIM/data/utqiagvik-nsa/KAZR/SPECCOPOL/2019/nsakazrspeccmaskgecopolC1.a0.20190126.100004.cdf";

# ╔═╡ 9a57a2d2-3c2e-421f-99a2-fc474cb21e12
spec=ARMtools.readSPECCOPOL(spec_file);

# ╔═╡ 05870c06-167e-4322-8b8c-5cde7beea03d
md"""
### Spectrum Index to show:
"""

# ╔═╡ 36a55803-29c3-4b86-8d78-78d67b2ee168
begin 
	i = 291
	idx_alt = findall(spec[:spect_mask][:,i] .≥ 0);
	idx= 1 .+ spec[:spect_mask][idx_alt, i];
end

# ╔═╡ d73a1172-d3da-456f-821b-56159e62a9b6
Plots.plot(spec[:vel_nn], spec[:height][idx_alt], spec[:η_hh][:,idx]',st=:heatmap, color=:berlin, title="$(spec[:time][i])")

# ╔═╡ 4650239c-6898-4805-951f-755afe1a11a4
@bind i_hgt Slider(80:30:4000; default=160,show_value=true) # m

# ╔═╡ 0b53165a-0c71-4db5-b81d-e15c265ae391
begin
	# for specific Altitude and time span:	
	tmp = findfirst(isapprox.(spec[:height][idx_alt], i_hgt, atol=9.9));
	# index corresponding to height i_hgt [m]
	idx_hgt = 1 .+ spec[:spect_mask][idx_alt[tmp], i];
end

# ╔═╡ 57bec460-08f2-467c-b043-8f60d90e119a
begin
	η = spec[:η_hh][:, idx_hgt]
	max_peak = maximum(η)
	mean_peak = mean(η)
	std_peak = std(η) + mean_peak
	plot(spec[:vel_nn],  η, title="$i_hgt [m] at $(spec[:time][i])");
	plot!(spec[:vel_nn][[1, end]], repeat([mean_peak],2))
	plot!(spec[:vel_nn][[1, end]], repeat([std_peak],2))
end

# ╔═╡ Cell order:
# ╟─32adbc2c-0472-11ec-087f-331c556a12c5
# ╠═d758115c-c216-4900-a9d1-393851b5af7a
# ╠═9fb8fbce-0637-42d9-87b8-ee00bfe93b2c
# ╠═9a57a2d2-3c2e-421f-99a2-fc474cb21e12
# ╟─05870c06-167e-4322-8b8c-5cde7beea03d
# ╠═36a55803-29c3-4b86-8d78-78d67b2ee168
# ╠═fb5569e0-e35c-44ba-82a0-b31a9d99103e
# ╠═d73a1172-d3da-456f-821b-56159e62a9b6
# ╠═b9851d02-cc5d-4a35-90ed-213eea344692
# ╠═4650239c-6898-4805-951f-755afe1a11a4
# ╠═0b53165a-0c71-4db5-b81d-e15c265ae391
# ╠═f329f96b-b09c-4568-b86e-be145e338bfc
# ╠═57bec460-08f2-467c-b043-8f60d90e119a
