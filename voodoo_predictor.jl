### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 99feaf54-05ed-11ec-0472-618993c00b94
using PyCall

# ╔═╡ d19215a4-20d0-40ad-a03a-c77ed3f91da9
torch=pyimport("torch")

# ╔═╡ 04caeb3e-b89f-4657-87e8-17001fac32c3
pushfirst!(PyVector(pyimport("sys")."path"), joinpath(@__DIR__, "Voodoo") )
	

# ╔═╡ a387a98f-b97e-4b62-a1d7-2289876df282
TM = pyimport("libVoodoo.TorchModel")

# ╔═╡ 48195c22-6cf5-4e14-9270-6b361312353d
X = rand(100,256,6);

# ╔═╡ 7ce47788-8069-43fe-89f8-901bda0bf1c4
begin
	
py"""
import toml
import os
import numpy as np
def VoodooPredictor(X):
    model_setup_file = f'Voodoo/VnetSettings-1.toml'
    torch_settings = toml.load(os.path.join(model_setup_file))['pytorch']
    torch_settings.update({'dev': 'cpu'})
    trained_model = 'Voodoo/Vnet0x60de1687-fnX-gpu0-VN.pt'
    p = 0.5

    print(f'Loading Vnet model ...... {model_setup_file}')

    # (n_samples, n_Doppler_bins, n_time_steps)
    X = X[:, :, :, np.newaxis]
    X = X.transpose(0, 3, 2, 1)
    X_test = torch.Tensor(X)

    model = TM.VoodooNet(X_test.shape, NCLASSES, **torch_settings)
    model.load_state_dict(torch.load(trained_model, map_location=model.device)['state_dict'])

    prediction = model.predict(X_test, batch_size=256)
    prediction = prediction.to('cpu')
    return prediction
"""
py"VoodooPredictor(np.random.rand(100,256,6))"
end

# ╔═╡ 96edf425-2db2-4e9c-9401-a203bda23e7e
toml = pyimport("toml")

# ╔═╡ baf62e65-86f3-4c25-a7a6-8f6b0cdc24cf
begin
	model_setup_file="Voodoo/VnetSettings-1.toml"
	torch_settings = toml.load(model_setup_file)["pytorch"]
end

# ╔═╡ 6feb15a9-dabd-4f44-97ee-949a4e15f74c
torch_settings["dev"]="cpu"

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"

[compat]
PyCall = "~1.92.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Conda]]
deps = ["JSON", "VersionParsing"]
git-tree-sha1 = "299304989a5e6473d985212c28928899c74e9421"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.5.2"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "0fb723cd8c45858c22169b2e42269e53271a6df7"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.7"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "438d35d2d95ae2c5e8780b330592b6de8494e779"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.0.3"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "169bb8ea6b1b143c5cf57df6d34d022a7b60c6db"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.92.3"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[VersionParsing]]
git-tree-sha1 = "80229be1f670524750d905f8fc8148e5a8c4537f"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.2.0"
"""

# ╔═╡ Cell order:
# ╠═99feaf54-05ed-11ec-0472-618993c00b94
# ╠═d19215a4-20d0-40ad-a03a-c77ed3f91da9
# ╠═04caeb3e-b89f-4657-87e8-17001fac32c3
# ╠═a387a98f-b97e-4b62-a1d7-2289876df282
# ╠═48195c22-6cf5-4e14-9270-6b361312353d
# ╠═7ce47788-8069-43fe-89f8-901bda0bf1c4
# ╠═96edf425-2db2-4e9c-9401-a203bda23e7e
# ╠═baf62e65-86f3-4c25-a7a6-8f6b0cdc24cf
# ╠═6feb15a9-dabd-4f44-97ee-949a4e15f74c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
