# File containing module for using Voodoo with ARM radars:
#

module voodoo

using PyCall
using TOML

# *******************************************************************
function MakePrediction(X_in::Array{<:AbstractFloat, 4}; sizeout::NTuple{2, Int} = (0,0))
    
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

    predict_var = try
        # Final output array with predictions (range x time):
        reshape(X_out, (sizeout..., NCLASSES));
    catch
        X_out
    end

    return predict_var
end

# ***********************************************************************
# Function to normalize the spectrum
"""
Function to normalize spectrum data.
   > η = η₀₁(Znn)
with default normalization min and max limits -100 and -55, respectively.
or
   > η = η₀₁(Znn, η0 = -90, η1 = -50)
for other min/max limit values e.g. -90 and -50.
All values outside the range min to max are set-up to 0 and 1. 

"""
function η₀₁(η::Array{<:AbstractFloat, 4}; ηlim::NTuple{2, Int} = (-100, -50))
    η0 = ηlim[1]
    η1 = ηlim[2]
    H_out = @. (η - η0)/(η1 - η0)
    @. H_out = min(1, H_out) # H_out[H_out > 1] = 1f0
    @. H_out = max(0, H_out) #[H_out < 0] = 0f0
    # removing NaNs:
    @. H_out[isnan(H_out)] = 0
    return H_out
end
# end of function

# Alternative function to normalize based on (min, max) of given dimension:
# Input Arrray must have (n_samples, 1, 6, n_vel)
function Norm_dim(X::Array{<:AbstractFloat, 4}; dims=1)
    X0, X1 = extrema(X, dims=4)
    return η₀₁(X, η0=X0, η1=X1)
end

end  # end of module
# ----!
# end of script
