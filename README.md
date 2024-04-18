# Voodoo4AC3
### Implementation of Voodoo for ARM radars
B07 project within (AC)3

#### Gemerating Zarr files from ARM radar data:
A script to run a Zarr file generator for multiple days is ```create_radar_features2zarr.jl```.
For example to generate for the 26 and 27th of October 2019, run from CLI as follow:
```
# to run the days indicated in the script:
> ./create_radar_features2zarr.jl

# to provide specific date:
> ./create_radar_features2zarr.jl year=(2019) month=(10) day=(26,27)
```
Important parameters to edit in ```create_radar_features2zarr.jl``` are:
```julia
# defining date and time to analyze:
years = (2019)
months = (1)
days = (26)
hours = ()

# defining Site information:
SITE = "utqiagvik-nsa"  # or "arctic-mosaic" 
PRODUCT = "KAZR/GE" # or alternatives "MWACR",  "KAZR/MD"

# defining Directory paths:
const BASE_PATH = "/projekt2/remsens/data_new/site-campaign" # base path for ARM data
const PATH_DATA = joinpath(BASE_PATH, SITE)  # where the ARM data is located. e.g. "LIM/remsens/utqiagvik-nsa/"
const PROD_TYPE = "$(PRODUCT)/SPECCOPOL"     # product to use, e.g. "KAZR/GE/SPECCOPOL" for General Mode spectrum
const B07_PATH  =  joinpath("/projekt2/ac3data/B07-data/", SITE, "CloudNet")  # path where Clodunet data is located
const CLNET_PATH = joinpath(BASE_PATH, "output")  # path where to put the ouputs.
```
#### Limits for spectral normalization variable:
```julia
spec_params = Dict(:Znn=>(-105, -10), :SNR=>(0, 80));
spec_var = :Znn  # or optional :SNR for signal-to-noise-ratio
```

#### Reading spectrum file and convert it to zarr:
```julia
> using CloudnetTools
> using Voodoo4AC3

# reading Cloudnet categorize file:
> clnet = CloudnetTools.readCLNFile("cloudnet_data/20200415_moasic_categorize.nc");

> spec = spec = ARMtools.readSPECCOPOL("moskazrcfrspcgecopolM1.a1.20200415.060005.nc");
> XdB = voodoo.adapt_RadarData(spec,
                                cln_time=clnet[:time],
                                var=spec_var,
                                LimHm=(10 ,10f3),
                                Δs=3,
                                Normalize=spec_params);
> voodoo.to_zarr(XdB, clnet, "input_zarr/moskazrcfrspcgecopol.voodoo_Znn.20200415.060005.zarr"; var=spec_var)
```



#### Running Zarr files with Voodoo
In MONSUM or BARAT servers, log in to remsens01 account.
Automatic script to run script:
```
> cd /home/remsens01/code/Voodoo
> conda activate py10
> ./NSA_voodoo_predict.sh
```
To run individual Torch model:
```
> python voodoo_test.py time="20190126" p=0.4 model='Vnet2_0-dy0-00-fn0-NSA-cuda0.pt'
```
where the script ```voodoo_test.py``` uses the file ```VnetSettings-1.toml``` to obtain information about data PATH and other parameters.
Most importatn is the ```[paths.NSA]``` field hourly_path which indicates the location of the ZARR hourly files used as input for Voodoo:
```
[paths.NSA]
    # north slope of alaska
    hourly_path = '/projekt2/ac3data/B07-data/utqiagvik-nsa/CloudNet/voodoo/'
    categorize_path = '/projekt2/ac3data/B07-data/utqiagvik-nsa/CloudNet/output/CEIL10m/'
    classification_path = '/projekt2/ac3data/B07-data/utqiagvik-nsa/CloudNet/output/CEIL10m/'
    ceilometer_path = '/projekt2/ac3data/B07-data/utqiagvik-nsa/CloudNet/input/CEIL10m/'

> ls /projekt2/ac3data/B07-data/utqiagvik-nsa/CloudNet/voodoo/2020/01/10
nsakazrcfrspcgecopolC1.a1.20200110.000008.zarr
nsakazrcfrspcgecopolC1.a1.20200110.080000.zarr
nsakazrcfrspcgecopolC1.a1.20200110.160003.zarr
...
```

There are 10 available Torch models trainded for KARZ, those can be selected in the file ```NSA_voodoo_predict.sh```. All Torch models are located at:
```
>  ls /home/remsens01/code/Voodoo/torch_models/Vnet2_0-dy0-00/*.pt
torch_models/Vnet2_0-dy0-00/Vnet2_0-dy0-00-fn0-NSA-cuda0.pt  torch_models/Vnet2_0-dy0-00/Vnet2_0-dy0-00-fn5-NSA-cuda1.pt
torch_models/Vnet2_0-dy0-00/Vnet2_0-dy0-00-fn1-NSA-cuda1.pt  torch_models/Vnet2_0-dy0-00/Vnet2_0-dy0-00-fn6-NSA-cuda2.pt
torch_models/Vnet2_0-dy0-00/Vnet2_0-dy0-00-fn2-NSA-cuda2.pt  torch_models/Vnet2_0-dy0-00/Vnet2_0-dy0-00-fn7-NSA-cuda3.pt
torch_models/Vnet2_0-dy0-00/Vnet2_0-dy0-00-fn3-NSA-cuda3.pt  torch_models/Vnet2_0-dy0-00/Vnet2_0-dy0-00-fn8-NSA-cuda0.pt
torch_models/Vnet2_0-dy0-00/Vnet2_0-dy0-00-fn4-NSA-cuda0.pt  torch_models/Vnet2_0-dy0-00/Vnet2_0-dy0-00-fn9-NSA-cuda1.pt
```
When using a specific Torch model, for instance ```Vnet2_0-dy0-00-fn9-NSA-cuda1.pt```, the script will generate the Voodoo outputs in the folder:
* ```torch_models/Vnet2_0-dy0-00/nc-fn9-NSA-cuda1```         : for the NetCDF cloudnet retrievals after modified by voodoo,
* ```torch_models/Vnet2_0-dy0-00/plots-fn9-NSA-cuda1```      : for plots

------

---<br>
(c) 2021, Pablo Saavedra Garfias<br>
pablo.saavedra@uni-leipzig.de <br>
University of Leipzig<br>
Faculty of Physics and Geosciences <br>
LIM <br>

See LICENSE
