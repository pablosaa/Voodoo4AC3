# function to adapt KAZR spectrum data for Voodoo's input
function adapt_KARZ_data4voodoo(η_kazr::Dict)
    using Dates

    # making time steps everey 30 seconds:
    TIME_STEP = 30
    cln_time = η_kazr[:time][1]:Seconds(TIME_STEP):η_kazr[:time][end]

    # spectra time indexes to match cln_time:
    thr_ts = minimum(diff(spec[:time]))
    idx_ts = map(x -> findfirst((abs.(x .- spec[:time])) .< thr_ts), cln_time);

    # selecting the 6 spectra around time step:
    ii_6spec = map(j -> range(j<3 ? 1 : j-2, length=6), idx_ts);

    # dimensions for output file:
    n_rg, n_ts = spec[:spact_mask][:, idx_ts];

    for i_ts ∈ (1:n_ts)
        for i_6p ∈ (1:6)
            in_ts = spec[:spect_mask][:, idx_ts]
            in_ts .+= 1
            idx_hgt = filter(x-> x≥ 0, spec[:spect_mask][:, ii_6spec[i_ts]])
            n_hgt = length(idx_hgt)
            # features dimension [n_samples, 1, 6, n_vel]
            features[:, 1, i_6p, :] = η_kazr[:, in_ts]
        end
    end
    return π
end
