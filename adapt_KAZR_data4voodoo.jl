# function to adapt KAZR spectrum data for Voodoo's input
function adapt_KARZ_data4voodoo(spec::Dict)

    # making time steps everey 30 seconds:
    TIME_STEP = 30
    cln_time = round(spec[:time][1], Second(TIME_STEP)):Second(TIME_STEP):round(spec[:time][end], Second(TIME_STEP))

    # spectra time indexes to match cln_time:
    thr_ts = minimum(diff(spec[:time]))
    idx_ts = map(x -> findfirst((abs.(x .- spec[:time])) .< thr_ts), cln_time);

    # selecting the 6 spectra around time step:
    ii_6spec = map(idx_ts) do j
        let n_ts = length(spec[:time])
            x_ini = ifelse( 3 ≤ j ≤ n_ts-5, j-2, min(j, n_ts-5))
            range(x_ini, length=6) |> collect
        end
    end
    ii_6spec = hcat(ii_6spec...);  # (6 x n_ts)

    # dimensions for output file:
    n_samples = map(1:6) do i_6p
        local n = spec[:spect_mask][:, ii_6spec[i_6p,:] ]
        length(n[n .≥ 0])
    end |> maximum
    
    n_vel = length(spec[:vel_nn])
    
    n_ts = length(spec[:time])

    # defining output array: features
    # according to mattermost slide: (n_samples, n_vel, n_ts=6, n_ch=1)
    # according to voodoo predictor: (n_samples, n_ch=1, n_ts=6, n_vel)
    features = fill(NaN32, (n_vel, n_samples, 1, 6))
    
    for i_6p ∈ (1:6)
        garbage = map(spec[:spect_mask][:, ii_6spec[i_6p, :] ]) do x
            x .≥ 0 ? x+1 : missing
        end;

        let var = skipmissing(garbage) |> collect
            local n_subsample = length(var)
            features[:, 1:n_subsample, 1, i_6p] = spec[:η_hh][:, var]
        end
    end

    return PermutedDimsArray(features, (2, 3, 4, 1))
end

# end of function
# ----/

