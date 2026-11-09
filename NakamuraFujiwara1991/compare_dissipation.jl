using DelimitedFiles
using Printf

# Compares the energy removed by the transfers (transfer_dissipation.csv from solve!)
# with the total energy loss from conservation.csv (conservation_check.jl).
function main()
    output_dir = isempty(ARGS) ? "output" : ARGS[1]

    diss, _ = readdlm(joinpath(output_dir, "transfer_dissipation.csv"), ',', header = true)
    cons, _ = readdlm(joinpath(output_dir, "analysis", "conservation.csv"), ',', header = true)

    total = cons[cons[:, 2] .== 0, :]
    t_c   = total[:, 1]
    E_c   = total[:, 11] .+ total[:, 12] .+ total[:, 13]
    E0    = E_c[1]

    @printf("%9s %12s %12s %12s %10s\n", "t (µs)", "E lost (J)", "transfer (J)", "rate (W)", "transfer / lost")
    for k in axes(diss, 1)
        t = diss[k, 1]
        i = argmin(abs.(t_c .- t))
        lost = E0 - E_c[i]
        @printf("%9.2f %12.2f %12.2f %12.3e %10.3f\n", 1e6 * t, lost, diss[k, 4], diss[k, 3], diss[k, 4] / lost)
    end
end

main()
