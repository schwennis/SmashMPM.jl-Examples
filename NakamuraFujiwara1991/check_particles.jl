using HDF5
using Printf
using StaticArrays
using LinearAlgebra
using Glob

const V_LIMIT = 10e3     # m/s, well above impact velocity and shock particle velocities

function check_file(file_path::String)
    h5open(file_path, "r") do file
        t   = attrs(file)["time"]
        id  = read(file["particles/id"])
        vel = SVector{3, Float64}.(eachcol(read(file["particles/velocity"])))
        J   = [det(SMatrix{3, 3, Float64, 9}(col)) for col in eachcol(read(file["particles/F"]))]
        p   = Float64.(read(file["particles/mat_state/eos_state/p"]))
        e   = Float64.(read(file["particles/mat_state/e"]))
        c   = Float64.(read(file["particles/mat_state/c"]))
        D   = Float64.(read(file["particles/mat_state/damage"]))

        pos   = SVector{3, Float64}.(eachcol(read(file["particles/position"])))
        v_abs = norm.(vel)
        bad   = findall(i -> !all(isfinite, pos[i]) || !(v_abs[i] < V_LIMIT) || !(J[i] > 0) ||
                             !isfinite(e[i]) || isinf(p[i]), eachindex(id))

        @printf("t = %7.3f µs  bad = %d\n", 1e6 * t, length(bad))
        for b in sort!(unique(id))
            sel = findall(==(b), id)
            @printf("    body %5d  max|v| = %8.1f  J = [%.3f, %7.3f]  p = [%10.3e, %10.3e]  min e = %10.3e  max c = %7.1f\n",
                    b, maximum(v_abs[sel]), minimum(J[sel]), maximum(J[sel]),
                    minimum(p[sel]), maximum(p[sel]), minimum(e[sel]), maximum(c[sel]))
        i_J = sel[argmax(J[sel])]
        i_p = sel[argmin(p[sel])]
        @printf("        max J: J = %7.3f  D = %.4f  p = %10.3e  (1-D)p = %10.3e\n", J[i_J], D[i_J], p[i_J], (1 - D[i_J]) * min(p[i_J], 0))
        @printf("        min p: J = %7.3f  D = %.4f  p = %10.3e  (1-D)p = %10.3e\n", J[i_p], D[i_p], p[i_p], (1 - D[i_p]) * min(p[i_p], 0))
        end
        for i in first(bad, 10)
            @printf("    bad: body %5d  x = %s  |v| = %9.1f  J = %9.3e  p = %10.3e  e = %10.3e  c = %8.1f  D = %.3f\n",
                    id[i], string(round.(pos[i], sigdigits=3)), v_abs[i], J[i], p[i], e[i], c[i], D[i])
        end
        return length(bad)
    end
end

function main()
    files = isempty(ARGS) ? glob("output/*.h5") : ARGS
    sort!(files, by = f -> parse(Int, match(r"(\d+)\.h5$", basename(f))[1]))
    for f in files
        check_file(f) > 0 && break
    end
end

main()