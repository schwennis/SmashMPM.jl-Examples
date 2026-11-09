using HDF5
using Printf
using StaticArrays
using LinearAlgebra
using CairoMakie
using Glob

# Conservation of linear momentum, angular momentum and energy per body, including the APIC contribution.
# With quadratic B-splines D_p = Σ_i w_ip (x_i - x_p)(x_i - x_p)ᵀ = dx²/4 · I, so that for the grid
# velocities v_i = v_p + C_p (x_i - x_p):
#   P      = Σ m v
#   L      = Σ m x × v  +  Σ m D (C₃₂ - C₂₃, C₁₃ - C₃₁, C₂₁ - C₁₂)
#   E_kin  = ½ Σ m |v|²  +  ½ Σ m D ‖C‖²_F
const DX = parse(Float64, get(ARGS, 2, "1.6e-3"))   # grid spacing, only used if the file has no grid/dx

struct Moments
    mass::Float64
    P::SVector{3, Float64}
    L_orbital::SVector{3, Float64}
    L_apic::SVector{3, Float64}
    E_kin::Float64
    E_apic::Float64
    E_int::Float64
end

L_total(m::Moments) = m.L_orbital + m.L_apic
E_total(m::Moments) = m.E_kin + m.E_apic + m.E_int

axial(C) = SVector(C[3, 2] - C[2, 3], C[1, 3] - C[3, 1], C[2, 1] - C[1, 2])

function moments(idx, m, x, v, C, e, D)
    return Moments(
        sum(m[i] for i in idx),
        sum(m[i] * v[i] for i in idx),
        sum(m[i] * cross(x[i], v[i]) for i in idx),
        sum(m[i] * D * axial(C[i]) for i in idx),
        0.5 * sum(m[i] * dot(v[i], v[i]) for i in idx),
        0.5 * sum(m[i] * D * sum(abs2, C[i]) for i in idx),
        sum(m[i] * e[i] for i in idx),
    )
end

function read_moments(file_path::String)
    h5open(file_path, "r") do file
        t  = Float64(attrs(file)["time"])
        id = read(file["particles/id"])
        m  = Float64.(read(file["particles/mass"]))
        e  = Float64.(read(file["particles/mat_state/e"]))
        x  = SVector{3, Float64}.(eachcol(read(file["particles/position"])))
        v  = SVector{3, Float64}.(eachcol(read(file["particles/velocity"])))
        C  = [SMatrix{3, 3, Float64, 9}(Float64.(col))' for col in eachcol(read(file["particles/affine"]))]
        dx = haskey(file, "grid/dx") ? Float64(read(file["grid/dx"])) : DX
        D  = dx^2 / 4

        bodies = Dict{Int, Moments}()
        for b in sort!(unique(id))
            bodies[b] = moments(findall(==(b), id), m, x, v, C, e, D)
        end
        bodies[0] = moments(eachindex(id), m, x, v, C, e, D)
        return t, bodies
    end
end


function write_csv(file_path::String, times, history)
    open(file_path, "w") do io
        println(io, "t,body,mass,Px,Py,Pz,Lx,Ly,Lz,Lz_apic,E_kin,E_apic,E_int")
        for (t, bodies) in zip(times, history), b in sort!(collect(keys(bodies)))
            mo = bodies[b]
            L  = L_total(mo)
            @printf(io, "%.9e,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e\n",
                    t, b, mo.mass, mo.P..., L..., mo.L_apic[3], mo.E_kin, mo.E_apic, mo.E_int)
        end
    end
end

function plot_conservation(file_path::String, times, history)
    t_µs  = 1e6 .* times
    ids   = filter(!=(0), sort!(collect(keys(first(history)))))
    P0    = norm(first(history)[0].P)
    L0    = L_total(first(history)[0])
    E0    = E_total(first(history)[0])
    names = Dict(0 => "total", [b => "body $b" for b in ids]...)

    fig = Figure(size = (1000, 800))
    for (k, comp) in enumerate(("x", "y"))
        ax = Axis(fig[1, k], xlabel = "t (µs)", ylabel = "P_$comp / |P(0)|")
        for b in vcat(0, ids)
            lines!(ax, t_µs, [h[b].P[k] / P0 for h in history], label = names[b])
        end
        k == 1 && axislegend(ax, position = :lb)
    end

    ax = Axis(fig[2, 1], xlabel = "t (µs)", ylabel = "|L(t) - L(0)| / |L(0)|", yscale = log10)
    lines!(ax, t_µs, [max(norm(L_total(h[0]) - L0) / norm(L0), 1e-16) for h in history], label = "total")
    lines!(ax, t_µs, [max(norm(h[0].L_orbital - first(history)[0].L_orbital) / norm(L0), 1e-16) for h in history],
           label = "without APIC part")
    axislegend(ax, position = :lt)

    ax = Axis(fig[2, 2], xlabel = "t (µs)", ylabel = "E / E_tot(0)")
    lines!(ax, t_µs, [E_total(h[0]) / E0 for h in history], label = "E_tot")
    lines!(ax, t_µs, [(h[0].E_kin + h[0].E_apic) / E0 for h in history], label = "E_kin")
    lines!(ax, t_µs, [h[0].E_int / E0 for h in history], label = "E_int")
    axislegend(ax, position = :rc)

    save(file_path, fig)
end


function main()
    output_dir = isempty(ARGS) ? "output" : ARGS[1]
    files = glob("*.h5", output_dir)
    sort!(files, by = f -> parse(Int, match(r"(\d+)\.h5$", basename(f))[1]))

    times   = Float64[]
    history = Dict{Int, Moments}[]
    for (i, file_path) in enumerate(files)
        print("Processing $i / $(length(files)): $(basename(file_path))\r")
        t, bodies = read_moments(file_path)
        push!(times, t)
        push!(history, bodies)
    end
    println()

    first_total, last_total = first(history)[0], last(history)[0]
    L0, E0 = L_total(first_total), E_total(first_total)

    @printf("%9s %12s %12s %12s %12s %12s\n", "t (µs)", "Px", "Py", "|ΔP|/|P0|", "|ΔL|/|L0|", "ΔE/E0")
    for i in unique(round.(Int, range(1, length(times), length = 12)))
        tot = history[i][0]
        @printf("%9.2f %12.4e %12.4e %12.3e %12.3e %12.3e\n", 1e6 * times[i], tot.P[1], tot.P[2],
                norm(tot.P - first_total.P) / norm(first_total.P),
                norm(L_total(tot) - L0) / norm(L0),
                E_total(tot) / E0 - 1)
    end

    println("\nPer body at t = $(round(1e6 * times[end], digits = 2)) µs (0 = total, relative to |P(0)| = $(round(norm(first_total.P), sigdigits = 4)) kg m/s)")
    @printf("%5s %12s %12s %12s %12s %12s %12s %12s\n", "body", "mass (kg)", "Px / |P0|", "Py / |P0|", "E_kin (J)", "E_apic (J)", "E_int (J)", "E_tot / E0")
    for b in sort!(collect(keys(history[end])))
        mo = history[end][b]
        @printf("%5d %12.4e %12.4f %12.4f %12.4e %12.4e %12.4e %12.4f\n", b, mo.mass, mo.P[1] / norm(first_total.P),
                mo.P[2] / norm(first_total.P), mo.E_kin, mo.E_apic, mo.E_int, E_total(mo) / E0)
    end

    analysis_dir = joinpath(output_dir, "analysis")
    mkpath(analysis_dir)
    write_csv(joinpath(analysis_dir, "conservation.csv"), times, history)
    plot_conservation(joinpath(analysis_dir, "conservation.png"), times, history)
end

main()