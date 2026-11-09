using HDF5
using Printf
using StaticArrays
using LinearAlgebra
using Statistics
using CairoMakie
using Glob

# Fragment identification as in Yan et al., Icarus 455 (2026) 117080, Section 4.1.2:
#   1. points with D ≥ D_FAIL are failed
#   2. unfailed points with ≥ MIN_NEIGHBORS unfailed neighbors are extendable (DBSCAN core points)
#   3. friends-of-friends over extendable points, then attach non-extendable points and the failed layer
#   4. everything left over is dust below the resolution
# Kinematics assume rigid body motion (Eq. 20).
const D_FAIL        = 0.9
const LINK_FACTOR   = 1.5     # linking length in units of the initial particle spacing
const MIN_NEIGHBORS = 9
const TARGET_ID     = 1       # particle set of the target

struct ParticleData
    t::Float64
    pos::Vector{SVector{3, Float64}}
    vel::Vector{SVector{3, Float64}}
    mass::Vector{Float64}
    vol::Vector{Float64}
    damage::Vector{Float64}
end

struct Fragment
    mass::Float64
    x_c::SVector{3, Float64}
    v_c::SVector{3, Float64}
    ω::SVector{3, Float64}
    n_points::Int
end


function read_particles(file_path::String, body_id::Integer)
    h5open(file_path, "r") do file
        sel = findall(==(body_id), read(file["particles/id"]))
        vec3(name) = SVector{3, Float64}.(eachcol(read(file["particles/" * name])))[sel]
        ParticleData(
            Float64(attrs(file)["time"]),
            vec3("position"),
            vec3("velocity"),
            Float64.(read(file["particles/mass"]))[sel],
            Float64.(read(file["particles/volume"]))[sel],
            Float64.(read(file["particles/mat_state/damage"]))[sel],
        )
    end
end


cell_key(x, h) = (floor(Int, x[1] / h), floor(Int, x[2] / h), floor(Int, x[3] / h))

function neighbor_lists(pos::Vector{SVector{3, Float64}}, h)
    cells = Dict{NTuple{3, Int}, Vector{Int}}()
    for (i, x) in enumerate(pos)
        push!(get!(Vector{Int}, cells, cell_key(x, h)), i)
    end

    empty = Int[]
    return map(eachindex(pos)) do i
        ci = cell_key(pos[i], h)
        nb = Int[]
        for dz in -1:1, dy in -1:1, dx in -1:1
            for j in get(cells, ci .+ (dx, dy, dz), empty)
                j != i && sum(abs2, pos[j] - pos[i]) <= h^2 && push!(nb, j)
            end
        end
        nb
    end
end


function attach!(frag, source, nbrs, eligible)
    for i in eachindex(frag)
        (frag[i] == 0 && eligible(i)) || continue
        k = findfirst(j -> source[j] != 0, nbrs[i])
        isnothing(k) || (frag[i] = source[nbrs[i][k]])
    end
    return frag
end

function identify_fragments(pd::ParticleData; d_fail = D_FAIL, link_factor = LINK_FACTOR,
                            min_neighbors = MIN_NEIGHBORS)
    n       = length(pd.mass)
    spacing = cbrt(median(pd.vol))
    nbrs    = neighbor_lists(pd.pos, link_factor * spacing)

    intact     = pd.damage .< d_fail
    extendable = [intact[i] && count(j -> intact[j], nbrs[i]) >= min_neighbors for i in 1:n]

    frag   = zeros(Int, n)
    n_frag = 0
    stack  = Int[]
    for seed in 1:n
        (extendable[seed] && frag[seed] == 0) || continue
        n_frag += 1
        frag[seed] = n_frag
        push!(stack, seed)
        while !isempty(stack)
            i = pop!(stack)
            for j in nbrs[i]
                (extendable[j] && frag[j] == 0) || continue
                frag[j] = n_frag
                push!(stack, j)
            end
        end
    end

    attach!(frag, copy(frag), nbrs, i -> intact[i])
    attach!(frag, copy(frag), nbrs, i -> !intact[i])
    return frag, n_frag
end


function fragment_kinematics(pd::ParticleData, frag::Vector{Int}, n_frag::Int)
    members = [Int[] for _ in 1:n_frag]
    for (i, f) in enumerate(frag)
        f > 0 && push!(members[f], i)
    end

    I3 = one(SMatrix{3, 3, Float64, 9})
    fragments = map(members) do idx
        m, x, v = pd.mass[idx], pd.pos[idx], pd.vel[idx]
        m_f = sum(m)
        x_c = sum(m .* x) / m_f
        v_c = sum(m .* v) / m_f

        L_c = sum(m[i] * cross(x[i] - x_c, v[i] - v_c) for i in eachindex(idx))
        J_c = sum(eachindex(idx)) do i
            r = x[i] - x_c
            m[i] * ((dot(r, r) + cbrt(pd.vol[idx[i]])^2 / 6) * I3 - r * r')
        end
        Fragment(m_f, x_c, v_c, J_c \ L_c, length(idx))
    end
    return sort!(fragments, by = f -> f.mass, rev = true)
end


function write_fragments(file_path::String, fragments::Vector{Fragment}, M_tar)
    open(file_path, "w") do io
        println(io, "rank,mass,mass_ratio,n_points,x,y,z,vx,vy,vz,v_abs,wx,wy,wz")
        for (k, f) in enumerate(fragments)
            @printf(io, "%d,%.6e,%.6e,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                    k, f.mass, f.mass / M_tar, f.n_points, f.x_c..., f.v_c..., norm(f.v_c), f.ω...)
        end
    end
end

function plot_mass_distribution(file_path::String, fragments::Vector{Fragment}, M_tar)
    ratios = [f.mass / M_tar for f in fragments]
    fig = Figure(size = (600, 450))
    ax  = Axis(fig[1, 1], xscale = log10, yscale = log10,
               xlabel = "m / M_tar", ylabel = "N(> m)", title = "Cumulative fragment mass distribution")
    stairs!(ax, ratios, 1:length(ratios), step = :post)
    save(file_path, fig)
end


function main()
    file_path = isempty(ARGS) ? last(sort(glob("output/*.h5"))) : ARGS[1]
    pd = read_particles(file_path, TARGET_ID)
    M_tar = sum(pd.mass)

    frag, n_frag = identify_fragments(pd)
    fragments    = fragment_kinematics(pd, frag, n_frag)
    m_dust       = M_tar - sum(f.mass for f in fragments; init = 0.0)

    @printf("t = %.2f µs, %d fragments, dust = %.1f %%\n", 1e6 * pd.t, n_frag, 100 * m_dust / M_tar)
    @printf("%5s %10s %8s %10s %10s\n", "rank", "m/M_tar", "points", "|v| (m/s)", "|ω| (1/s)")
    for (k, f) in enumerate(first(fragments, 10))
        @printf("%5d %10.4f %8d %10.3f %10.3f\n", k, f.mass / M_tar, f.n_points, norm(f.v_c), norm(f.ω))
    end

    mkpath("analysis")
    write_fragments("analysis/fragments.csv", fragments, M_tar)
    plot_mass_distribution("analysis/mass_distribution.png", fragments, M_tar)
end

main()