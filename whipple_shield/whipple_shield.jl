using SmashMPM
using StaticArrays
using KernelAbstractions: CPU, Backend


backend_to_use = :cuda
# const backend_to_use = :cpu

T = Float64 # Backend independent precision, has to be Float64 to conserve energy and angular momentum


if backend_to_use === :cuda
    println("Using CUDA backend...")
    using CUDA  # Import CUDA only if using CUDABackend
    CUDA.set_runtime_version!(local_toolkit=true)   # true if local CUDA toolkit is installed, false if using system CUDA
    CUDA.allowscalar(false)
    BACKEND = CUDABackend()
else
    println("Using CPU backend using $(Threads.nthreads()) threads...")
    BACKEND = CPU()
end


# ---------------------------------------------------------------------------- #
#                               Impact Parameters                              #
# ---------------------------------------------------------------------------- #
viscosity = BulkViscosity()
plasticity = J2Plasticity(σ_y0=2.75e8, H=6e8)
const PROJECTILE_MATERIAL = Aluminum(T, eos=:tillotson, elasticity=:hypoelastic, damage=:grady_kipp, viscosity=viscosity, plasticity=plasticity)
const BUMPER_MATERIAL = Aluminum(T, eos=:tillotson, elasticity=:hypoelastic, damage=:grady_kipp, viscosity=viscosity, plasticity=plasticity)
const REAR_WALL_MATERIAL = Aluminum(T, eos=:tillotson, elasticity=:hypoelastic, damage=:grady_kipp, viscosity=viscosity, plasticity=plasticity)

const IMPACT_VELOCITY = T(8e3) # m/s
const IMPACT_ANGLE = T(0) # rad
const PROJECTILE_RADIUS = T(2.5e-3) # m
const INITIAL_SEPARATION = T(10e-3) # m, distance between projectile and bumper

const PLATE_SIZE = T(50e-3) # m

const NR_BUMPERS = 1
const BUMPER_DISTANCE = T(10e-3) # m
const BUMPER_THICKNESS = T(1e-3)  # m

const REAR_WALL_SEPARATION = T(20e-3) # m
const REAR_WALL_THICKNESS = T(2e-3) # m, set to zero to remove rear wall

const CLAMPED_CELLS = 2 # Number of cells the plates are clamped to the boundary.
const BUFFER_CELLS_X = 6 # Number of cells to buffer the domain in the x-direction (to avoid projectile leaving the domain) 

# ---------------------------------------------------------------------------- #
#                             Simulation Parameters                            #
# ---------------------------------------------------------------------------- #
const DX = T(0.25e-3) # m, Bumper thickness should be at least a few DX
const PPC_1D = 2
const PADDING = CLAMPED_CELLS + 2
const CFL_NUMBER = T(0.4)
const DT_MAX = T(1e-7)
const T_MAX = 1.5 * (INITIAL_SEPARATION + NR_BUMPERS * BUMPER_DISTANCE + REAR_WALL_SEPARATION + REAR_WALL_THICKNESS) / IMPACT_VELOCITY # s, time to impact + time to travel through bumpers and rear wall

const N_EXPORTS = 60
const EXPORT_GRID = false


# ---------------------------------------------------------------------------- #
#                                    Checks                                    #
# ---------------------------------------------------------------------------- #
@assert BUMPER_THICKNESS >= 2 * DX "BUMPER_THICKNESS must be at least 2 * DX to avoid numerical issues."
@assert BUMPER_DISTANCE >= 2 * DX "BUMPER_DISTANCE must be at least 2 * DX to avoid numerical issues."
@assert REAR_WALL_THICKNESS >= 2 * DX "REAR_WALL_THICKNESS must be at least 2 * DX to avoid numerical issues."
@assert REAR_WALL_SEPARATION >= 2 * DX "REAR_WALL_SEPARATION must be at least 2 * DX to avoid numerical issues."
@assert INITIAL_SEPARATION >= 2 * DX "INITIAL_SEPARATION must be at least 2 * DX to avoid numerical issues."



# ---------------------------------------------------------------------------- #
#                                    Bodies                                    #
# ---------------------------------------------------------------------------- #
println("Creating bodies...")


# Projectile
impact_dir = @SVector [cos(IMPACT_ANGLE), sin(IMPACT_ANGLE), 0.0]
projectile_center = - impact_dir * (INITIAL_SEPARATION + PROJECTILE_RADIUS)
projectile_velocity = impact_dir * IMPACT_VELOCITY

projectile_shape = Sphere{T}(PROJECTILE_RADIUS, projectile_center)
projectile_body = Body(projectile_shape, projectile_velocity, zero(SVector{3, T}), PROJECTILE_MATERIAL)
bodies = (projectile_body,)

# Bumpers
function generate_plate(center_x, thickness, material)
    plate_shape = RectangularPrism(thickness, PLATE_SIZE, PLATE_SIZE, SVector(center_x, 0.0, 0.0), zero(SVector{3, T}))
    plate_velocity = zero(SVector{3, T})
    return Body(plate_shape, plate_velocity, zero(SVector{3, T}), material)
end
for i in 0:NR_BUMPERS-1
    global bodies
    bumper_center_x = i * BUMPER_DISTANCE + BUMPER_THICKNESS / 2
    bumper_body = generate_plate(bumper_center_x, BUMPER_THICKNESS, BUMPER_MATERIAL)
    bodies = (bodies..., bumper_body)
end


# Rear wall
if REAR_WALL_THICKNESS > 0
    rear_wall_center_x = NR_BUMPERS * BUMPER_DISTANCE + REAR_WALL_SEPARATION + REAR_WALL_THICKNESS / 2
    rear_wall_body = generate_plate(rear_wall_center_x, REAR_WALL_THICKNESS, REAR_WALL_MATERIAL)
    bodies = (bodies..., rear_wall_body)
end



# ---------------------------------------------------------------------------- #
#                                   Clamping                                   #
# ---------------------------------------------------------------------------- #
const BUFFER_WIDTH = SVector{3, Int}(BUFFER_CELLS_X, -CLAMPED_CELLS, -CLAMPED_CELLS)



# ---------------------------------------------------------------------------- #
#                                   Exporter                                   #
# ---------------------------------------------------------------------------- #
exporter = HDF5Exporter(
    output_dir=joinpath(@__DIR__, "output_gradykipp"),
    filename_prefix="whipple_shield_gradykipp_",
    write_xdmf=true,
    compression_level=5,
    export_grid=EXPORT_GRID
    )
    
    
println("Setting up simulation...")
sim_setup = SimulationSetup(
    dx                      = DX,
    grid_type               = SparseGrid,
    padding                 = PADDING,  
    domain                  = FittedBox(),
    cfl_number              = CFL_NUMBER,
    t_max                   = T_MAX,
    dt_max                  = DT_MAX,
    ppc_1d                  = PPC_1D,
    exporter                = exporter,
    export_time_interval    = T(T_MAX / N_EXPORTS),
    boundary_condition      = NoSlipBoundary(),
    buffer_width            = BUFFER_WIDTH,
    backend                 = BACKEND,
)

model = build_mpm_model(bodies, sim_setup)

N_total = sum(length(ps.particles) for ps in model.particle_sets)
if N_total > 1e6
    @warn "Total number of particles is $N_total, which may lead to long simulation times."
elseif N_total < 1e5
    @warn "Total number of particles is $N_total, which may lead to poor resolution."
else
    println("Total number of particles is $N_total, which is within the recommended range.")
end

println("Simulation set up with $(length(bodies)) bodies.")
println("Total number of particles: $N_total\nGrid dimensions: $(size(model.grid.state_read))")
println("Simulation time: $T_MAX s, Time step limit: $DT_MAX s, CFL number: $CFL_NUMBER")



# ---------------------------------------------------------------------------- #
#                                      Run                                     #
# ---------------------------------------------------------------------------- #
solve!(model, sim_setup)


println("Simulation complete. Exported data to $(exporter.output_dir).")