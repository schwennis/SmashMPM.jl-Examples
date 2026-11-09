# Basalt experiment after Nakamura and Fujiwara (1991) using SmashMPM.jl


using SmashMPM
using StaticArrays
using KernelAbstractions: CPU, Backend


backend_to_use = :cuda
# const backend_to_use = :cpu

T = Float64 # Backend independent precision, has to be Float64 to conserve energy and angular momentum


if backend_to_use === :cuda
    println("Using CUDA backend...")
    using CUDA  # Import CUDA only if using CUDABackend
    CUDA.allowscalar(false)
    BACKEND = CUDABackend()
else
    println("Using CPU backend using $(Threads.nthreads()) threads...")
    BACKEND = CPU()
end





# ---------------------------------------------------------------------------- #
#                               Impact Parameters                              #
# ---------------------------------------------------------------------------- #
const IMPACT_VELOCITY = 3.2e3   # m/s
const IMPACT_ANGLE = 30.0       # degrees, between velocity and surface normal (0 = head-on)
const PROJECTILE_RADIUS = 3.5e-3
const TARGET_RADIUS = 30e-3


# ---------------------------------------------------------------------------- #
#                             Simulation Parameters                            #
# ---------------------------------------------------------------------------- #
const DX = T(0.8e-3)
const GAP = 2 * DX
const T_MAX = T(GAP / IMPACT_VELOCITY + 12e-6)
const PPC_1D = 2
const cfl_number = T(0.2)
const DT_MAX = T(1e-3)
const DOMAIN = OpenDomain()


# ---------------------------------------------------------------------------- #
#                                Export Settings                               #
# ---------------------------------------------------------------------------- #
NR_EXPORTS = 60


# ---------------------------------------------------------------------------- #
#                               Sphere Materials                               #
# ---------------------------------------------------------------------------- #
const TARGET_MATERIAL = Basalt(T, eos=:tillotson, elasticity=:hypoelastic, damage=:grady_kipp,
                               plasticity=J2Plasticity(σ_y0=T(3.5e9), H=T(5.31e6)),
                               viscosity=BulkViscosity{T}(c_lin=0.3))

const PROJECTILE_MATERIAL = Nylon(T, eos=:tillotson, elasticity=:hypoelastic,
                                  plasticity=J2Plasticity(σ_y0=T(10e6), H=T(2.19e6)),
                                  viscosity=BulkViscosity{T}(c_lin=0.3))

# ---------------------------------------------------------------------------- #
#                               Sphere Parameters                              #
# ---------------------------------------------------------------------------- #
const TARGET_POSITION = zero(SVector{3, T})
const TARGET_VELOCITY = zero(SVector{3, T})

const CONTACT_DISTANCE = TARGET_RADIUS + PROJECTILE_RADIUS
const PROJECTILE_POSITION = SVector{3, T}(-CONTACT_DISTANCE * cosd(IMPACT_ANGLE) - GAP,
                                          CONTACT_DISTANCE * sind(IMPACT_ANGLE), 0.0)
const PROJECTILE_VELOCITY = SVector{3, T}(IMPACT_VELOCITY, 0.0, 0.0)



# ---------------------------------------------------------------------------- #
#                                 Create Bodies                                #
# ---------------------------------------------------------------------------- #
println("Setting up simulation...")

target_shape     = Sphere{T}(TARGET_RADIUS, TARGET_POSITION)
projectile_shape = Sphere{T}(PROJECTILE_RADIUS, PROJECTILE_POSITION)

target_body     = Body(target_shape, TARGET_VELOCITY, zero(SVector{3, T}), TARGET_MATERIAL)
projectile_body = Body(projectile_shape, PROJECTILE_VELOCITY, zero(SVector{3, T}), PROJECTILE_MATERIAL)

bodies = (target_body, projectile_body)   # target first: particle set id 1 in the export


exporter = HDF5Exporter(
    output_dir="output",
    filename_prefix = "nakamura_fujiwara_",
    write_xdmf = true,
    compression_level = 3,
    export_particles = true,
    export_grid = false
)

boundary_condition = NullBoundaryCondition()

sim_setup = SimulationSetup(
    domain=DOMAIN,
    grid_type=SparseGrid,
    dx=DX,
    cfl_number=cfl_number,
    t_max=T_MAX,
    dt_max=DT_MAX,
    ppc_1d=PPC_1D,
    exporter=exporter,
    export_time_interval=T_MAX/NR_EXPORTS,
    backend=BACKEND,
    boundary_condition=boundary_condition
)

model = build_mpm_model(bodies, sim_setup)

N_particles = length(model.particle_sets[1].particles) + length(model.particle_sets[2].particles)
grid_dimensions = size(model.grid.state_read)
println("Simulation setup complete. Number of particles: $N_particles, Grid dimensions: $grid_dimensions")


solve!(model, sim_setup)