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
const IMPACT_VELOCITY = 3.2e3 # m/s
const IMPACT_PARAMETER = 0    # Impact parameter (0 = head-on, 1 = grazing)
const SPHERE_RADIUS = 7e-3


# ---------------------------------------------------------------------------- #
#                             Simulation Parameters                            #
# ---------------------------------------------------------------------------- #
const DX = T(1e-3)
const INITIAL_SEPARATION = 2.0 * SPHERE_RADIUS + 2*DX
time_to_collision = (INITIAL_SEPARATION - 2 * SPHERE_RADIUS) / IMPACT_VELOCITY
const T_MAX = T(time_to_collision + 100e-6)  # Run simulation for thrice the time to collision
const PPC_1D = 2
const cfl_number = T(0.4)
const DT_MAX = T(1e-3)
const DOMAIN = OpenDomain()


# ---------------------------------------------------------------------------- #
#                                Export Settings                               #
# ---------------------------------------------------------------------------- #
NR_EXPORTS = 100 


# ---------------------------------------------------------------------------- #
#                               Sphere Materials                               #
# ---------------------------------------------------------------------------- #
ρ_nylon = T(1180.0) # kg/m^3
mu_nylon = T(219e6 / (2 * (1 + 0.45)))
nylon_eos = TillotsonEOS{T}(
    ρ_nylon,
    T(10.1e9),   # A   (Pa)
    T(33.8e9),   # B   (Pa)
    T(10.0),     # α
    T(5.0),      # β
    T(7.0e6),    # E0  (J/kg)
    T(2.0e6),    # Eiv (J/kg)
    T(2.4e6),    # Ecv (J/kg)
    T(0.6),      # a
    T(2.0),      # b
)
nylon_elasticity = HyperElasticity{T}(mu_nylon)
nylon_plasticity = J2Plasticity(σ_y0=3.5e9, H=5.31e6)
nylon_damage = NoDamage()

const SPHERE1_MATERIAL = SolidMaterial(; eos=nylon_eos, elasticity=nylon_elasticity, plasticity=nylon_plasticity, damage=nylon_damage)

const SPHERE2_MATERIAL = Basalt(T, eos=:tillotson, elasticity=:hypoelastic, damage=:grady_kipp)

# ---------------------------------------------------------------------------- #
#                               Sphere Parameters                              #
# ---------------------------------------------------------------------------- #
const SPHERE1_POSITION = SVector{3, T}(-INITIAL_SEPARATION/2, 0.0, 0.0)
const SPHERE1_VELOCITY = SVector{3, T}(IMPACT_VELOCITY/2, 0.0, 0.0)

const SPHERE2_POSITION = SVector{3, T}(INITIAL_SEPARATION/2, IMPACT_PARAMETER * 2 * SPHERE_RADIUS, 0.0)
const SPHERE2_VELOCITY = SVector{3, T}(-IMPACT_VELOCITY/2, 0.0, 0.0)



# ---------------------------------------------------------------------------- #
#                                 Crate Spheres                                #
# ---------------------------------------------------------------------------- #
println("Setting up simulation...")

sphere1_shape = Sphere{T}(SPHERE_RADIUS, SPHERE1_POSITION)
sphere2_shape = Sphere{T}(SPHERE_RADIUS, SPHERE2_POSITION)

sphere1_body = Body(sphere1_shape, SPHERE1_VELOCITY, zero(SVector{3, T}), SPHERE1_MATERIAL)
sphere2_body = Body(sphere2_shape, SPHERE2_VELOCITY, zero(SVector{3, T}), SPHERE2_MATERIAL)

bodies = (sphere1_body, sphere2_body)


exporter = HDF5Exporter(
    output_dir="output",
    filename_prefix = "colliding_spheres_",
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