#!/bin/bash
#SBATCH --job-name=nakamura_fujiwara
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

module purge
module load devel/cuda/13.3

export PATH="$HOME/.juliaup/bin:$PATH"

ws_dir="$(ws_find smashmpm)"
if [ ! -d "$ws_dir" ]; then
    echo "FEHLER: Workspace 'smashmpm' nicht gefunden oder abgelaufen!" >&2
    exit 1
fi

sim_dir="${ws_dir}/simulations/SmashMPM.jl-Examples/NakamuraFujiwara1991"
project_path="${ws_dir}/development/SmashMPM.jl"
julia_cmd=(julia -t "${SLURM_CPUS_PER_TASK}" --project="${project_path}")

# Debug-Lauf mit Exception-Details aus den GPU-Kernels: DEBUG=1 sbatch run_nakamura_fujiwara.sh
[ "${DEBUG:-0}" = "1" ] && julia_cmd+=(-g2)

cd "${sim_dir}"
echo "Starte Simulation auf Node: $(hostname) mit GPU: ${CUDA_VISIBLE_DEVICES:-?}"

set +e
"${julia_cmd[@]}" nakamura_fujiwara.jl
sim_status=$?
set -e
echo "Simulation beendet mit Exit-Code ${sim_status}"

# Läuft auch nach einem Crash, um den letzten Zustand zu untersuchen
# "${julia_cmd[@]}" check_particles.jl

if [ "${sim_status}" -eq 0 ]; then
    # "${julia_cmd[@]}" fragment_analysis.jl
    "${julia_cmd[@]}" conservation_check.jl output
    "${julia_cmd[@]}" compare_dissipaton.jl output
fi



exit "${sim_status}"