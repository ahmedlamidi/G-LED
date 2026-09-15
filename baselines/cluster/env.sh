# Sourced by every baselines job script (not submitted itself). Sets up the
# same environment as ablations/*.sh and defines `run MODULE [args...]`.

cd "$SLURM_SUBMIT_DIR"
source /apps/anaconda3/etc/profile.d/conda.sh
CONDA_ENV=proj
conda activate "$CONDA_ENV"

export TMPDIR=/home/a/ahmedlamidi/tmp
export TEMP=/home/a/ahmedlamidi/tmp
export TMP=/home/a/ahmedlamidi/tmp

# Call the env's interpreter by absolute path so srun cannot fall back to
# system python if the job step does not inherit the activated environment.
PY="$CONDA_PREFIX/bin/python"

# Same check as ablations/*.sh: the RTX PRO 6000 (sm_120) needs a torch built
# with CUDA 12.8 kernels, so test for kernels, not just for an import.
TORCH_VERSION=2.8.0
TORCHVISION_VERSION=0.23.0

torch_matches_gpu() {
	"$PY" - <<-'EOF' 2>/dev/null
	import sys
	try:
	    import torch
	except ImportError:
	    sys.exit(1)
	if not torch.cuda.is_available():
	    sys.exit(1)
	arch = 'sm_%d%d' % torch.cuda.get_device_capability()
	sys.exit(0 if arch in torch.cuda.get_arch_list() else 1)
	EOF
}

install_deps() {
	"$PY" -m pip install --no-cache-dir -r requirements.txt
	"$PY" -m pip install --no-cache-dir --upgrade \
		"torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"
}

if [ "${INSTALL_DEPS:-0}" = "1" ]; then
	install_deps
elif ! torch_matches_gpu; then
	echo "torch missing or built without kernels for this GPU - installing"
	install_deps
fi
if ! torch_matches_gpu; then
	echo "ERROR: torch has no kernels for this GPU's compute capability."
	exit 1
fi

run() {
	echo "Running: python -m $*"
	# Python's own temp files go on the node's local disk, not the NFS TMPDIR
	# above: NFS keeps .nfs* placeholders for open files, so DataLoader worker
	# cleanup there fails with "Device or resource busy".
	TMPDIR="${SLURM_TMPDIR:-/tmp}" srun --export=ALL "$PY" -m "$@"
}
