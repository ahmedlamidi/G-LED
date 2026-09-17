#!/bin/bash
# Where the disk space goes: writes disk_report.txt (or the file given as $1).
# Run from the repo root, on the login node; it only reads.
#
#   bash baselines/cluster/disk_report.sh
#   bash baselines/cluster/disk_report.sh ~/space_$(date +%m%d).txt

OUT="${1:-disk_report.txt}"
TOP="${TOP:-25}"          # lines per section

section() { printf '\n==== %s ====\n' "$1"; }
# sizes of the entries of a folder, largest first, human-readable
sizes() { [ -d "$1" ] && du -sh "$1"/* "$1"/.[!.]* 2>/dev/null | sort -rh | head -n "$TOP"; }

{
	echo "Disk report, $(date '+%Y-%m-%d %H:%M'), host $(hostname), repo $(pwd)"

	section "quota and filesystem"
	quota -s 2>/dev/null || echo "(no quota command)"
	df -h "$HOME" . 2>/dev/null

	section "home folder: $HOME"
	sizes "$HOME"

	section "repo root"
	sizes .

	section "data/"
	sizes data

	section "data/baselines_cache/ (sino = cached sinograms, shared by every setting and method)"
	sizes data/baselines_cache

	section "data/baselines_cache/<setting>/<split>/ arrays"
	for d in data/baselines_cache/limited*/ data/baselines_cache/sword_sweep/limited*/; do
		[ -d "$d" ] || continue
		echo "-- $d  $(du -sh "$d" 2>/dev/null | cut -f1)"
		du -shL "$d"*/*.npy 2>/dev/null | sort -rh | head -n 8 | sed 's/^/     /'
		# linked labels take no space of their own
		find "$d" -maxdepth 2 -type l -name '*.npy' 2>/dev/null | sed 's/^/     (link) /'
	done

	section "output/"
	sizes output

	section "output/baselines/<setting>/<method>/"
	du -sh output/baselines/*/* 2>/dev/null | sort -rh | head -n "$TOP"

	section "checkpoints (*.pt, best_model_sofar, checkpoint_epoch_*) over 100 MB"
	find . -type f \( -name '*.pt' -o -name 'best_model_sofar' -o -name 'latest_checkpoint' -o -name 'checkpoint_epoch_*' \) \
		-size +100M -exec du -h {} + 2>/dev/null | sort -rh | head -n "$TOP"

	section "best_model_folder/"
	sizes best_model_folder

	section "SLURM logs and wandb"
	echo "slurm-*.out: $(ls slurm-*.out 2>/dev/null | wc -l) files, $(du -ch slurm-*.out 2>/dev/null | tail -1 | cut -f1)"
	du -sh wandb 2>/dev/null

	section "conda environments"
	sizes "$HOME/.conda/envs"
	du -sh "$HOME/.conda/pkgs" "$HOME/.cache" 2>/dev/null

	section "30 largest files under the repo"
	find . -type f -size +200M -exec du -h {} + 2>/dev/null | sort -rh | head -n 30
} > "$OUT" 2>&1

echo "wrote $OUT ($(wc -l < "$OUT") lines)"
