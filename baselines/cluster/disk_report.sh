#!/bin/bash
# Where the disk space goes: writes disk_report.txt (or the file given as $1).
# Run from the repo root, on the login node; it only reads.
#
# Fast by default: ONE walk of the repo (the raw DICOMs under data/LIDC-IDRI and
# .git are skipped: many small files, known size) and nothing outside it.
# FULL=1 also sizes the home folder's top level and the conda environments,
# which means walking hundreds of thousands of small files: minutes on NFS.
#
#   bash baselines/cluster/disk_report.sh
#   FULL=1 bash baselines/cluster/disk_report.sh ~/space.txt

OUT="${1:-disk_report.txt}"
TOP="${TOP:-25}"
TMP="$(mktemp)"
trap 'rm -f "$TMP" "$TMP.files"' EXIT

section() { printf '\n==== %s ====\n' "$1"; }
human() { numfmt --to=iec --suffix=B --format='%.1f' 2>/dev/null || cat; }
# entries exactly `depth` levels below `prefix` from the single du listing, largest first
level() {    # level PREFIX DEPTH
	awk -F'\t' -v p="$1" -v d="$2" '
		index($2, p) == 1 { rest = substr($2, length(p) + 1); n = gsub(/\//, "/", rest); if (rest != "" && n == d - 1) print }' "$TMP" |
		sort -rn | head -n "$TOP" | awk -F'\t' '{printf "%s\t%s\n", $1, $2}' |
		while IFS=$'\t' read -r kb path; do printf '%8s  %s\n' "$(echo $((kb * 1024)) | human)" "$path"; done
}

START=$SECONDS
# the one walk: sizes in KB of every directory to depth 5, plus every big file
du -k --max-depth=5 --exclude=LIDC-IDRI --exclude=.git . 2>/dev/null > "$TMP"
find . \( -name LIDC-IDRI -o -name .git \) -prune -o -type f -size +100M -printf '%k\t%p\n' 2>/dev/null > "$TMP.files"

{
	echo "Disk report, $(date '+%Y-%m-%d %H:%M'), host $(hostname), repo $(pwd)"
	echo "(data/LIDC-IDRI and .git not walked; FULL=1 adds the home folder and conda)"

	section "quota and filesystem"
	quota -s 2>/dev/null || echo "(no quota command)"
	df -h "$HOME" . 2>/dev/null

	section "repo total"
	awk -F'\t' '$2 == "." {print $1 * 1024}' "$TMP" | human

	section "repo root"
	level "./" 1
	section "data/"
	level "./data/" 1
	section "data/baselines_cache/   (sino = cached sinograms, shared by every setting and method: keep)"
	level "./data/baselines_cache/" 1
	section "data/baselines_cache/<setting>/<split>/"
	level "./data/baselines_cache/" 2
	section "output/"
	level "./output/" 1
	section "output/baselines/<setting>/"
	level "./output/baselines/" 1
	section "output/baselines/<setting>/<method>/"
	level "./output/baselines/" 2
	section "best_model_folder/"
	level "./best_model_folder/" 1

	section "files over 100 MB (checkpoints, arrays), largest first"
	sort -rn "$TMP.files" | head -n 40 | while IFS=$'\t' read -r kb path; do
		printf '%8s  %s%s\n' "$(echo $((kb * 1024)) | human)" "$path" "$([ -L "$path" ] && echo '  (link)')"; done
	echo "count: $(wc -l < "$TMP.files") files, $(awk -F'\t' '{s += $1} END {print s * 1024}' "$TMP.files" | human) in total"

	section "SLURM logs"
	echo "slurm-*.out: $(ls slurm-*.out 2>/dev/null | wc -l) files, $(du -ch slurm-*.out 2>/dev/null | tail -1 | cut -f1)"

	if [ "${FULL:-0}" = "1" ]; then
		section "home folder top level (FULL=1)"
		du -sh "$HOME"/* "$HOME"/.[!.]* 2>/dev/null | sort -rh | head -n "$TOP"
		section "conda environments (FULL=1)"
		du -sh "$HOME"/.conda/envs/* "$HOME/.conda/pkgs" 2>/dev/null | sort -rh
	fi
	echo
	echo "took $((SECONDS - START)) s"
} > "$OUT" 2>&1

echo "wrote $OUT ($(wc -l < "$OUT") lines, $((SECONDS - START)) s)"
