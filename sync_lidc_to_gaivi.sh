#!/bin/bash
# Sync the LIDC-IDRI dataset to the GAIVI cluster, into the repo's data/ folder
# where Efficient_LIDC_DicomDataset looks for it (root_dir="data/LIDC-IDRI").
#
#   ./sync_lidc_to_gaivi.sh              # transfer; safe to re-run, resumes where it stopped
#   DRY_RUN=1 ./sync_lidc_to_gaivi.sh    # show what would be sent, send nothing
#   VERIFY=1 ./sync_lidc_to_gaivi.sh     # checksum-compare both sides, send nothing
#   REPAIR=1 ./sync_lidc_to_gaivi.sh     # re-send every file whose checksum differs
#
# SRC, REMOTE and DEST_DIR can each be overridden from the environment.
# REMOTE="" makes DEST_DIR a local path, which is handy for a test copy.

set -euo pipefail

SRC="${SRC:-/home/ahmed-lamidi/Documents/Yi_Sheng_lab/SD-FLow/G-LED/data/LIDC-IDRI}"
REMOTE="${REMOTE-ahmedlamidi@gaivi.cse.usf.edu}"
DEST_DIR="${DEST_DIR:-/home/a/ahmedlamidi/G-led/data/LIDC-IDRI}"

if [ ! -d "$SRC" ]; then
	echo "ERROR: source folder not found: $SRC" >&2
	exit 1
fi

if [ -n "$REMOTE" ]; then
	TARGET="$REMOTE:$DEST_DIR/"
else
	TARGET="$DEST_DIR/"
	mkdir -p "$DEST_DIR"
fi

echo "Source : $SRC  ($(du -sh "$SRC" | cut -f1), $(find "$SRC" -type f | wc -l) files)"
echo "Target : $TARGET"
echo

# -a        keep the patient/study/series tree, permissions and timestamps
# -h        human-readable sizes
# -z        compress on the wire; DICOM pixel data compresses well
# --partial keep half-sent files, so an interrupted run resumes mid-file
# No --delete: files that exist only on the cluster are never removed.
#
# The trailing slash on "$SRC/" copies the folder's contents, so the result is
# DEST_DIR/LIDC-IDRI-0001/..., not DEST_DIR/LIDC-IDRI/LIDC-IDRI-0001/...
OPTS=(-a -h -z --partial)

# Create the destination on the cluster in the same connection. This goes
# through --rsync-path rather than --mkpath, which older cluster rsyncs lack.
if [ -n "$REMOTE" ]; then
	OPTS+=(--rsync-path="mkdir -p '$DEST_DIR' && rsync")
fi

# A plain run decides what to send by size + mtime, so it skips a file whose
# content differs but whose size and timestamp match. REPAIR compares by
# checksum instead, which is what fixes a mismatch VERIFY reports.
if [ "${REPAIR:-0}" = "1" ]; then
	OPTS+=(-c)
fi

if [ "${VERIFY:-0}" = "1" ]; then
	# Checksum every file on both sides without sending anything. Each ">f"
	# line is a file that is missing or differs on the cluster.
	echo "Verifying by checksum (reads all $(du -sh "$SRC" | cut -f1) on both sides) ..."
	DIFFS=$(rsync "${OPTS[@]}" -n -c -i "$SRC/" "$TARGET" | grep '^>f' || true)
	if [ -z "$DIFFS" ]; then
		echo "OK: every file on the cluster matches the local copy."
	else
		echo "$DIFFS"
		echo
		echo "MISMATCH: $(echo "$DIFFS" | wc -l) file(s) missing or different. Fix with:  REPAIR=1 $0"
		exit 1
	fi
elif [ "${DRY_RUN:-0}" = "1" ]; then
	rsync "${OPTS[@]}" -n --stats "$SRC/" "$TARGET"
else
	rsync "${OPTS[@]}" --info=progress2 "$SRC/" "$TARGET"
	echo
	echo "Done. To confirm the copy is intact:  VERIFY=1 $0"
fi
