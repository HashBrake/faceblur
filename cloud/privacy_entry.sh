#!/usr/bin/env bash
# What the privacy image runs: every DF2 file in /in, de-identified into /out.
#
#   docker run --gpus all -v <in>:/in:ro -v <out>:/out faceblur-privacy
#
# One input writes /out/episode.h5 and /out/audit.jsonl; several write one
# folder each, /out/<stem>/. Extra run flags come from FACEBLUR_ARGS, e.g.
# -e FACEBLUR_ARGS="--spec-layout --mask face". Exit 2: nothing to do; 1: a
# run failed, or its audit did not verify.
set -euo pipefail
cd /app

# The weights are the ones the image was built and reviewed with, or nothing
# runs. A mismatch exits here through set -e.
sha256sum -c models/SHA256SUMS

# The audit names the image it ran in when it can. Without it the run is still
# valid, only less traceable, so this warns and goes on.
if [ ! -f /etc/faceblur_image_digest ] && [ -z "${FACEBLUR_IMAGE_DIGEST:-}" ]; then
    echo "note: no image digest (/etc/faceblur_image_digest or FACEBLUR_IMAGE_DIGEST); the audit will not name the image" >&2
fi

shopt -s nullglob
inputs=(/in/*.h5 /in/*.hdf5)
if [ ${#inputs[@]} -eq 0 ]; then
    echo "no input: put one or more .h5 or .hdf5 files in /in" >&2
    exit 2
fi

failed=0
outs=()
for f in "${inputs[@]}"; do
    if [ ${#inputs[@]} -eq 1 ]; then
        out=/out
    else
        stem=$(basename "$f"); stem=${stem%.*}
        out=/out/$stem
    fi
    # FACEBLUR_ARGS unquoted on purpose: it holds several flags.
    # shellcheck disable=SC2086
    if python -m faceblur.h5 run "$f" --out "$out" --workers "${FACEBLUR_WORKERS:-2}" ${FACEBLUR_ARGS:-}; then
        outs+=("$out")
    else
        echo "failed: $f" >&2
        failed=1
    fi
done

# A run that says it finished but leaves an audit that does not verify is a
# failure too: the audit is what the receiver checks.
for out in "${outs[@]}"; do
    if ! python -m faceblur.h5 verify "$out/audit.jsonl"; then
        echo "audit does not verify: $out/audit.jsonl" >&2
        failed=1
    fi
done
exit $failed
