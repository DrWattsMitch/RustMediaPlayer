#!/usr/bin/env bash
# Regenerates the High 10 / 4:2:2 / interlaced H.264 fixtures in
# tests/fixtures/h264 and prints FFmpeg's reference hashes for them
# (FNV-1a 64 of the decoded planar output, as the tests compute it).
# Needs FFmpeg with a 10-bit capable libx264, and python3.
#
# x264 writes neither field pictures nor mixed field/MBAFF streams; those
# are spliced from separately encoded top and bottom field streams by
# tools/mkpaff.py and tools/mkmix.py (CAVLC only: field pictures use other
# CABAC contexts). The results are valid streams: what matters is that
# FFmpeg and RMP decode them identically.
set -euo pipefail
cd "$(dirname "$0")/../tests/fixtures/h264"
TOOLS="$(cd ../../../tools && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
ff() { ffmpeg -loglevel error -y "$@"; }
PROG="testsrc2=s=128x96:r=25:d=0.48,noise=alls=6:allf=t"
INTL="testsrc2=s=128x96:r=50:d=0.48,noise=alls=6:allf=t,tinterlace=mode=interleave_top"
FIXED="scenecut=0:b-adapt=0:keyint=6:min-keyint=6:weightp=0:weightb=0"

x264() { # out source pix_fmt params [vf]
	ff -f lavfi -i "$2" ${5:+-vf "$5"} -pix_fmt "$3" -c:v libx264 -preset medium -crf 26 -x264-params "$4" "$1"
}

# Progressive High 10 and High 4:2:2 (cropped to 120x88).
x264 h264_high10.264 "$PROG,crop=120:88:0:0" yuv420p10le "bframes=2:b-pyramid=normal:weightp=2:ref=3"
x264 h264_high422_10.264 "$PROG,crop=120:88:0:0" yuv422p10le "bframes=2:weightp=1:ref=2"
x264 h264_high422_cavlc.264 "$PROG,crop=120:88:0:0" yuv422p "cabac=0:bframes=1:ref=2"

# MBAFF (x264 --interlaced with adaptive frame/field pairs).
x264 h264_mbaff.264 "$INTL" yuv420p "interlaced=1:bframes=2:b-pyramid=normal:weightp=2:ref=3"
x264 h264_mbaff_422_10_cavlc.264 "$INTL" yuv422p10le "interlaced=1:cabac=0:bframes=1:ref=2:direct=temporal"

# PAFF field pictures, temporal direct, bottom field first.
for f in top bottom; do
	ff -f lavfi -i "$INTL" -vf "field=type=$f" -pix_fmt yuv420p -c:v libx264 -preset medium -qp 26 \
		-x264-params "cabac=0:$FIXED:bframes=1:b-pyramid=none:ref=2:direct=temporal" "$TMP/paff.$f.264"
done
python3 "$TOOLS/mkpaff.py" "$TMP/paff.top.264" "$TMP/paff.bottom.264" h264_paff_bff.264 --bff

# MBAFF frames and field pairs in one stream, referencing each other.
ff -f lavfi -i "$INTL" -pix_fmt yuv420p -c:v libx264 -preset medium -qp 26 \
	-x264-params "interlaced=1:cabac=0:$FIXED:bframes=1:b-pyramid=none:ref=2:direct=temporal" "$TMP/mix.mbaff.264"
for f in top bottom; do
	ff -f lavfi -i "$INTL" -vf "field=type=$f" -pix_fmt yuv420p -c:v libx264 -preset medium -qp 26 \
		-x264-params "cabac=0:$FIXED:bframes=1:b-pyramid=none:ref=2:direct=temporal" "$TMP/mix.$f.264"
done
python3 "$TOOLS/mkmix.py" "$TMP/mix.mbaff.264" "$TMP/mix.top.264" "$TMP/mix.bottom.264" h264_paff_mbaff_mix.264 FfF

for s in h264_*.264; do
	pf=$(ffprobe -v error -show_entries stream=pix_fmt -of csv=p=0 "$s")
	ff -i "$s" -f rawvideo -pix_fmt "$pf" "$TMP/out.yuv"
	python3 - "$TMP/out.yuv" "$s" <<'PY'
import sys
h = 0xcbf29ce484222325
for b in open(sys.argv[1], 'rb').read():
    h = ((h ^ b) * 0x100000001b3) & 0xffffffffffffffff
print(f"{sys.argv[2]}: 0x{h:016x}")
PY
done
