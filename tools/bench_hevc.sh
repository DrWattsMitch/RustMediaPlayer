#!/usr/bin/env bash
# HEVC decode benchmark: RMP vs FFmpeg at several thread counts.
# Usage: tools/bench_hevc.sh [runs] [clip...]
# Default clips: target/bench/*.mp4. Writes target/bench/results.md and prints it.
# CORES=n overrides the detected core count (the last thread column).
# Each number is the best wall time of [runs] runs (default 5), decode to null.
#
# RMP columns: "NT" = N frame threads only (RMP_HEVC_ROWS=1); "NT + 2 rows" =
# N frame threads plus 2 WPP row threads per picture. The default since the
# row-thread change is "+ 2 rows" from 8 threads up, frame threads only below.
set -euo pipefail
cd "$(dirname "$0")/.."
runs=${1:-5}; shift || true
clips=("$@"); [ ${#clips[@]} -eq 0 ] && clips=(target/bench/*.mp4)
cores=${CORES:-$( (sysctl -n hw.ncpu 2>/dev/null || nproc) | head -1)}
echo "Building release decoder..." >&2
cargo build --release --example decode -q
rmp=target/release/examples/decode
now() { perl -MTime::HiRes=time -e 'printf "%.4f\n", time'; }
best() { # best wall time of $runs runs of "$@"
	local b=999 s e t
	for _ in $(seq "$runs"); do
		s=$(now); "$@" >/dev/null 2>&1 || { echo "FAIL"; return; }; e=$(now)
		t=$(perl -e "printf '%.3f', $e - $s"); b=$(perl -e "print $t < $b ? $t : $b")
	done
	echo "$b"
}
# Thread counts: 1, 2, 4, 8 and the core count (each once, in order).
counts=()
for t in 1 2 4 8 "$cores"; do
	[ "$t" -le "$cores" ] || continue
	case " ${counts[*]-} " in *" $t "*) ;; *) counts+=("$t") ;; esac
done
out=target/bench/results.md
{
	echo "# HEVC benchmark ($(uname -sm), $cores cores, $(ffmpeg -version | head -1 | cut -d' ' -f1-3), best of $runs)"
	echo
	head="| clip | frames"
	for t in "${counts[@]}"; do
		head="$head | RMP ${t}T"
		[ "$t" -ge 4 ] && head="$head | RMP ${t}T + 2 rows"
	done
	head="$head | RMP 1 frame + ${cores} rows"
	for t in "${counts[@]}"; do head="$head | FFmpeg ${t}T"; done
	echo "$head |"
	sep="|"; for _ in $(seq $(($(echo "$head" | tr -cd '|' | wc -c)))); do sep="$sep---|"; done
	echo "$sep"
	for c in "${clips[@]}"; do
		echo "  $c" >&2
		frames=$(ffprobe -v error -count_packets -select_streams v:0 -show_entries stream=nb_read_packets -of csv=p=0 "$c" 2>/dev/null || echo "?")
		row="| $(basename "$c") | $frames"
		for t in "${counts[@]}"; do
			row="$row | $(best env RMP_THREADS=$t RMP_HEVC_ROWS=1 "$rmp" "$c" - -)"
			[ "$t" -ge 4 ] && row="$row | $(best env RMP_THREADS=$t RMP_HEVC_ROWS=2 "$rmp" "$c" - -)"
		done
		row="$row | $(best env RMP_HEVC_FRAMES=1 RMP_HEVC_ROWS=$cores "$rmp" "$c" - -)"
		for t in "${counts[@]}"; do row="$row | $(best ffmpeg -v error -threads $t -i "$c" -f null -)"; done
		echo "$row |"
	done
} > "$out"
cat "$out"
