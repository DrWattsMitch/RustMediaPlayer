#!/usr/bin/env bash
# Regenerates the MP3/MP4 test fixtures with FFmpeg (built with libmp3lame
# and libx264). Fixture byte counts are asserted in tests, so if you change a
# command here, update the expected packet counts (see `ffprobe -count_packets`).
set -euo pipefail
cd "$(dirname "$0")/../tests/fixtures"
mkdir -p mp3 mp4
ff() { ffmpeg -hide_banner -loglevel error -y "$@"; }

ff -f lavfi -i "aevalsrc='0.4*sin(2*PI*440*t)+0.1*sin(2*PI*2500*t)*lt(mod(t,0.25),0.02)|0.4*sin(2*PI*660*t)':s=44100:d=1" \
	-c:a libmp3lame -b:a 128k -metadata title="RMP Fixture" -metadata artist="RMP Tests" -metadata album="Fixtures" \
	-metadata track="3/7" -metadata genre="Electronic" -metadata date=2026 -metadata comment="cbr fixture" \
	-id3v2_version 4 mp3/cbr_44k_stereo_tagged.mp3
ff -f lavfi -i "aevalsrc='0.4*sin(2*PI*440*t)|0.3*sin(2*PI*330*t)':s=44100:d=1" -c:a libmp3lame -q:a 4 \
	-id3v2_version 3 -write_id3v1 1 -metadata title="VBR" mp3/vbr_44k_xing_id3v1.mp3
ff -f lavfi -i "aevalsrc='0.5*sin(2*PI*300*t)+0.2*sin(2*PI*1700*t)':s=22050:d=1" -ac 1 -c:a libmp3lame -b:a 32k \
	-write_xing 0 -id3v2_version 0 mp3/mpeg2_22k_mono_raw.mp3
# Reference decode, without gapless trimming, for the decoder accuracy test.
ff -flags2 +skip_manual -i mp3/mpeg2_22k_mono_raw.mp3 -f s16le -c:a pcm_s16le mp3/mpeg2_22k_mono_raw.ffmpeg.s16le

ff -f lavfi -i testsrc2=s=64x48:r=25:d=1 -f lavfi -i "sine=f=440:d=1:sample_rate=48000" -c:v libx264 -preset veryfast \
	-bf 2 -g 10 -pix_fmt yuv420p -c:a aac -b:a 48k -metadata title="Clip" -metadata artist="RMP Tests" mp4/h264_bframes_aac.mp4
ff -f lavfi -i color=blue:s=16x16:d=1 -frames:v 1 /tmp/rmp_cover.png
ff -f lavfi -i "sine=f=440:d=1:sample_rate=44100" -i /tmp/rmp_cover.png -map 0 -map 1 -c:a aac -b:a 48k -c:v copy \
	-disposition:v attached_pic -metadata title="Song" -metadata artist="Band" -metadata album="Record" \
	-metadata track="2/9" -metadata disc="1/2" -metadata genre="Jazz" -metadata date=2023 -metadata comment="m4a fixture" \
	mp4/aac_tagged_cover.m4a

# --- WAV (PCM layouts, ADPCM) and Y4M: need only FFmpeg's built-in codecs. ---
mkdir -p wav/pcm wav/adpcm y4m
# One 16-bit source; every PCM layout is FFmpeg's encode of it, so the PCM
# tests check both directions (decode -> reference, encode source -> fixture).
ff -f lavfi -i "aevalsrc='0.9*sin(2*PI*440*t)+0.1*sin(2*PI*3000*t)':s=8000:d=0.1" -c:a pcm_s16le wav/pcm/s16le.wav
for codec in u8 s24le s32le s64le f32le f64le alaw mulaw; do
	ff -i wav/pcm/s16le.wav -c:a pcm_$codec wav/pcm/$codec.wav
done
for f in wav/pcm/*.wav; do ff -i "$f" -f s16le "${f%.wav}.ffmpeg.s16le"; done

ff -f lavfi -i "aevalsrc='0.6*sin(2*PI*440*t)+0.2*sin(2*PI*2500*t)*lt(mod(t,0.1),0.02)|0.5*sin(2*PI*660*t)':s=22050:d=0.25" \
	-c:a adpcm_ima_wav wav/adpcm/ima_stereo_22k.wav
ff -f lavfi -i "aevalsrc='0.7*sin(2*PI*300*t)+0.2*sin(2*PI*1900*t)':s=8000:d=0.5" -c:a adpcm_ms -block_size 256 \
	wav/adpcm/ms_mono_8k.wav
ff -f lavfi -i "aevalsrc='0.5*sin(2*PI*500*t)|0.5*sin(2*PI*750*t)':s=11025:d=0.3" -c:a adpcm_ms wav/adpcm/ms_stereo_11k.wav
for f in wav/adpcm/*.wav; do ff -i "$f" -f s16le "${f%.wav}.ffmpeg.s16le"; done

ff -f lavfi -i testsrc2=s=34x18:r=30000/1001 -frames:v 3 -pix_fmt yuv420p y4m/yuv420p_34x18.y4m
ff -f lavfi -i testsrc2=s=16x8:r=25 -frames:v 2 -pix_fmt yuv422p10le -strict -1 y4m/yuv422p10_16x8.y4m
for f in y4m/*.y4m; do ff -i "$f" -f rawvideo -c:v rawvideo "${f%.y4m}.ffmpeg.yuv"; done
