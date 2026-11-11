#!/usr/bin/env bash
# Regenerates the test fixtures with FFmpeg (built with libmp3lame, libx264,
# libvpx, libopus and libvorbis). Fixture byte counts are asserted in tests, so if you change a
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

# --- Matroska / WebM, FLAC, Ogg, Opus and the processing layer ------------
# Needs FFmpeg with libx264, libvpx, libopus and libvorbis. References
# (`*.packets.csv`, `*.ffmpeg.*`) are derived from the fixtures themselves,
# so after re-encoding a fixture, rerun its reference line too. A few tests
# also assert sample counts, sizes or tags, which follow from these commands.
mkdir -p mkv flac ogg opus process
packets() { ffprobe -v error -show_packets -show_data_hash adler32 -of csv=p=0 \
	-show_entries packet=stream_index,pts,duration,size,flags,data_hash "$1" > "${1%.*}.packets.csv"; }
printf '1\n00:00:00,100 --> 00:00:00,500\nBonjour\n\n2\n00:00:00,600 --> 00:00:01,000\nAu revoir\n' > /tmp/rmp.srt
printf 'WEBVTT\n\n00:00:00.100 --> 00:00:00.500\nHello\n\n00:00:00.600 --> 00:00:01.000\nBye\n' > /tmp/rmp.vtt

ff -i mp4/h264_bframes_aac.mp4 -c copy mkv/h264_aac.mkv
ff -f lavfi -i testsrc2=s=64x48:r=25:d=2 -f lavfi -i "sine=f=330:d=2:sample_rate=44100" -i /tmp/rmp.srt \
	-map 0 -map 1 -map 2 -c:v libx264 -preset veryfast -c:a flac -c:s srt -metadata:s:a:0 language=fra \
	-metadata title="MKV Song" -metadata artist="RMP Tests" -metadata album="Fixtures" mkv/h264_flac_srt.mkv
ff -f lavfi -i testsrc2=s=64x48:r=25:d=1 -f lavfi -i "sine=f=440:d=1" -i /tmp/rmp.vtt -map 0 -map 1 -map 2 \
	-c:v libvpx-vp9 -deadline realtime -c:a libopus -ac 1 -b:a 32k -c:s webvtt -metadata title="WebM clip" mkv/vp9_opus_vtt.webm
# Live WebM: unknown element sizes, no cues or duration.
ff -f lavfi -i testsrc2=s=64x48:r=25:d=1 -f lavfi -i "sine=f=440:d=1" -c:v libvpx-vp9 -deadline realtime \
	-c:a libopus -ac 1 -b:a 32k -metadata title="WebM clip" -f webm -live 1 mkv/live_vp9_opus.webm
for f in mkv/*.mkv mkv/*.webm; do packets "$f"; done

ff -f lavfi -i "aevalsrc='0.5*sin(2*PI*440*t)|0.4*sin(2*PI*554*t)':s=44100:d=0.5" -c:a flac \
	-metadata title="FLAC Song" -metadata artist="RMP Tests" -metadata album="Fixtures" -metadata track="4/12" \
	-metadata genre="Ambient" -metadata date=2025 flac/s16_stereo_44k.flac
# Nearly identical channels (mid/side), level 8.
ff -f lavfi -i "aevalsrc='0.5*sin(2*PI*300*t)|0.5*sin(2*PI*300*t)+0.01*sin(2*PI*5000*t)':s=48000:d=0.3" \
	-c:a flac -compression_level 8 flac/s16_ms_48k_lvl8.flac
ff -f lavfi -i "anoisesrc=d=0.2:a=0.5:r=22050" -c:a flac flac/noise_22k.flac
# Samples that are multiples of 128 (7 wasted bits), level 0.
ff -f lavfi -i "aevalsrc='floor(0.5*sin(2*PI*200*t)*256)/256':s=8000:d=0.5" -c:a flac -compression_level 0 \
	flac/wasted_mono_8k_lvl0.flac
ff -f lavfi -i "aevalsrc='0.3*sin(2*PI*440*t)|0.3*sin(2*PI*550*t)|0.3*sin(2*PI*660*t)|0.1*sin(2*PI*60*t)|0.2*sin(2*PI*880*t)|0.2*sin(2*PI*990*t)':s=96000:d=0.06" \
	-c:a flac -sample_fmt s32 -bits_per_raw_sample 24 -compression_level 12 flac/s24_51_96k_lvl12.flac
ff -f lavfi -i "aevalsrc='0.6*sin(2*PI*700*t)|-0.6*sin(2*PI*700*t)+0.05*sin(2*PI*90*t)':s=32000:d=0.2" \
	-c:a flac -sample_fmt s32 -bits_per_raw_sample 24 flac/s24_rs_32k.flac
for f in flac/s16*.flac flac/noise_22k.flac flac/wasted*.flac; do ff -i "$f" -f s16le "${f%.flac}.ffmpeg.s16le"; done
for f in flac/s24*.flac; do ff -i "$f" -f s32le "${f%.flac}.ffmpeg.s32le"; done

# Ogg: Opus (tags, a METADATA_BLOCK_PICTURE cover, 5.1), FLAC and Vorbis.
ff -f lavfi -i "sine=f=220:d=4" -f lavfi -i "anoisesrc=d=4:a=0.05" -filter_complex "[0][1]amix=2,volume=2" -ar 48000 /tmp/rmp_voice_mono.wav
ff -f lavfi -i "sine=f=300:d=4" -f lavfi -i "sine=f=450:d=4" -f lavfi -i "anoisesrc=d=4:a=0.03" \
	-filter_complex "[0][2]amix=2[l];[1][2]amix=2[r];[l][r]join=inputs=2:channel_layout=stereo" -ar 48000 /tmp/rmp_voice_st.wav
cover=$(python3 -c "
import struct, base64
img = open('/tmp/rmp_cover.png', 'rb').read(); mime = b'image/png'; d = b'Cover'
print(base64.b64encode(struct.pack('>II', 3, len(mime)) + mime + struct.pack('>I', len(d)) + d
	+ struct.pack('>IIIII', 16, 16, 24, 0, len(img)) + img).decode())")
ff -i /tmp/rmp_voice_mono.wav -t 2 -c:a libopus -b:a 12k -metadata title="Ogg Test" -metadata artist="RMP" ogg/mono.opus
ff -i /tmp/rmp_voice_st.wav -t 3 -c:a libopus -b:a 48k -metadata album=Album -metadata METADATA_BLOCK_PICTURE="$cover" ogg/st_cover.opus
ff -f lavfi -i "sine=f=440:d=1.5" -af "aformat=channel_layouts=5.1" -c:a libopus -b:a 96k ogg/51.opus
ff -i /tmp/rmp_voice_st.wav -t 2 -ar 44100 -c:a flac -metadata title=FlacOgg ogg/flac.oga
ff -i /tmp/rmp_voice_st.wav -t 2 -c:a libvorbis -q:a 2 ogg/vorbis.ogg
for f in ogg/*.opus ogg/*.oga ogg/*.ogg; do packets "$f"; done

# Vorbis decoder references: FFmpeg's native float decoder (first-packet
# priming and Ogg end trimming applied, as RMP does).
mkdir -p vorbis
ff -i /tmp/rmp_voice_mono.wav -t 1 -ar 22050 -c:a libvorbis -q:a -1 vorbis/mono_22k_q-1.ogg
ff -i /tmp/rmp_voice_st.wav -t 0.8 -ar 44100 -c:a libvorbis -q:a 8 vorbis/st_44k_q8.webm
ff -f lavfi -i "aevalsrc='0.3*sin(2*PI*300*t)|0.3*sin(2*PI*500*t)|0.3*sin(2*PI*700*t)|0.2*sin(2*PI*90*t)|0.3*sin(2*PI*1100*t)|0.3*sin(2*PI*1300*t):s=22050:d=0.5:c=5.1'" \
	-c:a libvorbis -q:a 3 vorbis/51_22k.ogg
for f in vorbis/*.ogg vorbis/*.webm; do ff -c:a vorbis -i "$f" -f s16le "${f%.*}.ffmpeg.s16le"; done

# VP9 decoder references: per-frame adler32 (framecrc) of FFmpeg's decode.
mkdir -p vp9
ff -f lavfi -i testsrc2=s=176x144:r=25:d=1 -c:v libvpx-vp9 -b:v 150k -g 25 -auto-alt-ref 1 -lag-in-frames 10 vp9/altref_176x144.webm
ff -f lavfi -i "mandelbrot=s=99x61:r=25" -t 0.6 -c:v libvpx-vp9 -crf 40 -b:v 0 vp9/odd_444_99x61.webm
ff -f lavfi -i testsrc2=s=64x48:r=25:d=0.4 -c:v libvpx-vp9 -lossless 1 vp9/lossless_64x48.webm
ff -f lavfi -i testsrc2=s=512x128:r=25:d=0.4 -c:v libvpx-vp9 -b:v 200k -tile-columns 1 -tile-rows 2 vp9/tiles_512x128.webm
ff -f lavfi -i testsrc2=s=96x64:r=25:d=0.6 -pix_fmt yuv422p -c:v libvpx-vp9 -b:v 100k vp9/p1_422_96x64.webm
ff -f lavfi -i "testsrc2=s=160x96:r=30:d=0.8,noise=alls=12:allf=t" -c:v libvpx-vp9 -deadline realtime -cpu-used 8 \
	-aq-mode 3 -error-resilient 1 -b:v 150k vp9/rt_aq_errres_160x96.webm
ff -f lavfi -i "mandelbrot=s=128x72:r=30" -t 0.6 -pix_fmt yuv420p -c:v libvpx-vp9 -frame-parallel 1 -b:v 120k vp9/frame_parallel_128x72.webm
for f in vp9/*.webm; do ff -c:v vp9 -i "$f" -f framecrc "${f%.webm}.framecrc"; done

# Opus decoder references: libopus through FFmpeg, pre-skip kept.
ff -i /tmp/rmp_voice_mono.wav -t 1 -ar 8000 -c:a libopus -b:a 8k -application voip opus/silk_nb_mono.webm
ff -i /tmp/rmp_voice_st.wav -t 1 -ar 16000 -c:a libopus -b:a 24k -application voip -frame_duration 40 opus/silk_wb_st_40ms.webm
ff -i /tmp/rmp_voice_st.wav -t 1 -c:a libopus -b:a 20k -application voip opus/hybrid_st.webm
ff -f lavfi -i "sine=f=440:d=0.5" -af "aformat=channel_layouts=5.1" -c:a libopus -b:a 128k opus/celt_51.webm
ff -f lavfi -i "sine=f=220:d=0.6" -f lavfi -i "anoisesrc=d=0.6:a=0.3:c=pink" \
	-filter_complex "[0][1]concat=n=2:v=0:a=1,aformat=channel_layouts=mono,asplit[a][b];[b]areverse[r];[a][r]concat=n=2:v=0:a=1" \
	-c:a libopus -b:a 20k -frame_duration 10 opus/mode_switch.webm
for f in opus/*.webm; do ff -flags2 +skip_manual -c:a libopus -i "$f" -f s16le "${f%.webm}.ffmpeg.s16le"; done

# Opus packet-loss references (tests/codecs/opus_plc_test.rs): libopus 1.4
# itself, through tools/opus_plc_ref.c, since FFmpeg cannot drop packets.
# Needs a built libopus 1.4 checkout in $LIBOPUS (./configure && make).
ff -f lavfi -i "aevalsrc='0.25*(sin(2*PI*(140+20*sin(2*PI*0.7*t))*t)+0.5*sin(2*PI*2*(140+20*sin(2*PI*0.7*t))*t)+0.3*sin(2*PI*3*(140+20*sin(2*PI*0.7*t))*t))*gt(sin(2*PI*0.9*t),-0.2)+0.004*(random(0)-0.5)':s=48000:d=2" /tmp/rmp_speechlike.wav
ff -i /tmp/rmp_speechlike.wav -filter_complex "[0]asplit[a][b];[b]volume=0.8,adelay=3[c];[a][c]join=inputs=2:channel_layout=stereo" /tmp/rmp_speechlike_st.wav
ff -i /tmp/rmp_speechlike.wav -ar 16000 -c:a libopus -b:a 28k -application voip -fec 1 -packet_loss 30 opus/plc_silk_fec.webm
ff -i /tmp/rmp_speechlike_st.wav -c:a libopus -b:a 32k -application voip -fec 1 -packet_loss 20 opus/plc_hybrid_fec_st.webm
ff -i /tmp/rmp_speechlike.wav -c:a libopus -b:a 64k -application audio opus/plc_celt.webm
cc -O2 -I "$LIBOPUS/include" ../../tools/opus_plc_ref.c "$LIBOPUS/.libs/libopus.a" -lm -o /tmp/opus_plc_ref
# The dropped packets: the same rule as `lost` in opus_plc_test.rs.
loss_pattern() {
	python3 -c "
import struct, sys
d = open(sys.argv[1], 'rb').read(); at = 8 + d[4]; n = 0
while at < len(d):
    at += 4 + struct.unpack('<I', d[at:at + 4])[0]; n += 1
lost = lambda i: i < n - 1 and (i in (5, 9, 10) or 15 <= i < 23 or 30 <= i < 33 or i == n * 3 // 4)
print(''.join('x' if lost(i) else '.' for i in range(n)))" "$1"
}
for spec in silk_wb_st_40ms:plc hybrid_st:plc celt_51:plc mode_switch:plc plc_celt:plc \
	plc_silk_fec:plc plc_silk_fec:fec plc_hybrid_fec_st:plc plc_hybrid_fec_st:fec; do
	f=${spec%%:*} mode=${spec##*:}
	python3 ../../tools/opus_packets.py "opus/$f.webm" "/tmp/$f.opkt"
	/tmp/opus_plc_ref "/tmp/$f.opkt" "$(loss_pattern "/tmp/$f.opkt")" "$mode" "opus/$f.loss_$mode.s16le"
done

# Processing layer: libswscale / libswresample references.
ff -f lavfi -i testsrc2=s=96x64:r=25 -frames:v 1 -pix_fmt yuv420p process/testsrc_96x64.y4m
ff -i process/testsrc_96x64.y4m -vf scale=100:75 -f rawvideo -pix_fmt yuv420p process/testsrc_100x75.yuv
ff -i process/testsrc_96x64.y4m -vf scale=48:32 -f rawvideo -pix_fmt yuv420p process/testsrc_48x32.yuv
ff -i process/testsrc_96x64.y4m -f rawvideo -pix_fmt rgb24 process/testsrc_96x64.rgb24
ff -i process/testsrc_96x64.y4m -sws_flags bicubic+accurate_rnd+full_chroma_int -f rawvideo -pix_fmt rgb24 \
	process/testsrc_96x64.accurate.rgb24
ff -f rawvideo -pix_fmt rgb24 -s 96x64 -i process/testsrc_96x64.rgb24 -f rawvideo -pix_fmt yuv420p process/from_rgb_96x64.yuv
ff -f rawvideo -pix_fmt rgb24 -s 96x64 -i process/testsrc_96x64.rgb24 -sws_flags bicubic+accurate_rnd+full_chroma_int \
	-f rawvideo -pix_fmt yuv420p process/from_rgb_96x64.accurate.yuv
ff -f lavfi -i "aevalsrc='0.5*sin(2*PI*1000*t)|0.4*sin(2*PI*3000*t)':s=44100:d=0.25" -c:a pcm_s16le process/tone_44k_stereo.wav
ff -i process/tone_44k_stereo.wav -ar 48000 -f f32le process/tone_48k_stereo.ffmpeg.f32le
ff -i process/tone_44k_stereo.wav -ar 8000 -ac 1 -f f32le process/tone_8k_mono.ffmpeg.f32le
