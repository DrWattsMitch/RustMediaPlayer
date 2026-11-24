#!/usr/bin/env python3
"""Encodes a VP9 stream whose frame size changes without key frames, so the
decoder must predict from scaled references (libvpx `vp9_scale.c`).

FFmpeg's libvpx wrapper cannot change the size of a running encode, so this
drives libvpx's encoder API through ctypes: `vpx_codec_enc_config_set` with a
new `g_w`/`g_h` makes libvpx keep coding inter frames from references of the
old size (it needs one-pass encoding and at most one frame of lag).

    vp9_resize_enc.py LIBVPX OUT.ivf FORMAT SEGMENT...

FORMAT is the bit depth (8, 10 or 12), optionally followed by `:444` for
4:4:4 (default 4:2:0). SEGMENT is `WxH:FRAMES:RAWFILE`, RAWFILE holding at
least FRAMES planar frames at that size (8-bit, or 16-bit little-endian
above 8 bits), as `ffmpeg -f rawvideo -pix_fmt yuv4{20,44}p[10le|12le]`
writes it.
Segments must not be larger than the first (libvpx would force a key frame)
and each must be within 2x down / 16x up of the previous one.
"""

import ctypes as C
import struct
import sys

VPX_IMG_FMT_I420 = 0x102
VPX_IMG_FMT_I444 = 0x106
VPX_IMG_FMT_HIGHBITDEPTH = 0x800
VPX_CODEC_USE_HIGHBITDEPTH = 0x40000
VPX_DL_GOOD_QUALITY = 1000000
VP8E_SET_CPUUSED = 13

# Leading fields of vpx_codec_enc_cfg_t (stable across libvpx versions).
CFG_PROFILE, CFG_W, CFG_H, CFG_BIT_DEPTH, CFG_INPUT_BIT_DEPTH = 8, 12, 16, 20, 24
CFG_TB_NUM, CFG_TB_DEN, CFG_LAG = 28, 32, 44
# vpx_image_t (64-bit): planes at 48, stride at 80.
IMG_PLANES, IMG_STRIDE = 48, 80


def main():
	lib = C.CDLL(sys.argv[1])
	out_path, fmt_arg = sys.argv[2], sys.argv[3].split(":")
	bit_depth, yuv444 = int(fmt_arg[0]), fmt_arg[1:] == ["444"]
	segments = []
	for s in sys.argv[4:]:
		size, frames, raw = s.split(":")
		w, h = map(int, size.split("x"))
		segments.append((w, h, int(frames), raw))

	lib.vpx_codec_vp9_cx.restype = C.c_void_p
	iface = lib.vpx_codec_vp9_cx()
	cfg = (C.c_ubyte * 4096)()
	if lib.vpx_codec_enc_config_default(C.c_void_p(iface), cfg, 0):
		sys.exit("vpx_codec_enc_config_default failed")
	u32 = lambda off, v: struct.pack_into("<I", cfg, off, v)
	w0, h0 = segments[0][0], segments[0][1]
	u32(CFG_W, w0)
	u32(CFG_H, h0)
	u32(CFG_TB_NUM, 1)
	u32(CFG_TB_DEN, 25)
	u32(CFG_LAG, 0)
	u32(CFG_PROFILE, (2 if bit_depth > 8 else 0) + int(yuv444))
	if bit_depth > 8:
		u32(CFG_BIT_DEPTH, bit_depth)
		u32(CFG_INPUT_BIT_DEPTH, bit_depth)
	ctx = (C.c_ubyte * 512)()
	flags = VPX_CODEC_USE_HIGHBITDEPTH if bit_depth > 8 else 0
	# The ABI version differs between libvpx releases: try them.
	for abi in range(64):
		if lib.vpx_codec_enc_init_ver(ctx, C.c_void_p(iface), cfg, C.c_long(flags), abi) == 0:
			break
	else:
		sys.exit("vpx_codec_enc_init_ver failed for every ABI version")
	lib.vpx_codec_control_(ctx, VP8E_SET_CPUUSED, C.c_int(2))

	lib.vpx_img_alloc.restype = C.c_void_p
	lib.vpx_codec_get_cx_data.restype = C.c_void_p
	bps = 2 if bit_depth > 8 else 1
	fmt = (VPX_IMG_FMT_I444 if yuv444 else VPX_IMG_FMT_I420) | (VPX_IMG_FMT_HIGHBITDEPTH if bit_depth > 8 else 0)
	packets = []

	def drain():
		it = C.c_void_p(0)
		while True:
			pkt = lib.vpx_codec_get_cx_data(ctx, C.byref(it))
			if not pkt:
				return
			kind = C.c_int.from_address(pkt).value
			if kind == 0:  # VPX_CODEC_CX_FRAME_PKT
				buf = C.c_void_p.from_address(pkt + 8).value
				size = C.c_size_t.from_address(pkt + 16).value
				pts = C.c_int64.from_address(pkt + 24).value
				packets.append((pts, C.string_at(buf, size)))

	pts = 0
	for w, h, frames, raw in segments:
		if (w, h) != (w0, h0) or pts > 0:
			u32(CFG_W, w)
			u32(CFG_H, h)
			if lib.vpx_codec_enc_config_set(ctx, cfg):
				sys.exit(f"vpx_codec_enc_config_set {w}x{h} failed")
		data = open(raw, "rb").read()
		cw, chh = (w, h) if yuv444 else ((w + 1) // 2, (h + 1) // 2)
		frame_bytes = (w * h + 2 * cw * chh) * bps
		img = lib.vpx_img_alloc(None, fmt, w, h, 16)
		if bit_depth > 8:
			C.c_uint.from_address(img + 20).value = bit_depth
		for i in range(frames):
			frame = data[i * frame_bytes:(i + 1) * frame_bytes]
			at = 0
			for p in range(3):
				pw, ph = (w, h) if p == 0 else (cw, chh)
				base = C.c_void_p.from_address(img + IMG_PLANES + 8 * p).value
				stride = C.c_int.from_address(img + IMG_STRIDE + 4 * p).value
				for y in range(ph):
					row = frame[at:at + pw * bps]
					C.memmove(base + y * stride, row, len(row))
					at += pw * bps
			if lib.vpx_codec_encode(ctx, C.c_void_p(img), C.c_int64(pts), C.c_ulong(1), C.c_long(0), C.c_ulong(VPX_DL_GOOD_QUALITY)):
				sys.exit("vpx_codec_encode failed")
			pts += 1
			drain()
		lib.vpx_img_free(C.c_void_p(img))
	while True:
		before = len(packets)
		lib.vpx_codec_encode(ctx, None, C.c_int64(pts), C.c_ulong(1), C.c_long(0), C.c_ulong(VPX_DL_GOOD_QUALITY))
		drain()
		if len(packets) == before:
			break

	with open(out_path, "wb") as f:
		f.write(b"DKIF" + struct.pack("<HHIHHIIII", 0, 32, 0x30395056, w0, h0, 25, 1, len(packets), 0))
		for p, data in packets:
			f.write(struct.pack("<IQ", len(data), p) + data)


if __name__ == "__main__":
	main()
