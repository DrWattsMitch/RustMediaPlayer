#!/usr/bin/env python3
"""Synthetic HEVC range-extension streams for differential testing.

x265 codes 4:0:0, 4:2:2 and 4:4:4 but never uses the range extension coding
tools (cross-component prediction, RDPCM, large and rotated transform skip,
persistent Rice adaptation, CU chroma QP offsets...), and the JCT-VC
conformance streams are not always at hand. This writes streams that do:
valid VPS/SPS/PPS and slice headers with those tools switched on, followed
by slice data made of random bytes. Every byte string is *some* CABAC
bitstream, so two decoders that implement the standard the same way must
produce the same pictures from it. Bytes are drawn with few one bits so
that levels stay small (random bypass bins would otherwise give huge
coefficients and Rice parameters, where FFmpeg's error handling takes over).

Usage: hevc_rext_streams.py OUT.hevc SEED [key=value ...]
Keys (defaults in CONFIG): chroma, depth, w, h, ctb, frames, and the
0/1 tool flags. Prints the configuration used.
"""

import os
import random
import sys

CONFIG = dict(
	chroma=3, depth=8, w=64, h=64, ctb=16, min_cb=8, min_tb=4, max_tb=16,
	depth_intra=2, depth_inter=2, frames=4, density=0.12,
	# SPS range extension.
	ts_rotation=1, ts_context=1, implicit_rdpcm=1, explicit_rdpcm=1,
	intra_smoothing_disabled=0, high_precision=0, persistent_rice=1,
	# SPS / PPS tools.
	scaling=0, amp=1, sao=1, tmvp=1, strong=1, sign_hiding=1, constrained_intra=0,
	transform_skip=1, log2_max_ts=4, cu_qp_delta=0, cb_qp_offset=2, cr_qp_offset=-3,
	weighted=1, bypass=0, deblock=1, ccp=1, chroma_qp_list=1, sao_scale_luma=0,
	sao_scale_chroma=0, init_qp=30,
	# Slice data bytes per picture (0: enough for any picture).
	bytes=0,
)


class Bits:
	def __init__(self):
		self.bits = []

	def u(self, n, v):
		for i in range(n - 1, -1, -1):
			self.bits.append((v >> i) & 1)

	def ue(self, v):
		v += 1
		n = v.bit_length()
		self.u(n - 1, 0)
		self.u(n, v)

	def se(self, v):
		self.ue(2 * v - 1 if v > 0 else -2 * v)

	def trailing(self):
		self.bits.append(1)
		while len(self.bits) % 8:
			self.bits.append(0)

	def align_ones_zero(self):
		# byte_alignment(): alignment_bit_equal_to_one, then zeros.
		self.trailing()

	def bytes(self):
		assert len(self.bits) % 8 == 0
		return bytes(int(''.join(map(str, self.bits[i:i + 8])), 2) for i in range(0, len(self.bits), 8))


def nal(t, payload):
	"""Annex B NAL unit with emulation prevention."""
	out = bytearray([0, 0, 0, 1, t << 1, 1])
	zeros = 0
	for b in payload:
		if zeros >= 2 and b <= 3:
			out.append(3)
			zeros = 0
		out.append(b)
		zeros = zeros + 1 if b == 0 else 0
	return bytes(out)


def profile_tier_level(b, c):
	b.u(2, 0)
	b.u(1, 0)
	b.u(5, 4)  # format range extensions
	b.u(32, 1 << (31 - 4))
	b.u(4, 0b1001)  # progressive, frame only
	b.u(43, 0)
	b.u(1, 0)
	b.u(8, 120)


def vps(c):
	b = Bits()
	b.u(4, 0)
	b.u(1, 1)
	b.u(1, 1)
	b.u(6, 0)
	b.u(3, 0)
	b.u(1, 1)
	b.u(16, 0xFFFF)
	profile_tier_level(b, c)
	b.u(1, 1)
	b.ue(3)
	b.ue(0)
	b.ue(0)
	b.u(6, 0)
	b.ue(0)
	b.u(1, 0)
	b.u(1, 0)
	b.trailing()
	return nal(32, b.bytes())


def log2(v):
	return v.bit_length() - 1


def sps(c):
	b = Bits()
	b.u(4, 0)
	b.u(3, 0)
	b.u(1, 1)
	profile_tier_level(b, c)
	b.ue(0)
	b.ue(c['chroma'])
	if c['chroma'] == 3:
		b.u(1, 0)
	b.ue(c['w'])
	b.ue(c['h'])
	b.u(1, 0)
	b.ue(c['depth'] - 8)
	b.ue(c['depth'] - 8)
	b.ue(4)  # log2_max_poc_lsb 8
	b.u(1, 1)
	b.ue(3)
	b.ue(0)
	b.ue(0)
	b.ue(log2(c['min_cb']) - 3)
	b.ue(log2(c['ctb']) - log2(c['min_cb']))
	b.ue(log2(c['min_tb']) - 2)
	b.ue(log2(c['max_tb']) - log2(c['min_tb']))
	b.ue(c['depth_inter'])
	b.ue(c['depth_intra'])
	b.u(1, c['scaling'])
	if c['scaling']:
		b.u(1, 0)  # default lists
	b.u(1, c['amp'])
	b.u(1, c['sao'])
	b.u(1, 0)  # pcm
	b.ue(1)  # one short-term RPS: the previous picture
	b.ue(1)
	b.ue(0)
	b.ue(0)
	b.u(1, 1)
	b.u(1, 0)  # long-term refs
	b.u(1, c['tmvp'])
	b.u(1, c['strong'])
	b.u(1, 0)  # vui
	b.u(1, 1)  # sps_extension_present_flag
	b.u(1, 1)
	b.u(3, 0)
	b.u(4, 0)
	for k in ['ts_rotation', 'ts_context', 'implicit_rdpcm', 'explicit_rdpcm']:
		b.u(1, c[k])
	b.u(1, 0)  # extended_precision_processing_flag
	b.u(1, c['intra_smoothing_disabled'])
	b.u(1, c['high_precision'])
	b.u(1, c['persistent_rice'])
	b.u(1, 0)  # cabac_bypass_alignment_enabled_flag
	b.trailing()
	return nal(33, b.bytes())


def pps(c, rng):
	b = Bits()
	b.ue(0)
	b.ue(0)
	b.u(1, 0)
	b.u(1, 0)
	b.u(3, 0)
	b.u(1, c['sign_hiding'])
	b.u(1, 0)
	b.ue(0)
	b.ue(0)
	b.se(c['init_qp'] - 26)
	b.u(1, c['constrained_intra'])
	b.u(1, c['transform_skip'])
	b.u(1, c['cu_qp_delta'])
	if c['cu_qp_delta']:
		b.ue(1)
	b.se(c['cb_qp_offset'])
	b.se(c['cr_qp_offset'])
	b.u(1, 0)
	b.u(1, c['weighted'])
	b.u(1, 0)
	b.u(1, c['bypass'])
	b.u(1, 0)  # tiles
	b.u(1, 0)  # wpp
	b.u(1, 1)  # loop filter across slices
	b.u(1, 1)  # deblocking_filter_control_present_flag
	b.u(1, 0)
	b.u(1, 0 if c['deblock'] else 1)
	if c['deblock']:
		b.se(rng.randint(-3, 3))
		b.se(rng.randint(-3, 3))
	b.u(1, 0)
	b.u(1, 0)
	b.ue(0)
	b.u(1, 0)
	b.u(1, 1)  # pps_extension_present_flag
	b.u(1, 1)
	b.u(3, 0)
	b.u(4, 0)
	if c['transform_skip']:
		b.ue(c['log2_max_ts'] - 2)
	b.u(1, c['ccp'])
	b.u(1, c['chroma_qp_list'])
	if c['chroma_qp_list']:
		b.ue(1)
		# Six entries: FFmpeg reads cu_chroma_qp_offset_idx with cMax 5
		# whatever the list length.
		b.ue(5)
		for _ in range(6):
			b.se(rng.randint(-12, 12))
			b.se(rng.randint(-12, 12))
	b.ue(c['sao_scale_luma'])
	b.ue(c['sao_scale_chroma'])
	b.trailing()
	return nal(34, b.bytes())


def slice_nal(c, rng, poc):
	b = Bits()
	idr = poc == 0
	b.u(1, 1)
	if idr:
		b.u(1, 0)
	b.ue(0)
	b.ue(2 if idr else 1)
	if not idr:
		b.u(8, poc & 255)
		b.u(1, 1)  # RPS from the SPS
		if c['tmvp']:
			b.u(1, 1)
	if c['sao']:
		b.u(1, 1)
		if c['chroma']:
			b.u(1, 1)
	if not idr:
		b.u(1, 0)
		if c['weighted']:
			hp = c['high_precision']
			half = (1 << (c['depth'] - 1)) if hp else 128
			ld = rng.randint(0, 7)
			b.ue(ld)
			cd = ld
			if c['chroma']:
				cd = rng.randint(0, 7)
				b.se(cd - ld)
			lf = rng.randint(0, 1)
			cf = rng.randint(0, 1) if c['chroma'] else 0
			b.u(1, lf)
			if c['chroma']:
				b.u(1, cf)
			if lf:
				dw, off = rng.randint(-20, 20), rng.randint(-half // 4, half // 4)
				b.se(dw)
				b.se(off)
				if os.environ.get('WP_DEBUG'):
					print(f'poc {poc} luma weight {(1 << ld) + dw} offset {off}', file=sys.stderr)
			if os.environ.get('WP_DEBUG'):
				print(f'poc {poc} denom {ld}/{cd} luma flag {lf}', file=sys.stderr)
			if cf:
				for _ in range(2):
					b.se(rng.randint(-20, 20))
					b.se(rng.randint(-half, half - 1))
		b.ue(rng.randint(0, 4))
	b.se(rng.randint(-4, 4))
	if c['chroma_qp_list']:
		b.u(1, 1)
	if c['sao'] or c['deblock']:
		b.u(1, 1)
	b.align_ones_zero()
	head = b.bytes()
	n = c['bytes'] or c['w'] * c['h'] * 2 + 256
	p = c['density']
	data = bytearray(sum((rng.random() < p) << i for i in range(8)) for _ in range(n))
	# The first 9 bits (ivlOffset) must not be 510 or 511 (9.3.2.5).
	data[0] &= 0x7F
	data = bytes(data)
	return nal(19 if idr else 1, head + data)


def main():
	out, seed = sys.argv[1], int(sys.argv[2])
	c = dict(CONFIG)
	for kv in sys.argv[3:]:
		k, v = kv.split('=')
		c[k] = type(CONFIG[k])(v)
	if c['chroma'] != 3:
		c['ccp'] = 0
	rng = random.Random(seed)
	s = vps(c) + sps(c) + pps(c, rng)
	for poc in range(c['frames']):
		s += slice_nal(c, rng, poc)
	open(out, 'wb').write(s)
	print(' '.join(f'{k}={v}' for k, v in c.items()))


if __name__ == '__main__':
	main()
