#!/usr/bin/env python3
"""Builds a PAFF (field-picture) H.264 stream from two progressive x264
CAVLC streams of the top and bottom fields (same GOP structure).

mkpaff.py top.264 bottom.264 out.264 [--bff]

Every coded picture k of each input becomes field k of the output: pairs
(top_k, bottom_k) share frame_num; the second field of an IDR pair is
turned into a non-IDR I field. Only CAVLC slice data can be relabelled
this way (CABAC uses other contexts for field pictures). The result is a
valid stream; with --ref 1 it even decodes to the original video.
"""
import sys


def nals(data):
    i, n, out = 0, len(data), []
    starts = []
    while True:
        j = data.find(b"\x00\x00\x01", i)
        if j < 0:
            break
        starts.append(j + 3)
        i = j + 3
    for k, s in enumerate(starts):
        e = starts[k + 1] - 3 if k + 1 < len(starts) else n
        while e > s and data[e - 1] == 0:
            e -= 1
        out.append(data[s:e])
    return out


def unescape(b):
    out, z = bytearray(), 0
    for c in b:
        if z >= 2 and c == 3:
            z = 0
            continue
        out.append(c)
        z = z + 1 if c == 0 else 0
    return bytes(out)


def escape(b):
    out, z = bytearray(), 0
    for c in b:
        if z >= 2 and c <= 3:
            out.append(3)
            z = 0
        out.append(c)
        z = z + 1 if c == 0 else 0
    return bytes(out)


class R:
    def __init__(self, b):
        self.b, self.p = b, 0

    def u(self, n):
        v = 0
        for _ in range(n):
            v = (v << 1) | ((self.b[self.p >> 3] >> (7 - (self.p & 7))) & 1)
            self.p += 1
        return v

    def ue(self):
        z = 0
        while self.u(1) == 0:
            z += 1
        return (1 << z) - 1 + self.u(z)

    def se(self):
        v = self.ue()
        return (v + 1) // 2 if v & 1 else -(v // 2)

    def rest_bits(self):
        bits = []
        total = len(self.b) * 8
        while self.p < total:
            bits.append(self.u(1))
        return bits


class W:
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

    def bytes(self):
        bits = self.bits + [0] * (-len(self.bits) % 8)
        return bytes(int("".join(map(str, bits[i:i + 8])), 2) for i in range(0, len(bits), 8))


def copy_bits(r, w, n):
    for _ in range(n):
        w.bits.append(r.u(1))


def parse_sps(rb):
    r = R(rb)
    s = {}
    s["profile"] = r.u(8)
    r.u(16)
    r.ue()
    if s["profile"] in (100, 110, 122, 244, 44, 83, 86, 118, 128):
        cf = r.ue()
        if cf == 3:
            r.u(1)
        r.ue(); r.ue(); r.u(1)
        if r.u(1):
            raise SystemExit("scaling lists not handled")
    s["log2_fn"] = r.ue() + 4
    s["poc_type"] = r.ue()
    if s["poc_type"] == 0:
        s["log2_lsb"] = r.ue() + 4
    elif s["poc_type"] == 1:
        raise SystemExit("poc type 1 not handled")
    s["nref_pos"] = r.p
    r.ue(); r.u(1)
    r.ue(); r.ue()
    s["fmo_pos"] = r.p  # frame_mbs_only_flag
    return s


def rewrite_sps(rb):
    """frame_mbs_only_flag 1 -> 0, mb_adaptive_frame_field_flag 0."""
    s = parse_sps(rb)
    r = R(rb)
    w = W()
    copy_bits(r, w, s["nref_pos"])
    # One more reference frame, so that the previous same-parity field
    # outlives the current frame's first field.
    w.ue(r.ue() + 1)
    copy_bits(r, w, s["fmo_pos"] - r.p)
    assert r.u(1) == 1
    w.u(1, 0)
    w.u(1, 0)
    rest = r.rest_bits()
    # Strip the trailing rbsp_stop bit and zeros, then re-append.
    while rest and rest[-1] == 0:
        rest.pop()
    w.bits += rest
    return w.bytes(), s


def parse_pps(rb):
    r = R(rb)
    p = {}
    r.ue(); r.ue()
    p["cabac"] = r.u(1)
    p["bfpo"] = r.u(1)
    assert r.ue() == 0
    p["nref"] = [r.ue() + 1, r.ue() + 1]
    p["wp"] = r.u(1)
    p["wb"] = r.u(2)
    return p


def rewrite_slice(nal, sps, pps, bottom, idr_to_i, frame_num_out, lsb_out):
    hdr = nal[0]
    ref_idc = (hdr >> 5) & 3
    kind = hdr & 31
    rb = unescape(nal[1:])
    r = R(rb)
    w = W()
    w.ue(r.ue())  # first_mb
    st = r.ue()
    w.ue(st)
    st %= 5
    w.ue(r.ue())  # pps id
    r.u(sps["log2_fn"])
    w.u(sps["log2_fn"], frame_num_out)
    w.u(1, 1)  # field_pic_flag
    w.u(1, bottom)
    idr = kind == 5
    if idr:
        v = r.ue()
        if not idr_to_i:
            w.ue(v)
    if sps["poc_type"] == 0:
        r.u(sps["log2_lsb"])
        w.u(sps["log2_lsb"], lsb_out)
        if pps["bfpo"]:
            raise SystemExit("bottom_field_pic_order present")
    if st == 1:
        w.u(1, r.u(1))  # direct_spatial
    if st in (0, 1):
        o = r.u(1)
        w.u(1, o)
        if o:
            w.ue(r.ue())
            if st == 1:
                w.ue(r.ue())
    for _ in range({0: 1, 1: 2}.get(st, 0)):
        f = r.u(1)
        w.u(1, f)
        if f:
            while True:
                idc = r.ue()
                w.ue(idc)
                if idc == 3:
                    break
                w.ue(r.ue())
    if (pps["wp"] and st == 0) or (pps["wb"] == 1 and st == 1):
        raise SystemExit("explicit weighted prediction not handled")
    if ref_idc:
        if idr:
            nop = r.u(1)
            ltr = r.u(1)
            if idr_to_i:
                w.u(1, 0)  # adaptive_ref_pic_marking_mode_flag
            else:
                w.u(1, nop)
                w.u(1, ltr)
        else:
            a = r.u(1)
            w.u(1, a)
            if a:
                while True:
                    op = r.ue()
                    w.ue(op)
                    if op == 0:
                        break
                    if op in (1, 3):
                        w.ue(r.ue())
                    if op == 2:
                        w.ue(r.ue())
                    if op in (3, 6):
                        w.ue(r.ue())
                    if op == 4:
                        w.ue(r.ue())
    rest = r.rest_bits()
    while rest and rest[-1] == 0:
        rest.pop()
    w.bits += rest
    new_kind = 1 if (idr and idr_to_i) else kind
    return bytes([(hdr & 0xE0) | new_kind]) + escape(w.bytes())


def pictures(stream):
    """Groups slice NALs into pictures (first_mb_in_slice == 0 starts one)."""
    params, pics = [], []
    for n in stream:
        kind = n[0] & 31
        if kind in (7, 8):
            params.append(n)
        elif kind in (1, 5):
            r = R(unescape(n[1:]))
            if r.ue() == 0:
                pics.append([])
            pics[-1].append(n)
    return params, pics


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    bff = "--bff" in sys.argv
    top_s, bot_s = nals(open(args[0], "rb").read()), nals(open(args[1], "rb").read())
    params, top = pictures(top_s)
    _, bot = pictures(bot_s)
    assert len(top) == len(bot), (len(top), len(bot))
    sps_nal = next(n for n in params if n[0] & 31 == 7)
    pps_nal = next(n for n in params if n[0] & 31 == 8)
    sps_rb, sps = rewrite_sps(unescape(sps_nal[1:]))
    pps = parse_pps(unescape(pps_nal[1:]))
    assert not pps["cabac"], "CAVLC only"
    out = bytearray()
    sc = b"\x00\x00\x00\x01"
    out += sc + bytes([sps_nal[0]]) + escape(sps_rb)
    out += sc + pps_nal
    for k in range(len(top)):
        first, second = (bot[k], top[k]) if bff else (top[k], bot[k])
        for field, (slices, bottom) in enumerate(((first, int(bff)), (second, 1 - int(bff)))):
            for n in slices:
                r = R(unescape(n[1:]))
                r.ue(); r.ue(); r.ue()
                fn = r.u(sps["log2_fn"])
                if n[0] & 31 == 5:
                    r.ue()
                lsb = r.u(sps["log2_lsb"]) if sps["poc_type"] == 0 else 0
                lsb_out = (lsb + field) % (1 << sps.get("log2_lsb", 4))
                out += sc + rewrite_slice(n, sps, pps, bottom, field == 1, fn, lsb_out)
    open(args[2], "wb").write(out)


if __name__ == "__main__":
    main()
