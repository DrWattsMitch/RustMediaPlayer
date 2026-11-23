#!/usr/bin/env python3
"""Interleaves MBAFF frames (from an x264 --interlaced CAVLC stream) with
field pairs (spliced from top/bottom field streams with the same GOP) into
one stream using the MBAFF stream's SPS/PPS: pictures whose index k has
pattern[k % len] == 'f' are coded as two field pictures.

mkmix.py mbaff.264 top.264 bottom.264 out.264 PATTERN   (e.g. FFfF)
"""
import sys
from mkpaff import R, W, nals, unescape, escape, parse_sps, parse_pps, pictures


def slice_info(n, sps):
    r = R(unescape(n[1:]))
    r.ue(); st = r.ue() % 5; r.ue()
    fn = r.u(sps["log2_fn"])
    return st, fn


def field_slice(nal, sps_in, pps_in, sps_out, bottom, idr_to_i, frame_num, lsb):
    hdr = nal[0]
    ref_idc = (hdr >> 5) & 3
    kind = hdr & 31
    r = R(unescape(nal[1:]))
    w = W()
    w.ue(r.ue())
    st_raw = r.ue(); w.ue(st_raw); st = st_raw % 5
    r.ue(); w.ue(0)  # pps id -> the MBAFF stream's PPS
    r.u(sps_in["log2_fn"]); w.u(sps_out["log2_fn"], frame_num)
    w.u(1, 1); w.u(1, bottom)
    idr = kind == 5
    if idr:
        v = r.ue()
        if not idr_to_i:
            w.ue(v)
    if sps_in["poc_type"] == 0:
        r.u(sps_in["log2_lsb"])
    w.u(sps_out["log2_lsb"], lsb)
    if st == 1:
        w.u(1, r.u(1))
    if st in (0, 1):
        counts = list(pps_in["nref"])
        if r.u(1):
            counts[0] = r.ue() + 1
            if st == 1:
                counts[1] = r.ue() + 1
        # Always explicit: the output PPS defaults may differ.
        w.u(1, 1)
        w.ue(counts[0] - 1)
        if st == 1:
            w.ue(counts[1] - 1)
    for _ in range({0: 1, 1: 2}.get(st, 0)):
        f = r.u(1); w.u(1, f)
        if f:
            while True:
                idc = r.ue(); w.ue(idc)
                if idc == 3:
                    break
                w.ue(r.ue())
    if ref_idc:
        if idr:
            nop, ltr = r.u(1), r.u(1)
            if idr_to_i:
                w.u(1, 0)
            else:
                w.u(1, nop); w.u(1, ltr)
        else:
            a = r.u(1); w.u(1, a)
            assert not a, "mmco not handled"
    # slice_qp_delta: keep the same QP under the other PPS.
    dqp = r.se()
    w.se(dqp + pps_in["init_qp"] - pps_out_init_qp)
    rest = r.rest_bits()
    while rest and rest[-1] == 0:
        rest.pop()
    w.bits += rest
    new_kind = 1 if (idr and idr_to_i) else kind
    return bytes([(hdr & 0xE0) | new_kind]) + escape(w.bytes())


def parse_pps_full(rb):
    r = R(rb)
    p = {}
    r.ue(); r.ue()
    p["cabac"] = r.u(1); p["bfpo"] = r.u(1)
    assert r.ue() == 0
    p["nref"] = [r.ue() + 1, r.ue() + 1]
    p["wp"] = r.u(1); p["wb"] = r.u(2)
    p["init_qp"] = 26 + r.se()
    return p


def lsb_of(n, sps, pps):
    r = R(unescape(n[1:]))
    r.ue(); r.ue(); r.ue(); r.u(sps["log2_fn"])
    # frame slices of the MBAFF stream: field_pic_flag
    if r.u(1):
        r.u(1)
    if n[0] & 31 == 5:
        r.ue()
    return r.u(sps["log2_lsb"])


def main():
    global pps_out_init_qp
    a_s, t_s, b_s = (nals(open(p, "rb").read()) for p in sys.argv[1:4])
    pattern = sys.argv[5]
    a_params, a_pics = pictures(a_s)
    t_params, t_pics = pictures(t_s)
    _, b_pics = pictures(b_s)
    assert len(a_pics) == len(t_pics) == len(b_pics)
    a_sps_nal = next(n for n in a_params if n[0] & 31 == 7)
    a_pps_nal = next(n for n in a_params if n[0] & 31 == 8)
    t_sps_nal = next(n for n in t_params if n[0] & 31 == 7)
    t_pps_nal = next(n for n in t_params if n[0] & 31 == 8)
    sps_a = parse_sps(unescape(a_sps_nal[1:]))
    sps_t = parse_sps(unescape(t_sps_nal[1:]))
    pps_a = parse_pps_full(unescape(a_pps_nal[1:]))
    pps_t = parse_pps_full(unescape(t_pps_nal[1:]))
    pps_out_init_qp = pps_a["init_qp"]
    out = bytearray()
    sc = b"\x00\x00\x00\x01"
    out += sc + a_sps_nal + sc + a_pps_nal
    fields = 0
    for k in range(len(a_pics)):
        st_a, fn_a = slice_info(a_pics[k][0], sps_a)
        st_t, fn_t = slice_info(t_pics[k][0], sps_t)
        assert st_a == st_t, (k, st_a, st_t)
        lsb = lsb_of(a_pics[k][0], sps_a, pps_a)
        if pattern[k % len(pattern)] == "f":
            fields += 1
            for field, (pic, bottom) in enumerate(((t_pics[k], 0), (b_pics[k], 1))):
                for n in pic:
                    l = (lsb + field) % (1 << sps_a["log2_lsb"])
                    out += sc + field_slice(n, sps_t, pps_t, sps_a, bottom, field == 1, fn_a, l)
        else:
            for n in a_pics[k]:
                out += sc + n
    open(sys.argv[4], "wb").write(out)
    print(f"{len(a_pics)} pictures, {fields} as field pairs")


main()
