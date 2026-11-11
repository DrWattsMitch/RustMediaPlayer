#!/usr/bin/env python3
"""Dumps the first audio stream's Opus packets for tools/opus_plc_ref.c.

Usage: opus_packets.py INPUT OUT.opkt   (needs ffprobe)

Output: "OPKT", channels, family, streams, coupled, mapping[channels], then
per packet a u32le length and the payload, in decode order.
"""
import json
import struct
import subprocess
import sys


def hexdump_bytes(dump):
    out = bytearray()
    for line in dump.strip().splitlines():
        # "00000000: 4f70 7573 ...  Opus..."
        body = line.split(":", 1)[1][:41]
        out += bytes.fromhex(body.replace(" ", ""))
    return bytes(out)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_streams",
        "-show_packets", "-show_data", "-of", "json", src]))
    head = hexdump_bytes(probe["streams"][0]["extradata"])
    channels, family = head[9], head[18]
    if family == 0:
        streams, coupled, mapping = 1, int(channels == 2), bytes(range(channels))
    else:
        streams, coupled, mapping = head[19], head[20], head[21:21 + channels]
    with open(dst, "wb") as f:
        f.write(b"OPKT" + bytes([channels, family, streams, coupled]) + mapping)
        for p in probe["packets"]:
            data = hexdump_bytes(p["data"])
            f.write(struct.pack("<I", len(data)) + data)


if __name__ == "__main__":
    main()
