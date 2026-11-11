/* Reference decodes for RMP's Opus packet-loss tests (libopus 1.4).
 *
 * Usage: opus_plc_ref PACKETS PATTERN plc|fec OUT.s16le
 *
 * PACKETS is the framed packet file written by tools/opus_packets.py:
 *   "OPKT", channels, family, streams, coupled, mapping[channels],
 *   then per packet: u32le length, payload.
 * PATTERN has one character per packet: '.' decodes it, 'x' drops it.
 * Each run of dropped packets followed by a received one is concealed the
 * way RMP's decoder conceals a timestamp gap: PLC calls (data = NULL) of at
 * most 120 ms (libopus' limit per call) from the start of the gap, the
 * last one covering the gap's final 120 ms or less; in "fec" mode that last
 * call uses the received packet's in-band FEC instead
 * (opus_multistream_decode_float(..., decode_fec = 1)). Dropped packets at
 * the end are not concealed.
 *
 * Output is interleaved s16le through libopus' FLOAT2INT16 rounding, with
 * the pre-skip kept (like the other Opus references).
 *
 * Build: cc -O2 -I opus/include tools/opus_plc_ref.c opus/.libs/libopus.a -lm
 */
#include <opus_multistream.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_PACKETS 20000
#define MAX_FRAME 5760

static unsigned char *packets[MAX_PACKETS];
static int sizes[MAX_PACKETS];

/* Vorbis -> FFmpeg channel order for mapping family 1, as RMP outputs. */
static const int vorbis_order[8][8] = {
	{0}, {0, 1}, {0, 2, 1}, {0, 1, 2, 3}, {0, 2, 1, 3, 4},
	{0, 2, 1, 5, 3, 4}, {0, 2, 1, 6, 5, 3, 4}, {0, 2, 1, 7, 5, 6, 3, 4},
};
static int family;

static void write_pcm(FILE *out, const float *pcm, int samples, int channels) {
	for (int i = 0; i < samples * channels; i++) {
		int c = i % channels;
		int src = family == 1 && channels <= 8 ? vorbis_order[channels - 1][c] : c;
		float x = pcm[i - c + src] * 32768.f;
		x = x > 32767.f ? 32767.f : (x < -32768.f ? -32768.f : x);
		int16_t s = (int16_t)lrintf(x);
		fwrite(&s, 2, 1, out);
	}
}

int main(int argc, char **argv) {
	if (argc != 5) {
		fprintf(stderr, "usage: %s PACKETS PATTERN plc|fec OUT\n", argv[0]);
		return 2;
	}
	FILE *in = fopen(argv[1], "rb");
	const char *pattern = argv[2];
	int fec = strcmp(argv[3], "fec") == 0;
	FILE *out = fopen(argv[4], "wb");
	if (!in || !out) return 1;
	unsigned char head[8], mapping[255];
	if (fread(head, 1, 8, in) != 8 || memcmp(head, "OPKT", 4)) return 1;
	int channels = head[4], streams = head[6], coupled = head[7];
	family = head[5];
	if (fread(mapping, 1, channels, in) != (size_t)channels) return 1;
	int count = 0;
	unsigned char len4[4];
	while (count < MAX_PACKETS && fread(len4, 1, 4, in) == 4) {
		sizes[count] = len4[0] | len4[1] << 8 | len4[2] << 16 | len4[3] << 24;
		packets[count] = malloc(sizes[count] ? sizes[count] : 1);
		if (fread(packets[count], 1, sizes[count], in) != (size_t)sizes[count]) return 1;
		count++;
	}
	int err;
	OpusMSDecoder *dec =
		opus_multistream_decoder_create(48000, channels, streams, coupled, mapping, &err);
	if (err != OPUS_OK) return 1;
	float *pcm = malloc(sizeof(float) * MAX_FRAME * 64 * channels);
	int gap = 0;
	for (int i = 0; i < count; i++) {
		int n = opus_packet_get_nb_samples(packets[i], sizes[i], 48000);
		if (i < (int)strlen(pattern) && pattern[i] == 'x') {
			gap += n;
			continue;
		}
		while (gap > 0) {
			int last = gap <= MAX_FRAME;
			int chunk = last ? gap : (gap - MAX_FRAME < MAX_FRAME ? gap - MAX_FRAME : MAX_FRAME);
			int got = fec && last
				? opus_multistream_decode_float(dec, packets[i], sizes[i], pcm, chunk, 1)
				: opus_multistream_decode_float(dec, NULL, 0, pcm, chunk, 0);
			if (got != chunk) {
				fprintf(stderr, "concealment of %d failed: %d\n", chunk, got);
				return 1;
			}
			write_pcm(out, pcm, got, channels);
			gap -= got;
		}
		int got = opus_multistream_decode_float(dec, packets[i], sizes[i], pcm, MAX_FRAME, 0);
		if (got < 0) {
			fprintf(stderr, "packet %d failed: %d\n", i, got);
			return 1;
		}
		write_pcm(out, pcm, got, channels);
	}
	fclose(out);
	return 0;
}
