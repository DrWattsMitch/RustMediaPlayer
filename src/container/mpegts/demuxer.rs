//! MPEG transport stream demuxer.
//!
//! Opening a file reads packets until every program in the PAT has its PMT
//! and every elementary stream has shown its codec parameters (or the
//! probe budget runs out), queueing the packets it produced; streams that
//! never showed parameters are dropped, as FFmpeg drops what
//! `avformat_find_stream_info` can't identify. The duration comes from the
//! timestamps at the end of the file.
//!
//! Reading assembles PES packets per PID and turns them into packets:
//! one per audio frame (timestamps interpolated within a PES), one per
//! video access unit (keyframes from the bitstream, durations from the
//! next decode time). Timestamps keep the file's 90 kHz values, unwrapped
//! past 33 bits, as FFmpeg reports them.
//!
//! Seeking bisects the file on the timestamps of a reference stream
//! (the first video stream, else the first stream), then backs up to the
//! last keyframe at or before the target.
//!
//! FFmpeg: `libavformat/mpegts.c`.

use std::collections::VecDeque;

use super::es::{self, Framing, TIME_BASE};
use super::packet::{self, Layout, PID_NULL, PID_PAT, PID_SDT, TS_SIZE, Unwrap};
use super::psi::{self, SectionBuffer};
use crate::core::{
	Demuxer, Metadata, Packet, RawValue, Stream, StreamId, StreamKind, StreamSet, Time,
};
use crate::io::{BufferedReader, Io};
use crate::message::{Message, Result};

/// Bytes read while looking for stream parameters (FFmpeg's default
/// `probesize`).
const PROBE_BYTES: u64 = 5_000_000;
/// Bytes from the end scanned for the last timestamps.
const TAIL_BYTES: u64 = 2 << 20;
/// Largest PES kept in memory; a bigger one is cut (and its data lost)
/// rather than growing without bound on a broken stream.
const MAX_PES: usize = 16 << 20;
/// Bytes read per step while bisecting.
const SEEK_WINDOW: u64 = 256 << 10;

/// One elementary stream from a PMT.
struct Es {
	pid: u16,
	codec: &'static str,
	kind: StreamKind,
	framing: Framing,
	language: Option<String>,
	/// Opus `channel_config_code` from the PMT.
	opus_config: Option<u8>,
	/// A SAMPLE-AES stream type: samples are partly encrypted.
	sample_aes: bool,
	/// Output stream index, once known (after probing).
	stream: Option<usize>,
	/// Parameters seen in the data: the stream can be created.
	ready: Option<Stream>,
	pes: Vec<u8>,
	/// A PES is being assembled (its start was seen).
	in_pes: bool,
	random_access: bool,
	unwrap: Unwrap,
	audio: AudioState,
	/// Video packet waiting for the next decode time (its duration).
	held: Option<Packet>,
	last_duration: Option<u64>,
}

#[derive(Default)]
struct AudioState {
	buf: Vec<u8>,
	/// Absolute offset of `buf[0]` in this stream's payload bytes.
	start: u64,
	/// PES timestamps by the payload offset where their PES began.
	stamps: VecDeque<(u64, u64)>,
	next_pts: Option<u64>,
}

impl Es {
	fn reset(&mut self) {
		self.pes.clear();
		self.in_pes = false;
		self.audio = AudioState::default();
		self.held = None;
	}
}

/// Everything that turns transport packets into elementary packets.
struct Core {
	pat: SectionBuffer,
	/// PMT PIDs from the PAT, with their section buffers.
	pmts: Vec<(u16, u16, SectionBuffer)>,
	pmts_seen: Vec<u16>,
	sdt: SectionBuffer,
	es: Vec<Es>,
	/// Output packets, tagged with their `es` index.
	out: VecDeque<(usize, Packet)>,
	metadata: Metadata,
	/// New streams are accepted only while probing.
	probing: bool,
}

impl Core {
	fn new() -> Self {
		Self {
			pat: SectionBuffer::default(),
			pmts: Vec::new(),
			pmts_seen: Vec::new(),
			sdt: SectionBuffer::default(),
			es: Vec::new(),
			out: VecDeque::new(),
			metadata: Metadata::default(),
			probing: true,
		}
	}

	/// Copy of the stream table, without any data in flight.
	fn fork(&self) -> Self {
		let mut core = Self::new();
		core.probing = false;
		core.es = self
			.es
			.iter()
			.map(|e| Es {
				pid: e.pid,
				codec: e.codec,
				kind: e.kind,
				framing: e.framing,
				language: None,
				opus_config: e.opus_config,
				sample_aes: e.sample_aes,
				stream: e.stream,
				ready: None,
				pes: Vec::new(),
				in_pes: false,
				random_access: false,
				unwrap: e.unwrap,
				audio: AudioState::default(),
				held: None,
				last_duration: None,
			})
			.collect();
		core
	}

	fn programs_known(&self) -> bool {
		!self.pmts.is_empty() && self.pmts.iter().all(|(_, pid, _)| self.pmts_seen.contains(pid))
	}

	fn all_ready(&self) -> bool {
		self.programs_known() && self.es.iter().all(|e| e.ready.is_some())
	}

	/// Handles one 188-byte transport packet.
	fn packet(&mut self, p: &[u8]) {
		let Some(h) = packet::parse(p) else { return };
		if !h.has_payload || h.pid == PID_NULL {
			return;
		}
		let payload = &p[h.payload_at..TS_SIZE];
		if h.pid == PID_PAT {
			for section in self.pat.push(payload, h.unit_start) {
				self.on_pat(&section);
			}
			return;
		}
		if h.pid == PID_SDT {
			for section in self.sdt.push(payload, h.unit_start) {
				self.on_sdt(&section);
			}
			return;
		}
		if let Some(i) = self.pmts.iter().position(|(_, pid, _)| *pid == h.pid) {
			for section in self.pmts[i].2.push(payload, h.unit_start) {
				self.on_pmt(h.pid, &section);
			}
			return;
		}
		if let Some(i) = self.es.iter().position(|e| e.pid == h.pid) {
			self.es_payload(i, payload, h.unit_start, h.random_access);
		}
	}

	fn on_pat(&mut self, data: &[u8]) {
		let Some(section) = psi::Section::parse(data) else { return };
		if section.table_id != 0 || !section.current {
			return;
		}
		for (program, pid) in psi::parse_pat(&section) {
			if !self.pmts.iter().any(|(_, p, _)| *p == pid) {
				self.pmts.push((program, pid, SectionBuffer::default()));
			}
		}
	}

	fn on_pmt(&mut self, pid: u16, data: &[u8]) {
		let Some(section) = psi::Section::parse(data) else { return };
		if section.table_id != 2 || !section.current {
			return;
		}
		let Some(pmt) = psi::parse_pmt(&section) else { return };
		if !self.pmts_seen.contains(&pid) {
			self.pmts_seen.push(pid);
		}
		if !self.probing {
			return;
		}
		for info in &pmt.streams {
			if self.es.iter().any(|e| e.pid == info.pid) {
				continue;
			}
			let Some((codec, kind, framing)) = es::classify(info, &pmt.descriptors) else { continue };
			self.es.push(Es {
				pid: info.pid,
				codec,
				kind,
				framing,
				language: es::language(info),
				opus_config: es::opus_channel_config(info),
				sample_aes: es::is_sample_aes(info.stream_type),
				stream: None,
				ready: None,
				pes: Vec::new(),
				in_pes: false,
				random_access: false,
				unwrap: Unwrap::default(),
				audio: AudioState::default(),
				held: None,
				last_duration: None,
			});
		}
	}

	fn on_sdt(&mut self, data: &[u8]) {
		let Some(section) = psi::Section::parse(data) else { return };
		// Actual transport stream only (0x46 describes other multiplexes).
		if section.table_id != 0x42 || !section.current {
			return;
		}
		if let Some(service) = psi::parse_sdt(&section).into_iter().next() {
			let raw = &mut self.metadata.raw;
			if !service.name.is_empty() {
				raw.insert("service_name".into(), RawValue::String(service.name));
			}
			if !service.provider.is_empty() {
				raw.insert("service_provider".into(), RawValue::String(service.provider));
			}
		}
	}

	fn es_payload(&mut self, i: usize, payload: &[u8], unit_start: bool, random_access: bool) {
		let e = &mut self.es[i];
		if unit_start {
			if e.in_pes {
				self.finish_pes(i);
			}
			let e = &mut self.es[i];
			e.in_pes = true;
			e.random_access = random_access;
			e.pes.clear();
		}
		let e = &mut self.es[i];
		if !e.in_pes {
			return; // the start of this PES was before our first packet
		}
		if e.pes.len() + payload.len() > MAX_PES {
			self.finish_pes(i);
			return;
		}
		e.pes.extend_from_slice(payload);
		// A bounded PES is complete as soon as its last byte arrives.
		if packet::parse_pes(&e.pes).and_then(|h| h.total_len).is_some_and(|total| e.pes.len() >= total)
		{
			self.finish_pes(i);
		}
	}

	/// Turns the assembled PES of stream `i` into packets.
	fn finish_pes(&mut self, i: usize) {
		let e = &mut self.es[i];
		e.in_pes = false;
		let pes = std::mem::take(&mut e.pes);
		let Some(h) = packet::parse_pes(&pes) else { return };
		let end = h.total_len.map_or(pes.len(), |t| t.min(pes.len()));
		let Some(data) = pes.get(h.payload_at..end) else { return };
		if data.is_empty() {
			return;
		}
		let pts = h.pts.map(|t| e.unwrap.apply(t));
		let dts = h.dts.map(|t| e.unwrap.apply(t)).or(pts);
		if e.framing == Framing::Unknown {
			match es::probe_audio(data) {
				Some((codec, framing)) if self.probing => (e.codec, e.framing) = (codec, framing),
				_ => return,
			}
		}
		match e.framing {
			Framing::Unknown => {}
			Framing::Adts | Framing::MpegAudio | Framing::Ac3 => self.audio_pes(i, data, pts),
			Framing::Opus => self.opus_pes(i, data, pts),
			Framing::H264 | Framing::Hevc | Framing::Mpeg2Video | Framing::Video => {
				self.video_pes(i, data.to_vec(), pts, dts)
			}
			Framing::Whole => {
				let e = &mut self.es[i];
				if e.ready.is_none() {
					e.ready = Some(es::make_stream(0, e.codec, e.kind, None, None));
				}
				let mut packet = Packet::new(StreamId(0), data.to_vec());
				packet.pts = pts.map(|t| Time::new(t, TIME_BASE));
				packet.dts = dts.map(|t| Time::new(t, TIME_BASE));
				self.out.push_back((i, packet));
			}
		}
	}

	fn video_pes(&mut self, i: usize, mut data: Vec<u8>, pts: Option<u64>, dts: Option<u64>) {
		let e = &mut self.es[i];
		// FFmpeg's HEVC parser splits access units at the 3-byte start code,
		// so the zero byte of a 4-byte one ends the previous unit. Done the
		// same way, packets match FFmpeg's byte for byte; both are valid
		// Annex B.
		if e.framing == Framing::Hevc
			&& data.starts_with(&[0, 0, 0, 1])
			&& let Some(held) = e.held.as_mut()
		{
			data.remove(0);
			held.data.push(0);
		}
		let info = es::inspect_video(e.framing, &data);
		if e.ready.is_none() {
			let has_params = match e.framing {
				Framing::H264 | Framing::Hevc => !info.parameter_sets.is_empty() && info.width > 0,
				Framing::Mpeg2Video => info.width > 0,
				_ => true,
			};
			if has_params {
				e.ready = Some(es::make_stream(0, e.codec, e.kind, None, Some(&info)));
			}
		}
		let mut packet = Packet::new(StreamId(0), data);
		packet.pts = pts.map(|t| Time::new(t, TIME_BASE));
		packet.dts = dts.map(|t| Time::new(t, TIME_BASE));
		packet.keyframe = info.keyframe || (e.framing == Framing::Video && e.random_access);
		// Release the previous access unit with its duration.
		if let Some(mut held) = e.held.take() {
			let step = match (held.dts, packet.dts) {
				(Some(a), Some(b)) => b.ticks().checked_sub(a.ticks()).filter(|&d| d > 0),
				_ => None,
			};
			if step.is_some() {
				e.last_duration = step;
			}
			held.duration = step.or(e.last_duration).map(|d| Time::new(d, TIME_BASE));
			self.out.push_back((i, held));
		}
		self.es[i].held = Some(packet);
	}

	fn audio_pes(&mut self, i: usize, data: &[u8], pts: Option<u64>) {
		let e = &mut self.es[i];
		let a = &mut e.audio;
		let at = a.start + a.buf.len() as u64;
		if let Some(pts) = pts {
			a.stamps.push_back((at, pts));
		}
		a.buf.extend_from_slice(data);
		let mut used = 0usize;
		loop {
			let rest = &a.buf[used..];
			if rest.len() < 7 {
				break;
			}
			let frame = match es::audio_frame(e.framing, rest) {
				Some(f) if f.len <= rest.len() => f,
				Some(_) => break, // the frame continues in the next PES
				None if rest.len() < 64 => break,
				None => {
					// Lost sync: skip to the next candidate header byte.
					let sync = if e.framing == Framing::Ac3 { 0x0B } else { 0xFF };
					used += rest[1..].iter().position(|&b| b == sync).map_or(rest.len(), |p| p + 1);
					continue;
				}
			};
			let offset = a.start + used as u64;
			let mut stamp = None;
			while a.stamps.front().is_some_and(|&(o, _)| o <= offset) {
				stamp = a.stamps.pop_front().map(|(_, t)| t);
			}
			let rate = u64::from(frame.sample_rate.max(1));
			let duration = u64::from(frame.samples) * 90_000 / rate;
			// A frame with its own PES timestamp keeps it. FFmpeg places the
			// frames after it differently per codec: its AAC parser leaves
			// the duration unset, so the next time comes from the exact frame
			// length (`av_add_stable`); the MPEG audio and AC-3 parsers set
			// it, so their whole-tick duration is added.
			let pts = stamp.or(a.next_pts);
			a.next_pts = pts.map(|t| match e.framing {
				Framing::Adts => add_stable(t, u64::from(frame.samples), rate),
				_ => t + duration,
			});
			if e.ready.is_none() {
				e.ready = Some(es::make_stream(0, e.codec, e.kind, Some(&frame), None));
			}
			let mut packet = Packet::new(StreamId(0), rest[..frame.len].to_vec());
			packet.pts = pts.map(|t| Time::new(t, TIME_BASE));
			packet.dts = packet.pts;
			packet.duration = Some(Time::new(duration, TIME_BASE));
			used += frame.len;
			self.out.push_back((i, packet));
		}
		a.buf.drain(..used.min(a.buf.len()));
		a.start += used as u64;
		// Stamps for bytes already skipped can't apply any more.
		while a.stamps.len() > 1 && a.stamps.get(1).is_some_and(|&(o, _)| o <= a.start) {
			a.stamps.pop_front();
		}
	}

	/// Splits Opus PES payloads at their control headers: one packet per
	/// access unit, without the header. Timestamps as for MPEG audio (a PES
	/// time for the first unit starting in it, then whole-tick durations,
	/// which are exact for Opus frame sizes); an end trim shortens the
	/// packet's duration, the first start trim becomes the pre-skip.
	fn opus_pes(&mut self, i: usize, data: &[u8], pts: Option<u64>) {
		let e = &mut self.es[i];
		let a = &mut e.audio;
		let at = a.start + a.buf.len() as u64;
		if let Some(pts) = pts {
			a.stamps.push_back((at, pts));
		}
		a.buf.extend_from_slice(data);
		let mut used = 0usize;
		loop {
			let rest = &a.buf[used..];
			if rest.len() < 3 {
				break;
			}
			let unit = match es::opus_control_header(rest) {
				Some(u) if u.header + u.size <= rest.len() => u,
				Some(_) => break,
				None if rest.len() < 512 && rest[0] == 0x7F => break,
				None => {
					used += rest[1..].iter().position(|&b| b == 0x7F).map_or(rest.len(), |p| p + 1);
					continue;
				}
			};
			let payload = &rest[unit.header..unit.header + unit.size];
			let offset = a.start + used as u64;
			let mut stamp = None;
			while a.stamps.front().is_some_and(|&(o, _)| o <= offset) {
				stamp = a.stamps.pop_front().map(|(_, t)| t);
			}
			let samples = crate::codecs::opus::packet_samples(payload).unwrap_or(960);
			let duration = u64::from(samples) * 90_000 / 48_000;
			let pts = stamp.or(a.next_pts);
			a.next_pts = pts.map(|t| t + duration);
			if e.ready.is_none() {
				let Some(head) = es::opus_head(e.opus_config, unit.start_trim) else {
					used += unit.header + unit.size;
					continue;
				};
				let mut stream = es::make_stream(0, "opus", StreamKind::Audio, None, None);
				stream.parameters.sample_rate = Some(48_000);
				stream.parameters.channels = Some(head.channels);
				stream.parameters.extradata = head.to_bytes();
				if unit.start_trim > 0 {
					stream
						.metadata
						.raw
						.insert("encoder_delay".into(), RawValue::String(unit.start_trim.to_string()));
				}
				e.ready = Some(stream);
			}
			let trimmed = u64::from(samples.saturating_sub(u32::from(unit.end_trim)));
			let mut packet = Packet::new(StreamId(0), payload.to_vec());
			packet.pts = pts.map(|t| Time::new(t, TIME_BASE));
			packet.dts = packet.pts;
			packet.duration = Some(Time::new(trimmed * 90_000 / 48_000, TIME_BASE));
			used += unit.header + unit.size;
			self.out.push_back((i, packet));
		}
		a.buf.drain(..used.min(a.buf.len()));
		a.start += used as u64;
		while a.stamps.len() > 1 && a.stamps.get(1).is_some_and(|&(o, _)| o <= a.start) {
			a.stamps.pop_front();
		}
	}

	/// End of input: completes every PES in progress and releases held
	/// video packets.
	fn flush(&mut self) {
		for i in 0..self.es.len() {
			if self.es[i].in_pes {
				self.finish_pes(i);
			}
			let e = &mut self.es[i];
			if let Some(mut held) = e.held.take() {
				held.duration = e.last_duration.map(|d| Time::new(d, TIME_BASE));
				self.out.push_back((i, held));
			}
		}
	}

	fn reset(&mut self) {
		self.out.clear();
		for e in &mut self.es {
			e.reset();
		}
	}
}

/// `ts + samples / rate` in 90 kHz ticks the way FFmpeg's `av_add_stable`
/// advances a parser's timestamps: `ts` is snapped to the grid of whole
/// frames, moved one frame on, and its offset from the grid kept. Repeated
/// additions then never drift, whatever the rounding of each frame.
fn add_stable(ts: u64, samples: u64, rate: u64) -> u64 {
	let (m, d) = (u128::from(samples) * 90_000, u128::from(rate));
	if d == 0 || m % d == 0 {
		return ts + (m / d.max(1)) as u64;
	}
	let round = |num: u128, den: u128| (num + den / 2) / den;
	let ts = u128::from(ts);
	let old = round(ts * d, m);
	let old_ts = round(old * m, d);
	let next = round((old + 1) * m, d);
	u64::try_from(next + ts - old_ts.min(next + ts)).unwrap_or(u64::MAX)
}

pub struct TsDemuxer {
	reader: BufferedReader,
	layout: Layout,
	core: Core,
	streams: StreamSet,
	duration: Time,
	/// Index in `core.es` of the stream used for seeking.
	reference: Option<usize>,
	eof: bool,
}

/// Reads the next transport packet into `buf`, resynchronising after
/// garbage. Returns the file position of the packet (its stride start),
/// or `None` at end of data.
fn next_packet(
	reader: &mut BufferedReader,
	layout: Layout,
) -> Result<Option<(u64, [u8; TS_SIZE])>> {
	loop {
		let need = layout.stride.max(layout.prefix + TS_SIZE);
		let position = reader.position();
		let buf = reader.fill(need * 2)?;
		if buf.len() < layout.prefix + TS_SIZE {
			return Ok(None);
		}
		let sync = buf[layout.prefix] == packet::SYNC
			&& buf.get(layout.stride + layout.prefix).is_none_or(|&b| b == packet::SYNC);
		if sync {
			let mut p = [0u8; TS_SIZE];
			p.copy_from_slice(&buf[layout.prefix..layout.prefix + TS_SIZE]);
			let step = layout.stride.min(buf.len());
			reader.consume(step);
			return Ok(Some((position, p)));
		}
		// Lost sync: find the next byte where two packets line up.
		let skip = (1..buf.len())
			.find(|&i| {
				buf.get(i + layout.prefix) == Some(&packet::SYNC)
					&& buf.get(i + layout.prefix + layout.stride).is_none_or(|&b| b == packet::SYNC)
			})
			.unwrap_or(buf.len());
		reader.consume(skip);
	}
}

impl TsDemuxer {
	pub fn new(io: Box<dyn Io>) -> Result<Self> {
		Self::open(io, true)
	}

	/// For one HLS segment: no end-of-file scan for the duration (the
	/// playlist has it).
	pub fn new_segment(io: Box<dyn Io>) -> Result<Self> {
		Self::open(io, false)
	}

	fn open(io: Box<dyn Io>, scan_duration: bool) -> Result<Self> {
		let mut reader = BufferedReader::with_capacity(io, 64 << 10);
		let head = reader.fill(32 * 204)?;
		let (layout, first) =
			packet::detect_layout(head).ok_or(Message::Container("mpeg-ts: no packet sync found"))?;
		reader.consume(first);
		let start = reader.position();

		let mut core = Core::new();
		while reader.position() - start < PROBE_BYTES && !core.all_ready() {
			match next_packet(&mut reader, layout)? {
				Some((_, p)) => core.packet(&p),
				None => {
					core.flush();
					break;
				}
			}
		}
		core.probing = false;
		if core.es.is_empty() {
			return Err(Message::Container("mpeg-ts: no supported elementary streams"));
		}

		// Streams in PMT order; those still without parameters are dropped.
		let mut streams = StreamSet::default();
		for e in &mut core.es {
			let Some(mut stream) = e.ready.take() else { continue };
			stream.id = StreamId(streams.streams.len());
			if let Some(lang) = e.language.take() {
				stream.metadata.raw.insert("language".into(), RawValue::String(lang));
			}
			if e.sample_aes {
				stream.metadata.raw.insert("encryption".into(), RawValue::String("SAMPLE-AES".into()));
			}
			e.stream = Some(stream.id.0);
			streams.add(stream);
		}
		if streams.streams.is_empty() {
			return Err(Message::Container("mpeg-ts: no stream parameters found"));
		}
		let reference = core
			.es
			.iter()
			.position(|e| e.stream.is_some() && e.kind == StreamKind::Video)
			.or_else(|| core.es.iter().position(|e| e.stream.is_some()));

		let mut demuxer = Self {
			reader,
			layout,
			core,
			streams,
			duration: Time::zero(TIME_BASE),
			reference,
			eof: false,
		};
		if scan_duration {
			demuxer.compute_durations()?;
		}
		Ok(demuxer)
	}

	/// Stream start and end times: starts from the probed packets, ends
	/// from demuxing the last [`TAIL_BYTES`] of the file.
	fn compute_durations(&mut self) -> Result<()> {
		let n = self.streams.streams.len();
		let mut start = vec![u64::MAX; n];
		for (i, p) in &self.core.out {
			if let (Some(s), Some(t)) = (self.core.es[*i].stream, p.pts.or(p.dts)) {
				start[s] = start[s].min(t.ticks());
			}
		}
		let Some(size) = self.reader.size() else { return Ok(()) };
		let resume = self.reader.position();
		let mut tail = self.core.fork();
		let mut end = vec![0u64; n];
		let mut reader_pos = size.saturating_sub(TAIL_BYTES);
		reader_pos -= reader_pos % self.layout.stride as u64;
		self.reader.seek(reader_pos)?;
		loop {
			let done = match next_packet(&mut self.reader, self.layout)? {
				Some((_, p)) => {
					tail.packet(&p);
					false
				}
				None => {
					tail.flush();
					true
				}
			};
			while let Some((i, p)) = tail.out.pop_front() {
				let (Some(s), Some(t)) = (tail.es[i].stream, p.pts.or(p.dts)) else { continue };
				let d = p.duration.map_or(0, |d| d.ticks());
				end[s] = end[s].max(t.ticks() + d);
			}
			if done {
				break;
			}
		}
		self.reader.seek(resume)?;
		let (mut first, mut last) = (u64::MAX, 0u64);
		for (s, stream) in self.streams.streams.iter_mut().enumerate() {
			if start[s] == u64::MAX || end[s] <= start[s] {
				continue;
			}
			stream.duration = Some(Time::new(end[s] - start[s], TIME_BASE));
			first = first.min(start[s]);
			last = last.max(end[s]);
		}
		if last > first {
			self.duration = Time::new(last - first, TIME_BASE);
		}
		Ok(())
	}

	/// First timestamp of the reference stream whose PES starts at or after
	/// `pos` (within one window), with that PES's position and whether it
	/// begins a keyframe. Every PES start in the window is returned.
	fn scan(&mut self, pos: u64, window: u64) -> Result<Vec<(u64, u64, bool)>> {
		let Some(r) = self.reference else { return Ok(Vec::new()) };
		let es = &self.core.es[r];
		let (pid, framing, mut unwrap) = (es.pid, es.framing, es.unwrap);
		let pos = pos - pos % self.layout.stride as u64;
		self.reader.seek(pos)?;
		let mut found = Vec::new();
		while let Some((at, p)) = next_packet(&mut self.reader, self.layout)? {
			if at >= pos + window {
				break;
			}
			let Some(h) = packet::parse(&p) else { continue };
			if h.pid != pid || !h.unit_start || !h.has_payload {
				continue;
			}
			let payload = &p[h.payload_at..];
			let Some(pes) = packet::parse_pes(payload) else { continue };
			let Some(t) = pes.dts.or(pes.pts) else { continue };
			let key = match framing {
				Framing::H264 | Framing::Hevc | Framing::Mpeg2Video => {
					h.random_access
						|| payload.get(pes.payload_at..).is_some_and(|d| es::inspect_video(framing, d).keyframe)
				}
				Framing::Video => h.random_access,
				_ => true,
			};
			found.push((at, unwrap.apply(t), key));
		}
		Ok(found)
	}
}

impl Demuxer for TsDemuxer {
	fn read(&mut self) -> Result<Option<Packet>> {
		loop {
			while let Some((i, mut packet)) = self.core.out.pop_front() {
				if let Some(s) = self.core.es[i].stream {
					packet.stream_id = StreamId(s);
					return Ok(Some(packet));
				}
			}
			if self.eof {
				return Ok(None);
			}
			// Several packets at a time keeps the per-call overhead low.
			for _ in 0..64 {
				match next_packet(&mut self.reader, self.layout)? {
					Some((_, p)) => self.core.packet(&p),
					None => {
						self.core.flush();
						self.eof = true;
						break;
					}
				}
				if !self.core.out.is_empty() {
					break;
				}
			}
		}
	}

	/// `time` is a presentation time in seconds on the file's own clock
	/// (the timestamps [`read`](Demuxer::read) returns), as with the other
	/// demuxers. Lands on the last reference-stream keyframe at or before it.
	fn seek(&mut self, time: f64) -> Result<()> {
		if !time.is_finite() || time < 0.0 {
			return Err(Message::Other("invalid seek time"));
		}
		let size = self.reader.size().ok_or(Message::Other("mpeg-ts: input is not seekable"))?;
		let target = (time * 90_000.0) as u64;
		let stride = self.layout.stride as u64;

		// 1. Bisect for the last window starting before the target.
		let (mut lo, mut hi) = (0u64, size);
		while hi - lo > SEEK_WINDOW {
			let mid = lo + (hi - lo) / 2;
			match self.scan(mid, SEEK_WINDOW)?.first() {
				Some(&(_, t, _)) if t <= target => lo = mid,
				Some(_) => hi = mid,
				None => hi = mid,
			}
		}
		// 2. Walk back window by window to a keyframe at or before it.
		let mut from = lo;
		let position = loop {
			let hit = self
				.scan(from, SEEK_WINDOW + (lo - from))?
				.into_iter()
				.rfind(|&(_, t, key)| key && t <= target)
				.map(|(at, _, _)| at);
			match hit {
				Some(at) => break at,
				None if from == 0 => break 0,
				None => from = from.saturating_sub(4 * SEEK_WINDOW),
			}
		};
		self.core.reset();
		self.eof = false;
		self.reader.seek(position - position % stride)?;
		Ok(())
	}

	fn duration(&self) -> Time {
		self.duration
	}

	fn metadata(&self) -> &Metadata {
		&self.core.metadata
	}

	fn streams(&self) -> &StreamSet {
		&self.streams
	}
}
