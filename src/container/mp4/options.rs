//! MP4 muxer options, in FFmpeg's vocabulary (`-movflags`, `-frag_duration`,
//! `-min_frag_duration`), so the same command lines work in both tools.
//!
//! ```text
//! -movflags +faststart                              moov before mdat (progressive download)
//! -movflags frag_keyframe+empty_moov+default_base_moof   fragmented MP4 (MSE, HLS/DASH)
//! -frag_duration 2000000                            cut a fragment every 2 s (microseconds)
//! ```
//!
//! Flags are parsed like FFmpeg's `AV_OPT_TYPE_FLAGS`: `a+b` sets exactly
//! those flags, `+a` adds to the current set, `-a` removes from it.

use crate::message::{Message, Result};

/// Every `-movflags` name RMP accepts, for error messages and `--help`.
pub const MOVFLAGS: &str = "faststart, frag_keyframe, frag_every_frame, empty_moov, \
	default_base_moof, separate_moof, delay_moov, omit_tfhd_offset, skip_trailer";

/// `-movflags` bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MovFlags {
	/// Move `moov` in front of `mdat` when the file is finished.
	pub faststart: bool,
	/// Start a fragment at every video keyframe.
	pub frag_keyframe: bool,
	/// Start a fragment at every packet.
	pub frag_every_frame: bool,
	/// Write a `moov` without samples; all media goes into fragments.
	/// RMP always does this when fragmenting (it is what MSE requires).
	pub empty_moov: bool,
	/// `tfhd` offsets relative to their `moof`. Always the case in RMP.
	pub default_base_moof: bool,
	/// One `moof` + `mdat` pair per track instead of one per fragment.
	pub separate_moof: bool,
	/// Don't write the `mfra` random-access index at the end.
	pub skip_trailer: bool,
}

impl MovFlags {
	fn bit(&mut self, name: &str) -> Result<Option<&mut bool>> {
		Ok(match name {
			"faststart" => Some(&mut self.faststart),
			"frag_keyframe" => Some(&mut self.frag_keyframe),
			"frag_every_frame" => Some(&mut self.frag_every_frame),
			"empty_moov" => Some(&mut self.empty_moov),
			"default_base_moof" => Some(&mut self.default_base_moof),
			"separate_moof" => Some(&mut self.separate_moof),
			"skip_trailer" => Some(&mut self.skip_trailer),
			// What RMP always does when fragmenting: the init segment is
			// written with the first fragment, tfhd never carries a base
			// data offset.
			"delay_moov" | "omit_tfhd_offset" => None,
			_ => return Err(Message::Container("unknown -movflags flag (see rmp --help)")),
		})
	}

	/// Applies an FFmpeg flags expression (`a+b`, `+a`, `-a`, `+a-b`).
	pub fn apply(&mut self, expression: &str) -> Result<()> {
		let expression = expression.trim();
		if expression.is_empty() {
			return Err(Message::Container("empty -movflags value"));
		}
		// Without a leading sign the expression replaces the current set.
		if !expression.starts_with(['+', '-']) {
			*self = Self::default();
		}
		let mut rest = expression;
		while !rest.is_empty() {
			let (add, body) = match rest.as_bytes()[0] {
				b'-' => (false, &rest[1..]),
				b'+' => (true, &rest[1..]),
				_ => (true, rest),
			};
			let end = body.find(['+', '-']).unwrap_or(body.len());
			let name = &body[..end];
			if name.is_empty() {
				return Err(Message::Container("malformed -movflags value"));
			}
			if let Some(bit) = self.bit(name)? {
				*bit = add;
			}
			rest = &body[end..];
		}
		Ok(())
	}
}

/// Everything that changes how [`super::muxer::Mp4Muxer`] lays out a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Mp4Options {
	pub flags: MovFlags,
	/// `-frag_duration`: cut a fragment once it lasts this long (µs).
	pub frag_duration_us: Option<u64>,
	/// `-min_frag_duration`: never cut a fragment shorter than this (µs).
	pub min_frag_duration_us: u64,
}

impl Mp4Options {
	/// Fragmented output was asked for. Like FFmpeg, any fragment rule
	/// (or `empty_moov` / `separate_moof`) turns fragmentation on.
	pub fn fragmented(&self) -> bool {
		let f = &self.flags;
		f.frag_keyframe
			|| f.frag_every_frame
			|| f.empty_moov
			|| f.separate_moof
			|| self.frag_duration_us.is_some()
	}

	/// Sets one option by its FFmpeg name. Returns `false` for names this
	/// muxer doesn't know.
	pub fn set(&mut self, key: &str, value: &str) -> Result<bool> {
		let micros = |v: &str| {
			v.trim()
				.parse::<u64>()
				.map_err(|_| Message::Container("fragment durations are integer microseconds"))
		};
		match key {
			"movflags" => self.flags.apply(value)?,
			"frag_duration" => {
				let us = micros(value)?;
				if us == 0 {
					return Err(Message::Container("-frag_duration must be above 0"));
				}
				self.frag_duration_us = Some(us);
			}
			"min_frag_duration" => self.min_frag_duration_us = micros(value)?,
			_ => return Ok(false),
		}
		Ok(true)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn parses_ffmpeg_flag_expressions() {
		let mut f = MovFlags::default();
		f.apply("frag_keyframe+empty_moov+default_base_moof").unwrap();
		assert!(f.frag_keyframe && f.empty_moov && f.default_base_moof && !f.faststart);
		f.apply("+faststart-empty_moov").unwrap();
		assert!(f.faststart && f.frag_keyframe && !f.empty_moov);
		// No sign: replace.
		f.apply("faststart").unwrap();
		assert_eq!(f, MovFlags { faststart: true, ..Default::default() });
		f.apply("+delay_moov+omit_tfhd_offset").unwrap();
		assert_eq!(f, MovFlags { faststart: true, ..Default::default() });
		assert!(f.apply("+nope").is_err());
		assert!(f.apply("faststart++frag_keyframe").is_err());
		assert!(f.apply("").is_err());
	}

	#[test]
	fn any_fragment_rule_enables_fragmentation() {
		let mut o = Mp4Options::default();
		assert!(!o.fragmented());
		assert!(o.set("movflags", "+faststart").unwrap());
		assert!(!o.fragmented());
		assert!(o.set("frag_duration", "500000").unwrap());
		assert!(o.fragmented());
		assert!(o.set("frag_duration", "0").is_err());
		assert!(!o.set("cluster_size_limit", "1").unwrap());
	}
}
