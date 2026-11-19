//! Decoded text subtitles: one timed cue per frame.
//!
//! FFmpeg decodes every text subtitle codec into an `AVSubtitle` whose
//! rectangles carry ASS dialogue lines. RMP keeps the cue's own markup and
//! adds a plain-text rendering, so a player can either draw the text as it
//! is or interpret the styling itself; rendering stays in the player.

use crate::core::Time;

/// The markup language of [`SubtitleFrame::markup`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubtitleFormat {
	/// SubRip (`.srt`, Matroska `S_TEXT/UTF8`): HTML-like `<b>`, `<i>`,
	/// `<u>`, `<font>` tags and the `{\an8}` position extension.
	SubRip,
	/// WebVTT cue text: `<b>`, `<i>`, `<u>`, `<c.class>`, `<v Speaker>`,
	/// `<ruby>`, timestamps and HTML character references.
	WebVtt,
	/// The `Text` field of an ASS/SSA `Dialogue` event: `{\override}`
	/// blocks, `\N` hard and `\n` soft line breaks, `\h` hard spaces.
	Ass,
	/// Unstyled text (MP4 `tx3g` / `mov_text`, plain `text`).
	Plain,
}

/// The fields of an ASS `Dialogue` event other than its times and text,
/// as Matroska and FFmpeg's `ass` packets carry them
/// (`ReadOrder,Layer,Style,Name,MarginL,MarginR,MarginV,Effect,Text`).
/// Margins of 0 mean "use the style's".
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AssEvent {
	pub read_order: i64,
	pub layer: i32,
	pub style: String,
	pub name: String,
	pub margin_l: i32,
	pub margin_r: i32,
	pub margin_v: i32,
	pub effect: String,
}

/// One subtitle cue.
#[derive(Debug, Clone, PartialEq)]
pub struct SubtitleFrame {
	/// When the cue appears (the packet's pts).
	pub pts: Option<Time>,
	/// How long it stays; `None` when the container gives no duration
	/// (the cue then lasts until the next one).
	pub duration: Option<Time>,
	/// The text with all markup removed and character references resolved;
	/// lines are separated by `\n`.
	pub text: String,
	/// The cue's text as coded, in [`SubtitleFrame::format`] syntax.
	pub markup: String,
	pub format: SubtitleFormat,
	/// The event fields of an ASS cue.
	pub ass: Option<AssEvent>,
}

impl SubtitleFrame {
	/// When the cue disappears, in the pts's time base.
	pub fn end(&self) -> Option<Time> {
		let pts = self.pts?;
		match self.duration {
			Some(d) => {
				pts.checked_add(d).or_else(|| d.rescale(pts.base()).and_then(|d| pts.checked_add(d)))
			}
			None => None,
		}
	}

	/// Start in seconds (0 without a pts).
	pub fn start_seconds(&self) -> f64 {
		self.pts.map_or(0.0, |t| t.as_seconds())
	}

	/// End in seconds; `f64::INFINITY` without a duration.
	pub fn end_seconds(&self) -> f64 {
		match self.duration {
			Some(d) => self.start_seconds() + d.as_seconds(),
			None => f64::INFINITY,
		}
	}

	/// `true` while the cue is on screen at `seconds` (start inclusive, end
	/// exclusive).
	pub fn is_showing(&self, seconds: f64) -> bool {
		seconds >= self.start_seconds() && seconds < self.end_seconds()
	}
}
