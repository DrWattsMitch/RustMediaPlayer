//! Seeks to a few positions and prints the first packet after each seek.
use rmp::Result;
use rmp::core::traits::*;
use rmp::io::Input;

pub fn main() -> Result<()> {
	let path = std::env::args().nth(1).ok_or("usage: seek_check <input>")?;
	let mut input = Input::open(&path)?;
	for t in [0.0, 1.0, 2.5, 3.9, 100.0, 0.5] {
		input.seek(t)?;
		match input.read()? {
			Some(p) => println!(
				"seek {t:>5}: stream {} pts={:.3}s key={} size={}",
				p.stream_id.0,
				p.pts.map_or(-1.0, |x| x.as_seconds()),
				p.keyframe,
				p.data.len()
			),
			None => println!("seek {t:>5}: end of stream"),
		}
	}
	Ok(())
}
