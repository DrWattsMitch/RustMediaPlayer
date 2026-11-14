use crate::{
	core::{Frame, Hint, Metadata, Packet, Parameters, State, Stream, StreamSet, Time},
	io::Io,
	message::Result,
};

/// What one decode or encode call produced.
///
/// The codec state stays in the caller's `&mut State`; it is not cloned
/// into the result (that cost a deep copy of the decoder state, with its
/// tables and buffers, on every packet).
pub struct StepOutput<O> {
	pub value: O,
}

pub type DecodeOut = StepOutput<Vec<Frame>>;
pub type EncodeOut = StepOutput<Vec<Packet>>;

pub struct FilterOut<S> {
	pub value: Vec<Frame>,
	pub state: S,
}

/// `Send` so that a decoder can run on a pipeline thread.
pub trait Decoder: Send {
	fn create_state(&self, parameters: &Parameters) -> State;
	fn decode(&self, state: &mut State, packet: Packet) -> Result<DecodeOut>;
	fn flush(&self, state: &mut State) -> Result<DecodeOut>;
}

/// `Send` so that an encoder can run on a pipeline thread.
pub trait Encoder: Send {
	fn create_state(&self, parameters: &Parameters) -> State;
	fn encode(&self, state: &mut State, frame: Frame) -> Result<EncodeOut>;
	fn flush(&self, state: &mut State) -> Result<EncodeOut>;
}

pub trait Filter {
	type State;
	fn filter(&self, state: &mut Self::State, frame: Frame) -> Result<FilterOut<Self::State>>;
	fn flush(&self, state: &mut Self::State) -> Result<FilterOut<Self::State>>;
}

pub trait Demuxer {
	fn read(&mut self) -> Result<Option<Packet>>;
	fn seek(&mut self, time: f64) -> Result<()>;
	fn duration(&self) -> Time;
	fn metadata(&self) -> &Metadata;
	fn streams(&self) -> &StreamSet;
}

/// `Send` so that muxing can run on a pipeline thread.
pub trait Muxer: Send {
	fn set_metadata(&mut self, metadata: Metadata) -> Result<()>;
	fn metadata(&self) -> &Metadata;
	fn add(&mut self, stream: &Stream) -> Result<usize>;

	fn add_all(&mut self, setter: &StreamSet) -> Result<usize> {
		for stream in setter.streams.iter() {
			self.add(stream)?;
		}
		Ok(0)
	}

	/// Sets a container option by its FFmpeg name (`movflags`,
	/// `frag_duration`, ...). Must be called before the first packet.
	/// Containers without options reject every key.
	#[allow(unused_variables)]
	fn set_option(&mut self, key: &str, value: &str) -> Result<()> {
		Err(crate::Message::Container("this output container has no such option"))
	}

	fn write(&mut self, packet: Packet) -> Result<()>;
	fn finish(&mut self) -> Result<()>;
}

pub trait Probe {
	fn demuxer_matches(data: &[u8]) -> bool;
	fn muxer_matches(hint: &Hint) -> bool;
	fn create_demuxer(io: Box<dyn Io>) -> Result<Box<dyn Demuxer>>;
	fn create_muxer(io: Box<dyn Io>) -> Result<Box<dyn Muxer>>;
}
