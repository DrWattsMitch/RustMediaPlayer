//! Reference-counted frame memory and buffer pools, RMP's counterparts of
//! FFmpeg's `AVBufferRef` and `AVBufferPool` (`libavutil/buffer.c`).
//!
//! * [`VecPool`] recycles `Vec<T>` allocations. A decoder takes a buffer
//!   per picture with [`VecPool::take_filled`] or [`VecPool::take_recycled`]
//!   and gets back a [`Pooled`] (unique, mutable, `Deref<Target = [T]>`).
//!   When the `Pooled` is dropped its allocation goes back to the pool, so
//!   once the pool has warmed up decoding allocates nothing per frame.
//! * [`FrameBuffer`] is an immutable, reference-counted view of bytes:
//!   cloning it is one atomic increment, never a copy. It can wrap a frozen
//!   [`Pooled<u8>`], a plain `Vec<u8>`, or *any* owner that implements
//!   [`BufferSource`]. Decoders use that last form to hand out their own
//!   reference pictures without copying them: the output frame keeps the
//!   decoder's picture alive, and the picture's planes go back to the pool
//!   once the decoder and every frame referencing it have let go.
//!
//! Everything here is safe Rust; buffers are always initialized memory
//! (a recycled buffer holds whatever its previous user wrote).

use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// Default number of idle buffers a pool keeps. Enough for a full H.264 DPB
/// (16 references) plus frames in flight and in the caller's hands.
pub const DEFAULT_POOL_IDLE: usize = 32;

struct PoolInner<T> {
	free: Mutex<Vec<Vec<T>>>,
	max_idle: usize,
	allocated: AtomicU64,
	reused: AtomicU64,
}

/// Allocation counters of a [`VecPool`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PoolStats {
	/// Buffers that had to be allocated (or grown).
	pub allocated: u64,
	/// Buffers handed out again without allocating.
	pub reused: u64,
	/// Buffers currently idle in the pool.
	pub idle: usize,
}

/// A thread-safe pool of `Vec<T>` allocations (FFmpeg's `AVBufferPool`).
///
/// Cloning a pool gives another handle to the same pool. Buffers may be
/// taken on one thread and returned (dropped) on another. A buffer that
/// outlives its pool is simply freed.
pub struct VecPool<T> {
	inner: Arc<PoolInner<T>>,
}

impl<T> Clone for VecPool<T> {
	fn clone(&self) -> Self {
		Self { inner: self.inner.clone() }
	}
}

impl<T> fmt::Debug for VecPool<T> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_struct("VecPool").field("stats", &self.stats()).finish()
	}
}

impl<T> Default for VecPool<T> {
	fn default() -> Self {
		Self::new(DEFAULT_POOL_IDLE)
	}
}

/// Pool of byte buffers (picture planes).
pub type BufferPool = VecPool<u8>;

impl<T> VecPool<T> {
	/// A pool that keeps at most `max_idle` unused buffers.
	pub fn new(max_idle: usize) -> Self {
		Self {
			inner: Arc::new(PoolInner {
				free: Mutex::new(Vec::new()),
				max_idle,
				allocated: AtomicU64::new(0),
				reused: AtomicU64::new(0),
			}),
		}
	}

	pub fn stats(&self) -> PoolStats {
		PoolStats {
			allocated: self.inner.allocated.load(Ordering::Relaxed),
			reused: self.inner.reused.load(Ordering::Relaxed),
			idle: self.lock().len(),
		}
	}

	/// Drops every idle buffer (after a resolution change, say).
	pub fn clear(&self) {
		self.lock().clear();
	}

	fn lock(&self) -> std::sync::MutexGuard<'_, Vec<Vec<T>>> {
		self.inner.free.lock().unwrap_or_else(|e| e.into_inner())
	}

	/// An idle buffer with room for `len` elements, emptied, or a new one.
	fn take_vec(&self, len: usize) -> Vec<T> {
		let candidate = {
			let mut free = self.lock();
			// Prefer a buffer that already fits; otherwise grow the newest one.
			match free.iter().rposition(|v| v.capacity() >= len) {
				Some(i) => Some(free.swap_remove(i)),
				None => free.pop(),
			}
		};
		match candidate {
			Some(mut v) if v.capacity() >= len => {
				self.inner.reused.fetch_add(1, Ordering::Relaxed);
				v.clear();
				v
			}
			_ => {
				self.inner.allocated.fetch_add(1, Ordering::Relaxed);
				Vec::with_capacity(len)
			}
		}
	}

	fn wrap(&self, data: Vec<T>) -> Pooled<T> {
		Pooled { data, home: Some(Arc::downgrade(&self.inner)) }
	}

	/// A buffer of `len` copies of `value` (like `vec![value; len]`).
	pub fn take_filled(&self, len: usize, value: T) -> Pooled<T>
	where
		T: Clone,
	{
		let mut v = self.take_vec(len);
		v.resize(len, value);
		self.wrap(v)
	}

	/// An empty buffer with capacity for at least `capacity` elements, to
	/// be filled with [`Pooled::vec_mut`].
	pub fn take_empty(&self, capacity: usize) -> Pooled<T> {
		let v = self.take_vec(capacity);
		self.wrap(v)
	}
}

impl<T: Clone + Default> VecPool<T> {
	/// A buffer of `len` elements whose contents are left over from its
	/// previous use (FFmpeg's pool buffers aren't cleared either). Only for
	/// buffers that are completely written before they are read. A new
	/// allocation is filled with `T::default()`.
	pub fn take_recycled(&self, len: usize) -> Pooled<T> {
		let mut free = self.lock();
		let found = free.iter().rposition(|v| v.len() == len);
		let v = found.map(|i| free.swap_remove(i));
		drop(free);
		match v {
			Some(v) => {
				self.inner.reused.fetch_add(1, Ordering::Relaxed);
				self.wrap(v)
			}
			None => self.take_filled(len, T::default()),
		}
	}
}

/// A uniquely owned buffer from a [`VecPool`]; returns to the pool when
/// dropped. Dereferences to `[T]`, so it indexes, slices and iterates like
/// the `Vec` it replaces, at the same cost.
pub struct Pooled<T> {
	data: Vec<T>,
	home: Option<Weak<PoolInner<T>>>,
}

impl<T> Pooled<T> {
	/// A buffer that belongs to no pool (freed normally on drop).
	pub fn detached(data: Vec<T>) -> Self {
		Self { data, home: None }
	}

	/// Takes the `Vec` out, detaching it from its pool.
	pub fn into_vec(mut self) -> Vec<T> {
		self.home = None;
		std::mem::take(&mut self.data)
	}

	/// The underlying `Vec`, for pushing into a buffer from
	/// [`VecPool::take_empty`]. Growing past the capacity reallocates.
	pub fn vec_mut(&mut self) -> &mut Vec<T> {
		&mut self.data
	}

	/// `true` if dropping this buffer returns it to a live pool.
	pub fn is_pooled(&self) -> bool {
		self.home.as_ref().is_some_and(|h| h.strong_count() > 0)
	}
}

impl Pooled<u8> {
	/// Makes the buffer shared and immutable (FFmpeg's `av_buffer_ref` of a
	/// freshly decoded picture). It still returns to its pool once the last
	/// [`FrameBuffer`] referencing it is dropped.
	pub fn freeze(self) -> FrameBuffer {
		FrameBuffer(Repr::Owned(Arc::new(self)))
	}
}

impl<T> Default for Pooled<T> {
	fn default() -> Self {
		Self::detached(Vec::new())
	}
}

impl<T> Drop for Pooled<T> {
	fn drop(&mut self) {
		let Some(home) = self.home.take().and_then(|h| h.upgrade()) else {
			return;
		};
		if self.data.capacity() == 0 {
			return;
		}
		let mut free = home.free.lock().unwrap_or_else(|e| e.into_inner());
		if free.len() < home.max_idle {
			free.push(std::mem::take(&mut self.data));
		}
	}
}

impl<T: Clone> Clone for Pooled<T> {
	/// Copies into another buffer of the same pool.
	fn clone(&self) -> Self {
		match self.home.as_ref().and_then(Weak::upgrade) {
			Some(inner) => {
				let pool = VecPool { inner };
				let mut out = pool.take_empty(self.data.len());
				out.data.extend_from_slice(&self.data);
				out
			}
			None => Self::detached(self.data.clone()),
		}
	}
}

impl<T> Deref for Pooled<T> {
	type Target = [T];
	#[inline(always)]
	fn deref(&self) -> &[T] {
		&self.data
	}
}

impl<T> DerefMut for Pooled<T> {
	#[inline(always)]
	fn deref_mut(&mut self) -> &mut [T] {
		&mut self.data
	}
}

impl<T: PartialEq> PartialEq for Pooled<T> {
	fn eq(&self, other: &Self) -> bool {
		self.data == other.data
	}
}

impl<T: Eq> Eq for Pooled<T> {}

impl<'a, T> IntoIterator for &'a Pooled<T> {
	type Item = &'a T;
	type IntoIter = std::slice::Iter<'a, T>;
	fn into_iter(self) -> Self::IntoIter {
		self.data.iter()
	}
}

impl<'a, T> IntoIterator for &'a mut Pooled<T> {
	type Item = &'a mut T;
	type IntoIter = std::slice::IterMut<'a, T>;
	fn into_iter(self) -> Self::IntoIter {
		self.data.iter_mut()
	}
}

impl<T> From<Vec<T>> for Pooled<T> {
	fn from(v: Vec<T>) -> Self {
		Self::detached(v)
	}
}

impl<T> fmt::Debug for Pooled<T> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(
			f,
			"Pooled({} elements{})",
			self.data.len(),
			if self.home.is_some() { ", pooled" } else { "" }
		)
	}
}

/// Something that owns bytes a [`FrameBuffer`] can point into: a decoder's
/// reference picture, a GPU staging buffer, a memory-mapped file...
pub trait BufferSource: Send + Sync + 'static {
	fn bytes(&self) -> &[u8];
}

impl BufferSource for Pooled<u8> {
	fn bytes(&self) -> &[u8] {
		&self.data
	}
}

/// Exposes one byte region of a shared owner, chosen by a projection
/// function (e.g. "the luma plane of this decoded picture").
struct Projected<O: Send + Sync + 'static> {
	owner: Arc<O>,
	project: fn(&O) -> &[u8],
}

impl<O: Send + Sync + 'static> BufferSource for Projected<O> {
	fn bytes(&self) -> &[u8] {
		(self.project)(&self.owner)
	}
}

#[derive(Clone)]
enum Repr {
	Owned(Arc<Pooled<u8>>),
	Shared(Arc<dyn BufferSource>),
}

/// Immutable, reference-counted bytes (FFmpeg's `AVBufferRef`). `clone` is
/// an atomic increment; the memory is freed (or returned to its pool) when
/// the last reference goes. `Send + Sync`, so a render thread can hold
/// frames while the decoder moves on.
#[derive(Clone)]
pub struct FrameBuffer(Repr);

impl FrameBuffer {
	/// Wraps a `Vec` (no copy).
	pub fn from_vec(v: Vec<u8>) -> Self {
		Pooled::detached(v).freeze()
	}

	/// Shares bytes owned by `source`.
	pub fn from_source(source: Arc<dyn BufferSource>) -> Self {
		Self(Repr::Shared(source))
	}

	/// Shares the region of `owner` that `project` returns, keeping `owner`
	/// alive for as long as the buffer (or any clone of it) exists.
	pub fn project<O: Send + Sync + 'static>(owner: Arc<O>, project: fn(&O) -> &[u8]) -> Self {
		Self(Repr::Shared(Arc::new(Projected { owner, project })))
	}

	#[inline]
	pub fn as_slice(&self) -> &[u8] {
		match &self.0 {
			Repr::Owned(p) => &p.data,
			Repr::Shared(s) => s.bytes(),
		}
	}

	/// Number of `FrameBuffer`s sharing this memory (`av_buffer_get_ref_count`).
	pub fn ref_count(&self) -> usize {
		match &self.0 {
			Repr::Owned(p) => Arc::strong_count(p),
			Repr::Shared(s) => Arc::strong_count(s),
		}
	}

	/// `true` if both buffers share the same memory.
	pub fn ptr_eq(&self, other: &Self) -> bool {
		match (&self.0, &other.0) {
			(Repr::Owned(a), Repr::Owned(b)) => Arc::ptr_eq(a, b),
			(Repr::Shared(a), Repr::Shared(b)) => Arc::ptr_eq(a, b),
			_ => false,
		}
	}

	/// The bytes as a `Vec`: free if this is the only reference to a
	/// buffer made from a `Vec` or a pool, a copy otherwise.
	pub fn into_vec(self) -> Vec<u8> {
		match self.0 {
			Repr::Owned(p) => match Arc::try_unwrap(p) {
				Ok(p) => p.into_vec(),
				Err(p) => p.data.clone(),
			},
			Repr::Shared(s) => s.bytes().to_vec(),
		}
	}

	/// A writable copy-on-write handle (`av_buffer_make_writable`): the
	/// buffer itself if nothing else references it, otherwise a copy.
	pub fn into_mut(self) -> Pooled<u8> {
		match self.0 {
			Repr::Owned(p) => Arc::try_unwrap(p).unwrap_or_else(|p| (*p).clone()),
			Repr::Shared(s) => Pooled::detached(s.bytes().to_vec()),
		}
	}
}

impl Deref for FrameBuffer {
	type Target = [u8];
	#[inline]
	fn deref(&self) -> &[u8] {
		self.as_slice()
	}
}

impl AsRef<[u8]> for FrameBuffer {
	fn as_ref(&self) -> &[u8] {
		self.as_slice()
	}
}

impl From<Vec<u8>> for FrameBuffer {
	fn from(v: Vec<u8>) -> Self {
		Self::from_vec(v)
	}
}

impl Default for FrameBuffer {
	fn default() -> Self {
		Self::from_vec(Vec::new())
	}
}

impl fmt::Debug for FrameBuffer {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "FrameBuffer({} bytes, {} refs)", self.len(), self.ref_count())
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn pool_recycles_after_last_reference() {
		let pool = BufferPool::new(4);
		let a = pool.take_filled(1000, 7);
		assert_eq!(pool.stats().allocated, 1);
		let shared = a.freeze();
		let other = shared.clone();
		assert_eq!(shared.ref_count(), 2);
		drop(shared);
		assert_eq!(pool.stats().idle, 0, "still referenced");
		drop(other);
		assert_eq!(pool.stats().idle, 1);
		let b = pool.take_recycled(1000);
		assert_eq!(&b[..3], &[7, 7, 7], "recycled buffers keep their contents");
		assert_eq!(pool.stats(), PoolStats { allocated: 1, reused: 1, idle: 0 });
		let c = pool.take_filled(500, 1);
		assert!(c.iter().all(|&v| v == 1));
	}

	#[test]
	fn pool_is_bounded_and_survives_its_buffers() {
		let pool = VecPool::<u32>::new(1);
		let a = pool.take_filled(8, 0);
		let b = pool.take_filled(8, 0);
		drop(a);
		drop(b);
		assert_eq!(pool.stats().idle, 1);
		let c = pool.take_filled(8, 0);
		drop(pool);
		drop(c); // pool gone: freed normally
	}

	#[test]
	fn into_vec_and_into_mut_avoid_copies_when_unique() {
		let v = vec![1u8, 2, 3];
		let ptr = v.as_ptr();
		let b = FrameBuffer::from_vec(v);
		assert_eq!(b.clone().into_vec(), vec![1, 2, 3]);
		let back = b.into_vec();
		assert_eq!(back.as_ptr(), ptr);
		let b = FrameBuffer::from_vec(back);
		let keep = b.clone();
		let mut m = b.into_mut();
		m[0] = 9;
		assert_eq!(&keep[..], &[1, 2, 3], "copy-on-write");
		assert_eq!(&m[..], &[9, 2, 3]);
	}

	#[test]
	fn projected_buffers_keep_their_owner_alive() {
		struct Pic {
			y: Vec<u8>,
		}
		let pic = Arc::new(Pic { y: vec![5; 16] });
		let b = FrameBuffer::project(pic.clone(), |p| &p.y);
		assert_eq!(Arc::strong_count(&pic), 2);
		assert_eq!(&b[..2], &[5, 5]);
		drop(b);
		assert_eq!(Arc::strong_count(&pic), 1);
	}
}
