//! Threading building blocks shared by the frame-, tile- and
//! pipeline-threaded code (FFmpeg `pthread_frame.c`, `pthread_slice.c`,
//! `libavutil/thread.h`), in safe Rust.
//!
//! * [`default_threads`] / [`set_default_threads`]: the thread count used
//!   when a caller doesn't choose one (FFmpeg's `-threads`, default auto).
//! * [`Progress`]: a monotonic counter that one thread reports and others
//!   wait on (`ff_thread_report_progress` / `ff_thread_await_progress`).
//! * [`WorkerPool`]: long-lived worker threads fed through a channel, for
//!   decoders whose state outlives a single call (a scoped thread can't
//!   outlive the packet that started it).

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

/// Hard upper bound on worker threads (FFmpeg caps frame threads at 16 too,
/// `MAX_AUTO_THREADS`).
pub const MAX_THREADS: usize = 16;

/// 0 = not set: use `RMP_THREADS` or the CPU count.
static DEFAULT_THREADS: AtomicUsize = AtomicUsize::new(0);

/// Sets the thread count that [`default_threads`] returns (`0` restores
/// automatic selection). The CLI's `-threads` lands here.
pub fn set_default_threads(n: usize) {
	DEFAULT_THREADS.store(n.min(MAX_THREADS), Ordering::Relaxed);
}

std::thread_local! {
	/// Per-thread override set by [`with_default_threads`] (0 = none).
	static LOCAL_THREADS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Runs `f` with [`default_threads`] returning `n` on this thread (`0`
/// leaves it unchanged). Lets one call, such as a transcode with
/// `-threads`, choose its thread count without changing the process-wide
/// default.
pub fn with_default_threads<R>(n: usize, f: impl FnOnce() -> R) -> R {
	struct Restore(usize);
	impl Drop for Restore {
		fn drop(&mut self) {
			LOCAL_THREADS.with(|c| c.set(self.0));
		}
	}
	let _restore = Restore(LOCAL_THREADS.with(|c| c.replace(n.min(MAX_THREADS))));
	f()
}

/// Threads to use when the caller doesn't say: [`with_default_threads`],
/// else [`set_default_threads`], else the `RMP_THREADS` environment
/// variable, else the number of CPUs (capped at [`MAX_THREADS`]). Always
/// at least 1.
pub fn default_threads() -> usize {
	let local = LOCAL_THREADS.with(|c| c.get());
	if local != 0 {
		return local;
	}
	let set = DEFAULT_THREADS.load(Ordering::Relaxed);
	if set != 0 {
		return set;
	}
	let env = std::env::var("RMP_THREADS").ok().and_then(|v| v.trim().parse::<usize>().ok());
	match env {
		Some(n) if n > 0 => n.min(MAX_THREADS),
		_ => std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, MAX_THREADS),
	}
}

/// A counter that only grows. The writer calls [`Progress::report`]; readers
/// call [`Progress::wait_for`], which returns at once when the value is
/// already reached (one atomic load, no lock).
#[derive(Debug, Default)]
pub struct Progress {
	value: AtomicU32,
	lock: Mutex<()>,
	changed: Condvar,
}

impl Progress {
	pub fn new(value: u32) -> Self {
		Self { value: AtomicU32::new(value), lock: Mutex::new(()), changed: Condvar::new() }
	}

	pub fn get(&self) -> u32 {
		self.value.load(Ordering::Acquire)
	}

	/// Raises the value to `v` (never lowers it) and wakes the waiters.
	pub fn report(&self, v: u32) {
		{
			// Updating under the lock means a waiter that has just checked the
			// value can't miss the notification.
			let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
			self.value.fetch_max(v, Ordering::AcqRel);
		}
		self.changed.notify_all();
	}

	/// Blocks until the value is at least `v`.
	pub fn wait_for(&self, v: u32) {
		if self.get() >= v {
			return;
		}
		let mut guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
		while self.get() < v {
			guard = self.changed.wait(guard).unwrap_or_else(|e| e.into_inner());
		}
	}
}

type Job = Box<dyn FnOnce() + Send + 'static>;

/// Fixed worker threads taking jobs in submission order from one shared
/// queue. Jobs that wait on earlier jobs (through [`Progress`]) can't
/// deadlock as long as every job only waits on jobs submitted before it:
/// the earliest unfinished job is always running.
pub struct WorkerPool {
	sender: Mutex<Option<Sender<Job>>>,
	workers: Mutex<Vec<JoinHandle<()>>>,
	size: usize,
}

impl std::fmt::Debug for WorkerPool {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("WorkerPool").field("size", &self.size).finish()
	}
}

impl WorkerPool {
	/// Starts `size` workers named `"{name}-{i}"`.
	pub fn new(size: usize, name: &str) -> Arc<Self> {
		let (tx, rx) = mpsc::channel::<Job>();
		let rx = Arc::new(Mutex::new(rx));
		let workers = (0..size.max(1))
			.filter_map(|i| {
				let rx = Arc::clone(&rx);
				std::thread::Builder::new()
					.name(format!("{name}-{i}"))
					.spawn(move || {
						loop {
							// Hold the queue lock only while taking a job.
							let job = {
								let queue = rx.lock().unwrap_or_else(|e| e.into_inner());
								queue.recv()
							};
							match job {
								Ok(job) => job(),
								Err(_) => break, // pool dropped
							}
						}
					})
					.ok()
			})
			.collect::<Vec<_>>();
		let size = workers.len();
		Arc::new(Self { sender: Mutex::new(Some(tx)), workers: Mutex::new(workers), size })
	}

	pub fn size(&self) -> usize {
		self.size
	}

	/// Queues a job. Runs it on the calling thread if the pool has no
	/// workers (thread creation failed), so the job always runs.
	pub fn submit(&self, job: impl FnOnce() + Send + 'static) {
		let job: Job = Box::new(job);
		let sender = self.sender.lock().unwrap_or_else(|e| e.into_inner());
		match sender.as_ref() {
			Some(tx) if self.size > 0 => {
				if let Err(mpsc::SendError(job)) = tx.send(job) {
					job();
				}
			}
			_ => {
				drop(sender);
				job();
			}
		}
	}
}

impl Drop for WorkerPool {
	fn drop(&mut self) {
		// Closing the channel ends the workers once the queue is empty.
		self.sender.lock().unwrap_or_else(|e| e.into_inner()).take();
		let workers = std::mem::take(&mut *self.workers.lock().unwrap_or_else(|e| e.into_inner()));
		for w in workers {
			let _ = w.join();
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn progress_wakes_waiters() {
		let p = Arc::new(Progress::new(0));
		let q = Arc::clone(&p);
		let t = std::thread::spawn(move || {
			q.wait_for(5);
			q.get()
		});
		for v in 1..=5 {
			p.report(v);
		}
		assert!(t.join().unwrap() >= 5);
		p.report(2);
		assert_eq!(p.get(), 5, "progress never goes back");
	}

	#[test]
	fn pool_runs_every_job() {
		let pool = WorkerPool::new(3, "test");
		let count = Arc::new(AtomicU32::new(0));
		let done = Arc::new(Progress::new(0));
		for _ in 0..100 {
			let (c, d) = (Arc::clone(&count), Arc::clone(&done));
			pool.submit(move || {
				let n = c.fetch_add(1, Ordering::SeqCst) + 1;
				d.report(n);
			});
		}
		done.wait_for(100);
		assert_eq!(count.load(Ordering::SeqCst), 100);
	}
}
