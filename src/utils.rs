// Copyright 2022 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! Utility functions used by several parts of this crate.
//!
//! This module is for anything that doesn't fit into the other top-level modules. Try not to add
//! new code here unless it really doesn't belong anywhere else.

use std::marker::Copy;
use std::ops::Add;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Sub;
use std::os::fd::OwnedFd;
use std::time::Duration;

use crate::Fourcc;
use crate::FrameLayout;
use crate::PlaneLayout;
use crate::Resolution;
use crate::decoder::BlockingMode;
use crate::decoder::DecodedHandle;
use crate::decoder::DecoderEvent;
use crate::decoder::stateless::DecodeError;
use crate::decoder::stateless::StatelessVideoDecoder;

/// Simple decoding loop that plays the stream once from start to finish.
#[allow(clippy::type_complexity)]
pub fn simple_playback_loop<D, I, H>(
    decoder: &mut D,
    stream_iter: I,
    on_new_frame: &mut dyn FnMut(H),
    allocate_new_frames: &mut dyn FnMut() -> Option<<H as DecodedHandle>::Frame>,
    _output_format: crate::DecodedFormat,
    blocking_mode: BlockingMode,
) -> anyhow::Result<()>
where
    D: StatelessVideoDecoder<Handle = H>,
    H: DecodedHandle,
    I: IntoIterator,
    I::Item: AsRef<[u8]>,
{
    let mut check_events = |decoder: &mut D| -> anyhow::Result<()> {
        while let Some(event) = decoder.next_event() {
            match event {
                DecoderEvent::FrameReady(frame) => {
                    on_new_frame(frame);
                }
                DecoderEvent::FormatChanged => {}
            }
        }
        Ok(())
    };

    for (frame_num, packet) in stream_iter.into_iter().enumerate() {
        let mut bitstream = packet.as_ref();
        loop {
            match decoder.decode(frame_num as u64, bitstream, allocate_new_frames) {
                Ok(bytes_decoded) => {
                    bitstream = &bitstream[bytes_decoded..];

                    if blocking_mode == BlockingMode::Blocking {
                        check_events(decoder)?;
                    }

                    if bitstream.is_empty() {
                        break;
                    }
                }
                Err(DecodeError::CheckEvents) | Err(DecodeError::NotEnoughOutputBuffers(_)) => {
                    decoder.wait_for_next_event(Duration::from_secs(3))?;
                    check_events(decoder)?
                }
                Err(e) => anyhow::bail!(e),
            }
        }
    }

    decoder.flush()?;
    check_events(decoder)
}

/// Frame allocation callback for the dummy backend.
#[cfg(any(test, feature = "fuzzing", fuzzing))]
pub fn simple_playback_loop_owned_frames() -> Option<crate::backend::dummy::decoder::DummyFrame> {
    Some(crate::backend::dummy::decoder::DummyFrame)
}

#[cfg(any(test, feature = "fuzzing", fuzzing))]
pub use crate::bitstream_utils::{IvfIterator, NalIterator};

pub fn align_up<T>(x: T, alignment: T) -> T
where
    T: Add<Output = T> + Sub<Output = T> + Div<Output = T> + Mul<Output = T> + From<u8> + Copy,
{
    ((x + alignment - T::from(1)) / alignment) * alignment
}

// This is the formula we use to approximate the maximum compressed buffer size for a video frame.
pub fn buffer_size_for_area(width: u32, height: u32) -> u32 {
    let area = width * height;
    let mut buffer_size: u32 = 1024 * 1024;

    if area > 720 * 480 {
        buffer_size *= 2;
    }
    if area > 1920 * 1080 {
        buffer_size *= 2;
    }
    buffer_size
}

/// A structure that holds user-allocated memory for a frame as well as its layout.
#[derive(Debug)]
pub struct UserPtrFrame {
    pub buffers: Vec<*mut u8>,
    pub mem_layout: std::alloc::Layout,
    pub layout: FrameLayout,
}

impl UserPtrFrame {
    /// Allocate enough memory to back a NV12 frame of `size` dimension.
    pub fn new_nv12(size: Resolution) -> Self {
        /// Add what is needed to a value in order to make it a multiple of some alignment.
        macro_rules! align {
            ($n:expr, $r:expr) => {
                ($n + ($r - 1)) & !($r - 1)
            };
        }

        // Align according to VAAPI constraints.
        let width = align!(size.width, 16) as usize;
        let height = align!(size.height, 4) as usize;
        let stride = align!(width, 64);
        let uv_start = height * stride;
        let uv_size = (height / 2) * stride;

        Self::alloc(
            FrameLayout {
                format: (Fourcc::from(b"NV12"), 0),
                size: Resolution::from((width as u32, height as u32)),
                planes: vec![
                    PlaneLayout { buffer_index: 0, offset: 0, stride },
                    PlaneLayout { buffer_index: 0, offset: uv_start, stride },
                ],
            },
            uv_start.max(uv_size),
        )
    }

    pub fn alloc(layout: FrameLayout, buffer_size: usize) -> Self {
        let buffer_count = layout
            .planes
            .iter()
            .map(|plane| plane.buffer_index)
            .collect::<std::collections::HashSet<usize>>()
            .len();

        // SAFETY: the invariants of `Layout` are respected.
        let mem_layout =
            unsafe { std::alloc::Layout::from_size_align_unchecked(buffer_size, 4096) };

        let buffers = (0..buffer_count)
            .map(|_| {
                // SAFETY: the invariants of `Layout` are respected.
                unsafe { std::alloc::alloc(mem_layout) }
            })
            .collect();

        Self { buffers, mem_layout, layout }
    }
}

#[derive(Debug)]
pub struct DmabufFrame {
    pub fds: Vec<OwnedFd>,
    pub layout: FrameLayout,
}

impl Drop for UserPtrFrame {
    fn drop(&mut self) {
        for buffer in std::mem::take(&mut self.buffers).into_iter() {
            // Safe because we allocated the memory using `std::alloc::alloc`.
            unsafe { std::alloc::dealloc(buffer, self.mem_layout) }
        }
    }
}
