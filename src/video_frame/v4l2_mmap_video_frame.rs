// Copyright 2025 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

use std::cell::RefCell;
use std::fmt;
use std::fmt::Debug;
use std::iter::zip;
use std::sync::Arc;

use crate::utils::align_up;
use crate::video_frame::ReadMapping;
use crate::video_frame::VideoFrame;
use crate::video_frame::WriteMapping;
use crate::Fourcc;
use crate::Resolution;

use crate::v4l2r::device::Device;
#[cfg(feature = "vaapi")]
use libva::Display;
#[cfg(feature = "vaapi")]
use libva::Surface;
use v4l2r::bindings::v4l2_plane;
use v4l2r::ioctl::{mmap, PlaneMapping, V4l2Buffer};
use v4l2r::memory::{MmapHandle, PlaneHandle};
use v4l2r::Format;

pub struct V4l2MmapMapping {
    planes: Vec<PlaneMapping>,
    // For contiguous formats
    logical_offsets: Vec<usize>,
    logical_sizes: Vec<usize>,
}

impl<'a> ReadMapping<'a> for V4l2MmapMapping {
    fn get(&self) -> Vec<&[u8]> {
        if self.logical_offsets.is_empty() {
            return self.planes.iter().map(|x| x.as_ref()).collect();
        }
        let base = self.planes[0].data.as_ptr();
        self.logical_offsets
            .iter()
            .zip(&self.logical_sizes)
            .map(|(&off, &size)| unsafe { std::slice::from_raw_parts(base.add(off), size) })
            .collect()
    }
}

impl<'a> WriteMapping<'a> for V4l2MmapMapping {
    fn get(&self) -> Vec<RefCell<&'a mut [u8]>> {
        if self.logical_offsets.is_empty() {
            return self
                .planes
                .iter()
                .map(|x| {
                    let ptr = x.data.as_ptr() as *mut u8;
                    let len = x.data.len();
                    unsafe { RefCell::new(std::slice::from_raw_parts_mut(ptr, len)) }
                })
                .collect();
        }
        let base = self.planes[0].data.as_ptr() as *mut u8;
        self.logical_offsets
            .iter()
            .zip(&self.logical_sizes)
            .map(|(&off, &size)| unsafe {
                RefCell::new(std::slice::from_raw_parts_mut(base.add(off), size))
            })
            .collect()
    }
}

pub struct V4l2MmapVideoFrame {
    fourcc: Fourcc,
    resolution: Resolution,
    handle: MmapHandle,
    device: Option<Arc<Device>>,
    queue_format: Option<Format>,
    buffer: Option<V4l2Buffer>,
}

impl Debug for V4l2MmapVideoFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MmapVideoFrame")
            .field("fourcc", &self.fourcc)
            .field("resolution", &self.resolution)
            .field("queue_format", &self.queue_format)
            .field("buffer", &self.buffer)
            .finish()
    }
}

impl V4l2MmapVideoFrame {
    pub fn new(fourcc: Fourcc, resolution: Resolution) -> Self {
        V4l2MmapVideoFrame {
            fourcc,
            resolution,
            handle: MmapHandle {},
            device: None,
            queue_format: None,
            buffer: None,
        }
    }

    fn map_helper(&self) -> Result<V4l2MmapMapping, String> {
        let device = self.device.as_ref().ok_or("No V4L2 device!".to_string())?;
        let buffer = self.buffer.as_ref().ok_or("No V4L2 buffer!")?;
        let v4l2_planes = buffer.as_v4l2_planes();

        let mapped: Vec<PlaneMapping> = v4l2_planes
            .iter()
            .map(|x| unsafe { mmap(device, x.m.mem_offset, x.length) })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| format!("Error mmap'ing buffer {err}"))?;

        let num_logical = self.num_planes();
        let num_v4l2 = mapped.len();

        if num_v4l2 >= num_logical {
            return Ok(V4l2MmapMapping {
                planes: mapped,
                logical_offsets: Vec::new(),
                logical_sizes: Vec::new(),
            });
        }

        let vertical_subsampling = self.get_vertical_subsampling();
        let horizontal_subsampling = self.get_horizontal_subsampling();
        let bpp = self.get_bytes_per_element();
        let plane_sizes: Vec<usize> = (0..num_logical)
            .map(|i| {
                align_up(self.resolution.width as usize, horizontal_subsampling[i])
                    / horizontal_subsampling[i]
                    * align_up(self.resolution.height as usize, vertical_subsampling[i])
                    / vertical_subsampling[i]
                    * bpp[i]
            })
            .collect();

        let full = mapped.into_iter().next().unwrap();
        let full_len = full.data.len();

        let mut offset = 0;
        let mut offsets = Vec::new();
        let mut sizes = Vec::new();
        for &size in &plane_sizes {
            assert!(offset + size <= full_len);
            offsets.push(offset);
            sizes.push(size);
            offset += size;
        }

        Ok(V4l2MmapMapping { planes: vec![full], logical_offsets: offsets, logical_sizes: sizes })
    }
}

impl VideoFrame for V4l2MmapVideoFrame {
    #[cfg(feature = "v4l2")]
    type NativeHandle = MmapHandle;

    #[cfg(feature = "vaapi")]
    type MemDescriptor = ();
    #[cfg(feature = "vaapi")]
    type VaapiHandle = Surface<()>;

    fn fourcc(&self) -> Fourcc {
        self.fourcc.clone()
    }

    fn resolution(&self) -> Resolution {
        self.resolution.clone()
    }

    fn get_plane_size(&self) -> Vec<usize> {
        match self.buffer.as_ref() {
            Some(buffer) => buffer.as_v4l2_planes().iter().map(|x| x.length as usize).collect(),
            None => {
                let mut plane_size: Vec<usize> = vec![];
                let vertical_subsampling = self.get_vertical_subsampling();
                let horizontal_subsampling = self.get_horizontal_subsampling();
                let bpp = self.get_bytes_per_element();
                for i in 0..self.num_planes() {
                    plane_size.push(
                        align_up(self.resolution.width as usize, horizontal_subsampling[i])
                            / horizontal_subsampling[i]
                            * align_up(self.resolution.height as usize, vertical_subsampling[i])
                            / vertical_subsampling[i]
                            * bpp[i],
                    );
                }
                plane_size
            }
        }
    }

    fn get_plane_pitch(&self) -> Vec<usize> {
        match self.queue_format.as_ref() {
            Some(format) => format.plane_fmt.iter().map(|x| x.bytesperline as usize).collect(),
            None => zip(self.get_bytes_per_element(), self.get_horizontal_subsampling())
                .map(|x| align_up(self.resolution.width as usize, x.1) / x.1 * x.0)
                .collect(),
        }
    }

    fn map<'a>(&'a self) -> Result<Box<dyn ReadMapping<'a> + 'a>, String> {
        Ok(Box::new(self.map_helper()?))
    }

    fn map_mut<'a>(&'a mut self) -> Result<Box<dyn WriteMapping<'a> + 'a>, String> {
        Ok(Box::new(self.map_helper()?))
    }

    #[cfg(feature = "v4l2")]
    fn fill_v4l2_plane(&self, _index: usize, plane: &mut v4l2_plane) {
        self.handle.fill_v4l2_plane(plane)
    }

    #[cfg(feature = "v4l2")]
    fn process_dqbuf(&mut self, device: Arc<Device>, format: &Format, buf: &V4l2Buffer) {
        self.device = Some(device);
        self.queue_format = Some(format.clone());
        self.buffer = Some(buf.clone());
    }

    #[cfg(feature = "vaapi")]
    fn to_native_handle(
        &self,
        _display: &Arc<Display>,
    ) -> Result<Self::VaapiHandle, String> {
        Err("V4L2 mmap frames are not compatible with VA-API".to_string())
    }
}
