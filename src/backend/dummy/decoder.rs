// Copyright 2022 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! This file contains a dummy backend whose only purpose is to let the decoder
//! run so we can test it in isolation.

use std::sync::Arc;

use crate::decoder::DecodedHandle;
use crate::decoder::StreamInfo;
use crate::decoder::stateless::StatelessDecoderBackend;
use crate::decoder::stateless::StatelessDecoderBackendPicture;
use crate::decoder::stateless::StatelessCodec;
use crate::video_frame::ReadMapping;
use crate::video_frame::VideoFrame;
use crate::video_frame::WriteMapping;
use crate::DecodedFormat;
use crate::Fourcc;
use crate::Resolution;

#[derive(Default, Debug)]
pub struct DummyFrame;

impl VideoFrame for DummyFrame {
    fn fourcc(&self) -> Fourcc {
        Fourcc::from(b"NV12")
    }

    fn resolution(&self) -> Resolution {
        Resolution::default()
    }

    fn get_plane_size(&self) -> Vec<usize> {
        vec![1, 1]
    }

    fn get_plane_pitch(&self) -> Vec<usize> {
        vec![1, 1]
    }

    fn map<'a>(&'a self) -> Result<Box<dyn ReadMapping<'a> + 'a>, String> {
        Err("dummy backend does not support mapping".to_string())
    }

    fn map_mut<'a>(&'a mut self) -> Result<Box<dyn WriteMapping<'a> + 'a>, String> {
        Err("dummy backend does not support mapping".to_string())
    }

    #[cfg(feature = "v4l2")]
    fn fill_v4l2_plane(&self, _index: usize, _plane: &mut v4l2r::bindings::v4l2_plane) {}

    #[cfg(feature = "v4l2")]
    fn process_dqbuf(
        &mut self,
        _device: Arc<crate::v4l2r::device::Device>,
        _format: &v4l2r::Format,
        _buf: &v4l2r::ioctl::V4l2Buffer,
    ) {
    }

    #[cfg(feature = "vaapi")]
    fn to_native_handle(
        &self,
        _display: &Arc<libva::Display>,
    ) -> Result<Self::VaapiHandle, String> {
        Err("dummy backend does not support VA-API export".to_string())
    }
}

pub struct Handle {
    pub frame: Arc<DummyFrame>,
}

impl Clone for Handle {
    fn clone(&self) -> Self {
        Self { frame: Arc::clone(&self.frame) }
    }
}

impl DecodedHandle for Handle {
    type Frame = DummyFrame;

    fn video_frame(&self) -> Arc<Self::Frame> {
        Arc::clone(&self.frame)
    }

    fn timestamp(&self) -> u64 {
        0
    }

    fn coded_resolution(&self) -> Resolution {
        Resolution::default()
    }

    fn display_resolution(&self) -> Resolution {
        Resolution::default()
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn sync(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Dummy backend that can be used for any codec.
pub struct Backend {
    stream_info: StreamInfo,
}

impl Backend {
    pub(crate) fn new() -> Self {
        Self {
            stream_info: StreamInfo {
                format: DecodedFormat::I420,
                min_num_frames: 4,
                coded_resolution: Resolution::from((320, 200)),
                display_resolution: Resolution::from((320, 200)),
            },
        }
    }
}

impl<Codec: StatelessCodec> StatelessDecoderBackendPicture<Codec> for Backend {
    type Picture = ();
}

impl StatelessDecoderBackend for Backend {
    type Handle = Handle;

    fn stream_info(&self) -> Option<&StreamInfo> {
        Some(&self.stream_info)
    }

    fn reset_backend(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
}
