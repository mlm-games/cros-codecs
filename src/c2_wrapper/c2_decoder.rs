// Copyright 2025 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

use nix::errno::Errno;
use nix::sys::epoll::Epoll;
use nix::sys::epoll::EpollCreateFlags;
use nix::sys::epoll::EpollEvent;
use nix::sys::epoll::EpollFlags;
use nix::sys::epoll::EpollTimeout;
use nix::sys::eventfd::EventFd;

use std::collections::VecDeque;
use std::marker::PhantomData;
use std::os::fd::AsFd;
#[cfg(feature = "vaapi")]
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use thiserror::Error;

use crate::EncodedFormat;
use crate::Fourcc;
use crate::c2_wrapper::C2DecodeJob;
use crate::c2_wrapper::C2State;
use crate::c2_wrapper::C2Status;
use crate::c2_wrapper::C2Worker;
use crate::c2_wrapper::DrainMode;
use crate::c2_wrapper::Job;
use crate::decoder::DecoderEvent;
use crate::decoder::StreamInfo;
use crate::decoder::stateless::DecodeError;
use crate::decoder::stateless::DynStatelessVideoDecoder;
use crate::image_processing::convert_video_frame;
use crate::video_frame::VideoFrame;
use crate::video_frame::frame_pool::FramePool;
use crate::video_frame::frame_pool::PooledVideoFrame;
#[cfg(feature = "gbm")]
use crate::video_frame::gbm_video_frame::{GbmDevice, GbmUsage, GbmVideoFrame};
#[cfg(feature = "v4l2")]
use crate::video_frame::v4l2_mmap_video_frame::V4l2MmapVideoFrame;

#[derive(Debug, Error)]
pub enum C2DecoderPollErrorWrapper {
    #[error("failed to create Epoll: {0}")]
    Epoll(Errno),
    #[error("failed to add poll FDs to Epoll: {0}")]
    EpollAdd(Errno),
}

pub trait C2DecoderBackend {
    type DecoderOptions: Clone + Send + 'static;

    fn new(options: Self::DecoderOptions) -> Result<Self, String>
    where
        Self: Sized;
    fn supported_output_formats(&self) -> Vec<Fourcc>;
    // TODO: Support stateful video decoders.
    fn get_decoder<V: VideoFrame + 'static>(
        &mut self,
        input_format: EncodedFormat,
    ) -> Result<DynStatelessVideoDecoder<V>, String>;
}

// An "importing decoder" can directly import the DMA bufs we are getting, while a "converting
// decoder" is used for performing image processing routines to convert between the video hardware
// output and a pixel format that can be consumed by the GPU and display controller.
// TODO: Come up with a better name for these?
enum C2Decoder<V: VideoFrame> {
    ImportingDecoder(DynStatelessVideoDecoder<V>),
    #[cfg(feature = "gbm")]
    ConvertingDecoderGbm(DynStatelessVideoDecoder<PooledVideoFrame<GbmVideoFrame>>),
    #[cfg(feature = "v4l2")]
    ConvertingDecoderV4l2(DynStatelessVideoDecoder<PooledVideoFrame<V4l2MmapVideoFrame>>),
}

pub struct C2DecoderWorker<V, B>
where
    V: VideoFrame,
    B: C2DecoderBackend,
{
    decoder: C2Decoder<V>,
    epoll_fd: Epoll,
    awaiting_job_event: Arc<EventFd>,
    #[cfg(feature = "gbm")]
    auxiliary_frame_pool_gbm: Option<FramePool<GbmVideoFrame>>,
    #[cfg(feature = "v4l2")]
    auxiliary_frame_pool_v4l2: Option<FramePool<V4l2MmapVideoFrame>>,
    error_cb: Arc<Mutex<dyn FnMut(C2Status) + Send + 'static>>,
    work_done_cb: Arc<Mutex<dyn FnMut(C2DecodeJob<V>) + Send + 'static>>,
    framepool_hint_cb: Arc<Mutex<dyn FnMut(StreamInfo) + Send + 'static>>,
    alloc_cb: Arc<Mutex<dyn FnMut() -> Option<V> + Send + 'static>>,
    work_queue: Arc<Mutex<VecDeque<C2DecodeJob<V>>>>,
    frame_num: u64,
    state: Arc<Mutex<C2State>>,
    _phantom: PhantomData<B>,
}

impl<V, B> C2DecoderWorker<V, B>
where
    V: VideoFrame,
    B: C2DecoderBackend,
{
    #[allow(unused_variables)]
    fn init_converting(
        mut backend: B,
        input_fourcc: Fourcc,
    ) -> Result<
        (C2Decoder<V>, Option<FramePool<GbmVideoFrame>>, Option<FramePool<V4l2MmapVideoFrame>>),
        String,
    > {
        let _ = backend;
        #[cfg(feature = "gbm")]
        {
            let gbm_device = Arc::new(
                GbmDevice::open(PathBuf::from("/dev/dri/renderD128"))
                    .expect("Could not open GBM device!"),
            );
            let framepool = FramePool::new(move |stream_info: &StreamInfo| {
                <Arc<GbmDevice> as Clone>::clone(&gbm_device)
                    .new_frame(
                        Fourcc::from(stream_info.format),
                        stream_info.display_resolution.clone(),
                        stream_info.coded_resolution.clone(),
                        GbmUsage::Decode,
                    )
                    .expect("Could not allocate frame for auxiliary frame pool!")
            });
            return Ok((
                C2Decoder::ConvertingDecoderGbm(
                    backend.get_decoder(EncodedFormat::from(input_fourcc))?,
                ),
                Some(framepool),
                None,
            ));
        }
        #[cfg(all(feature = "v4l2", not(feature = "gbm")))]
        {
            let framepool = FramePool::new(move |stream_info: &StreamInfo| {
                V4l2MmapVideoFrame::new(
                    Fourcc::from(stream_info.format),
                    stream_info.display_resolution.clone(),
                )
            });
            Ok((
                C2Decoder::ConvertingDecoderV4l2(
                    backend.get_decoder(EncodedFormat::from(input_fourcc))?,
                ),
                None,
                Some(framepool),
            ))
        }
        #[cfg(not(any(feature = "gbm", feature = "v4l2")))]
        Err("No auxiliary frame backend available for converting decoder".to_string())
    }
    // Processes events from the decoder. Primarily these are frame decoded events and DRCs.
    fn check_events(&mut self) {
        loop {
            #[cfg(feature = "gbm")]
            let stream_info_gbm = match &self.decoder {
                C2Decoder::ImportingDecoder(decoder) => decoder.stream_info().map(|x| x.clone()),
                C2Decoder::ConvertingDecoderGbm(decoder) => {
                    decoder.stream_info().map(|x| x.clone())
                }
                #[cfg(feature = "v4l2")]
                C2Decoder::ConvertingDecoderV4l2(_) => None,
            };
            #[cfg(feature = "v4l2")]
            let stream_info_v4l2 = match &self.decoder {
                C2Decoder::ImportingDecoder(decoder) => decoder.stream_info().map(|x| x.clone()),
                C2Decoder::ConvertingDecoderV4l2(decoder) => {
                    decoder.stream_info().map(|x| x.clone())
                }
                #[cfg(feature = "gbm")]
                C2Decoder::ConvertingDecoderGbm(_) => None,
            };
            match &mut self.decoder {
                C2Decoder::ImportingDecoder(decoder) => match decoder.next_event() {
                    Some(DecoderEvent::FrameReady(frame)) => {
                        frame.sync().unwrap();
                        let mut job: C2DecodeJob<V> = Default::default();
                        job.output = Some(frame.video_frame());
                        (*self.work_done_cb.lock().unwrap())(job);
                    }
                    Some(DecoderEvent::FormatChanged) => {
                        let stream_info: Option<StreamInfo> = {
                            #[cfg(feature = "gbm")]
                            { stream_info_gbm.clone() }
                            #[cfg(not(feature = "gbm"))]
                            {
                                #[cfg(feature = "v4l2")]
                                { stream_info_v4l2.clone() }
                                #[cfg(not(feature = "v4l2"))]
                                { None }
                            }
                        };
                        match stream_info {
                            Some(stream_info) => {
                                (*self.framepool_hint_cb.lock().unwrap())(stream_info);
                            }
                            None => {
                                log::debug!("Could not get stream info after format change!");
                                *self.state.lock().unwrap() = C2State::C2Error;
                                (*self.error_cb.lock().unwrap())(C2Status::C2BadValue);
                            }
                        }
                    }
                    _ => break,
                },
                #[cfg(feature = "gbm")]
                C2Decoder::ConvertingDecoderGbm(decoder) => match decoder.next_event() {
                    Some(DecoderEvent::FrameReady(frame)) => {
                        frame.sync().unwrap();
                        let mut job: C2DecodeJob<V> = Default::default();
                        let mut dst_frame =
                            (*self.alloc_cb.lock().unwrap())().expect("Allocation failed!");
                        let src_frame = &*frame.video_frame();
                        if let Err(err) = convert_video_frame(src_frame, &mut dst_frame) {
                            log::debug!("Error converting VideoFrame! {err}");
                            *self.state.lock().unwrap() = C2State::C2Error;
                            (*self.error_cb.lock().unwrap())(C2Status::C2BadValue);
                        }
                        job.output = Some(Arc::new(dst_frame));
                        (*self.work_done_cb.lock().unwrap())(job);
                    }
                    Some(DecoderEvent::FormatChanged) => match stream_info_gbm.clone() {
                        Some(stream_info) => {
                            (*self.framepool_hint_cb.lock().unwrap())(stream_info.clone());
                            self.auxiliary_frame_pool_gbm.as_mut().unwrap().resize(&stream_info);
                        }
                        None => {
                            log::debug!("Could not get stream info after format change!");
                            *self.state.lock().unwrap() = C2State::C2Error;
                            (*self.error_cb.lock().unwrap())(C2Status::C2BadValue);
                        }
                    },
                    _ => break,
                },
                #[cfg(feature = "v4l2")]
                C2Decoder::ConvertingDecoderV4l2(decoder) => match decoder.next_event() {
                    Some(DecoderEvent::FrameReady(frame)) => {
                        frame.sync().unwrap();
                        let mut job: C2DecodeJob<V> = Default::default();
                        let mut dst_frame =
                            (*self.alloc_cb.lock().unwrap())().expect("Allocation failed!");
                        let src_frame = &*frame.video_frame();
                        if let Err(err) = convert_video_frame(src_frame, &mut dst_frame) {
                            log::debug!("Error converting VideoFrame! {err}");
                            *self.state.lock().unwrap() = C2State::C2Error;
                            (*self.error_cb.lock().unwrap())(C2Status::C2BadValue);
                        }
                        job.output = Some(Arc::new(dst_frame));
                        (*self.work_done_cb.lock().unwrap())(job);
                    }
                    Some(DecoderEvent::FormatChanged) => match stream_info_v4l2.clone() {
                        Some(stream_info) => {
                            (*self.framepool_hint_cb.lock().unwrap())(stream_info.clone());
                            self.auxiliary_frame_pool_v4l2.as_mut().unwrap().resize(&stream_info);
                        }
                        None => {
                            log::debug!("Could not get stream info after format change!");
                            *self.state.lock().unwrap() = C2State::C2Error;
                            (*self.error_cb.lock().unwrap())(C2Status::C2BadValue);
                        }
                    },
                    _ => break,
                },
            }
        }
    }
}

impl<V, B> C2Worker<C2DecodeJob<V>> for C2DecoderWorker<V, B>
where
    V: VideoFrame,
    B: C2DecoderBackend,
{
    type Options = <B as C2DecoderBackend>::DecoderOptions;

    fn new(
        input_fourcc: Fourcc,
        output_fourcc: Fourcc,
        awaiting_job_event: Arc<EventFd>,
        error_cb: Arc<Mutex<dyn FnMut(C2Status) + Send + 'static>>,
        work_done_cb: Arc<Mutex<dyn FnMut(C2DecodeJob<V>) + Send + 'static>>,
        work_queue: Arc<Mutex<VecDeque<C2DecodeJob<V>>>>,
        state: Arc<Mutex<C2State>>,
        framepool_hint_cb: Arc<Mutex<dyn FnMut(StreamInfo) + Send + 'static>>,
        alloc_cb: Arc<Mutex<dyn FnMut() -> Option<V> + Send + 'static>>,
        options: Self::Options,
    ) -> Result<Self, String> {
        let mut backend = B::new(options)?;
        let backend_fourccs = backend.supported_output_formats();
        let (decoder, auxiliary_frame_pool_gbm, auxiliary_frame_pool_v4l2) =
            if backend_fourccs.contains(&output_fourcc) {
                (
                    C2Decoder::ImportingDecoder(
                        backend.get_decoder(EncodedFormat::from(input_fourcc))?,
                    ),
                    None,
                    None,
                )
            } else {
                Self::init_converting(backend, input_fourcc)?
            };
        Ok(Self {
            decoder,
            epoll_fd: Epoll::new(EpollCreateFlags::empty())
                .map_err(C2DecoderPollErrorWrapper::Epoll)
                .unwrap(),
            awaiting_job_event,
            error_cb,
            work_done_cb,
            framepool_hint_cb: framepool_hint_cb,
            alloc_cb,
            work_queue,
            frame_num: 0,
            state,
            auxiliary_frame_pool_gbm,
            auxiliary_frame_pool_v4l2,
            _phantom: Default::default(),
        })
    }

    fn process_loop(&mut self) {
        self.epoll_fd = Epoll::new(EpollCreateFlags::empty())
            .map_err(C2DecoderPollErrorWrapper::Epoll)
            .unwrap();
        let poll_fd = match &self.decoder {
            C2Decoder::ImportingDecoder(decoder) => decoder.poll_fd(),
            #[cfg(feature = "gbm")]
            C2Decoder::ConvertingDecoderGbm(decoder) => decoder.poll_fd(),
            #[cfg(feature = "v4l2")]
            C2Decoder::ConvertingDecoderV4l2(decoder) => decoder.poll_fd(),
        };
        let _ = self
            .epoll_fd
            .add(poll_fd, EpollEvent::new(EpollFlags::EPOLLIN, 1))
            .map_err(C2DecoderPollErrorWrapper::EpollAdd);
        self.epoll_fd
            .add(self.awaiting_job_event.as_fd(), EpollEvent::new(EpollFlags::EPOLLIN, 2))
            .map_err(C2DecoderPollErrorWrapper::EpollAdd)
            .unwrap();

        while *self.state.lock().unwrap() == C2State::C2Running {
            // Poll for decoder events or pending job events.
            let mut events = [EpollEvent::empty()];
            let _nb_fds = self.epoll_fd.wait(&mut events, EpollTimeout::NONE).unwrap();

            if events == [EpollEvent::new(EpollFlags::EPOLLIN, 2)] {
                self.awaiting_job_event.read().unwrap();
            }

            // We want to try sending compressed buffers to the decoder regardless of what event
            // woke us up, because we either have new work, or we might more output buffers
            // available.
            let mut possible_job = (*self.work_queue.lock().unwrap()).pop_front();
            while let Some(mut job) = possible_job {
                if job.get_drain() != DrainMode::NoDrain {
                    let flush_result = match &mut self.decoder {
                        C2Decoder::ImportingDecoder(decoder) => decoder.flush(),
                        #[cfg(feature = "gbm")]
                        C2Decoder::ConvertingDecoderGbm(decoder) => decoder.flush(),
                        #[cfg(feature = "v4l2")]
                        C2Decoder::ConvertingDecoderV4l2(decoder) => decoder.flush(),
                    };
                    if let Err(_) = flush_result {
                        log::debug!("Error handling drain request!");
                        *self.state.lock().unwrap() = C2State::C2Error;
                        (*self.error_cb.lock().unwrap())(C2Status::C2BadValue);
                    } else {
                        if job.get_drain() == DrainMode::EOSDrain {
                            *self.state.lock().unwrap() = C2State::C2Stopped;
                        }
                        self.check_events();
                    }
                    break;
                } else {
                    let bitstream = job.input.as_slice();
                    let decode_result = match &mut self.decoder {
                        C2Decoder::ImportingDecoder(decoder) => decoder.decode(
                            self.frame_num,
                            bitstream,
                            &mut *self.alloc_cb.lock().unwrap(),
                        ),
                        #[cfg(feature = "gbm")]
                        C2Decoder::ConvertingDecoderGbm(decoder) => {
                            decoder.decode(self.frame_num, bitstream, &mut || {
                                self.auxiliary_frame_pool_gbm.as_mut().unwrap().alloc()
                            })
                        }
                        #[cfg(feature = "v4l2")]
                        C2Decoder::ConvertingDecoderV4l2(decoder) => {
                            decoder.decode(self.frame_num, bitstream, &mut || {
                                self.auxiliary_frame_pool_v4l2.as_mut().unwrap().alloc()
                            })
                        }
                    };
                    match decode_result {
                        Ok(num_bytes) => {
                            self.frame_num += 1;
                            if num_bytes != job.input.len() {
                                job.input = (&job.input[num_bytes..]).to_vec();
                                (*self.work_queue.lock().unwrap()).push_front(job);
                            }
                        }
                        Err(DecodeError::NotEnoughOutputBuffers(_) | DecodeError::CheckEvents) => {
                            (*self.work_queue.lock().unwrap()).push_front(job);
                            break;
                        }
                        Err(e) => {
                            log::debug!("Unhandled error message from decoder {e:?}");
                            *self.state.lock().unwrap() = C2State::C2Error;
                            (*self.error_cb.lock().unwrap())(C2Status::C2BadValue);
                            break;
                        }
                    }
                }
                possible_job = (*self.work_queue.lock().unwrap()).pop_front();
            }
            self.check_events();
        }
    }
}
