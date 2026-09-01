// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

use std::rc::Rc;

use crate::BlockingMode;
use crate::codec::h265::parser::Pps;
use crate::codec::h265::parser::Slice;
use crate::codec::h265::parser::Sps;
use crate::encoder::EncodeResult;
use crate::encoder::PredictionStructure;
use crate::encoder::Tunings;
use crate::encoder::h265::EncoderConfig;
use crate::encoder::h265::H265;
use crate::encoder::stateless::BackendPromise;
use crate::encoder::stateless::BitstreamPromise;
use crate::encoder::stateless::FrameMetadata;
use crate::encoder::stateless::Predictor;
use crate::encoder::stateless::StatelessBackendResult;
use crate::encoder::stateless::StatelessCodec;
use crate::encoder::stateless::StatelessEncoderBackendImport;
use crate::encoder::stateless::StatelessEncoderExecute;
use crate::encoder::stateless::StatelessVideoEncoderBackend;

#[cfg(feature = "vaapi")]
pub mod vaapi;

mod predictor;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum IsReference {
    No,
    ShortTerm,
    LongTerm,
}

#[derive(Clone, Debug)]
pub struct DpbEntryMeta {
    pub poc: i32,
    pub is_reference: IsReference,
}

pub struct DpbEntry<R> {
    pub recon_pic: R,
    pub meta: DpbEntryMeta,
}

pub struct BackendRequest<P, R> {
    pub sps: Rc<Sps>,
    pub pps: Rc<Pps>,
    pub slice: Slice<'static>,

    pub input: P,
    pub input_meta: FrameMetadata,
    pub dpb_meta: DpbEntryMeta,

    pub ref_list_0: Vec<Rc<DpbEntry<R>>>,
    pub ref_list_1: Vec<Rc<DpbEntry<R>>>,

    pub intra_period: u32,
    pub ip_period: u32,

    pub is_idr: bool,
    pub tunings: Tunings,

    pub coded_output: Vec<u8>,
}

pub struct ReferencePromise<P>
where
    P: BackendPromise,
{
    recon: P,
    dpb_meta: DpbEntryMeta,
}

impl<P> BackendPromise for ReferencePromise<P>
where
    P: BackendPromise,
{
    type Output = DpbEntry<P::Output>;

    fn is_ready(&self) -> bool {
        self.recon.is_ready()
    }

    fn sync(self) -> StatelessBackendResult<Self::Output> {
        let recon_pic = self.recon.sync()?;
        Ok(DpbEntry {
            recon_pic,
            meta: self.dpb_meta,
        })
    }
}

impl<Backend> StatelessCodec<Backend> for H265
where
    Backend: StatelessVideoEncoderBackend<H265>,
{
    type Reference = DpbEntry<Backend::Reconstructed>;
    type Request = BackendRequest<Backend::Picture, Backend::Reconstructed>;
    type CodedPromise = BitstreamPromise<Backend::CodedPromise>;
    type ReferencePromise = ReferencePromise<Backend::ReconPromise>;
}

pub trait StatelessH265EncoderBackend: StatelessVideoEncoderBackend<H265> {
    fn encode_slice(
        &mut self,
        request: BackendRequest<Self::Picture, Self::Reconstructed>,
    ) -> StatelessBackendResult<(Self::ReconPromise, Self::CodedPromise)>;
}

pub type StatelessEncoder<Handle, Backend> =
    crate::encoder::stateless::StatelessEncoder<H265, Handle, Backend>;

impl<Handle, Backend> StatelessEncoderExecute<H265, Handle, Backend> for StatelessEncoder<Handle, Backend>
where
    Backend: StatelessH265EncoderBackend,
{
    fn execute(
        &mut self,
        request: BackendRequest<Backend::Picture, Backend::Reconstructed>,
    ) -> EncodeResult<()> {
        let meta = request.input_meta.clone();
        let dpb_meta = request.dpb_meta.clone();
        self.predictor_frame_count -= 1;
        let (recon, bitstream) = self.backend.encode_slice(request)?;
        let slice_promise = BitstreamPromise { bitstream, meta };
        self.output_queue.add_promise(slice_promise);
        let ref_promise = ReferencePromise { recon, dpb_meta };
        self.recon_queue.add_promise(ref_promise);
        Ok(())
    }
}

impl<Handle, Backend> StatelessEncoder<Handle, Backend>
where
    Backend: StatelessH265EncoderBackend,
    Backend: StatelessEncoderBackendImport<Handle, Backend::Picture>,
{
    pub fn new_h265(backend: Backend, config: EncoderConfig, mode: BlockingMode) -> EncodeResult<Self> {
        let predictor: Box<dyn Predictor<_, _, _>> = match config.pred_structure {
            PredictionStructure::LowDelay { limit } => {
                Box::new(predictor::LowDelayH265::new(config, limit))
            }
        };
        Self::new(backend, mode, predictor)
    }
}
