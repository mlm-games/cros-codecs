// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

use std::rc::Rc;

use crate::codec::h265::parser::Pps;
use crate::codec::h265::parser::Sps;
use crate::encoder::EncodeError;
use crate::encoder::EncodeResult;
use crate::encoder::Tunings;
use crate::encoder::h265::EncoderConfig;
use crate::encoder::stateless::FrameMetadata;
use crate::encoder::stateless::h265::BackendRequest;
use crate::encoder::stateless::h265::DpbEntry;
use crate::encoder::stateless::h265::DpbEntryMeta;
use crate::encoder::stateless::h265::IsReference;
use crate::encoder::stateless::predictor::LowDelay;
use crate::encoder::stateless::predictor::LowDelayDelegate;

pub(crate) struct LowDelayH265Delegate {
    sps: Option<Rc<Sps>>,
    pps: Option<Rc<Pps>>,
    update_param_sets: bool,
    config: EncoderConfig,
}

pub(crate) type LowDelayH265<Picture, Reference> = LowDelay<
    Picture,
    DpbEntry<Reference>,
    LowDelayH265Delegate,
    BackendRequest<Picture, Reference>,
>;

impl<Picture, Reference> LowDelayH265<Picture, Reference> {
    pub(super) fn new(config: EncoderConfig, limit: u16) -> Self {
        Self {
            queue: Default::default(),
            references: Default::default(),
            counter: 0,
            limit,
            tunings: config.initial_tunings.clone(),
            delegate: LowDelayH265Delegate {
                config,
                update_param_sets: false,
                sps: None,
                pps: None,
            },
            tunings_queue: Default::default(),
            _phantom: Default::default(),
        }
    }

    fn new_sequence(&mut self) {
        // In real hardware path, SPS/PPS would be generated via synthesizer.
        self.delegate.sps = Some(Rc::new(unsafe { std::mem::zeroed() }));
        self.delegate.pps = Some(Rc::new(unsafe { std::mem::zeroed() }));
        self.delegate.update_param_sets = true;
    }
}

impl<Picture, Reference>
    LowDelayDelegate<Picture, DpbEntry<Reference>, BackendRequest<Picture, Reference>>
    for LowDelayH265<Picture, Reference>
{
    fn request_keyframe(
        &mut self,
        input: Picture,
        input_meta: FrameMetadata,
        idr: bool,
    ) -> EncodeResult<BackendRequest<Picture, Reference>> {
        if idr {
            self.new_sequence();
        }
        let sps = self.delegate.sps.clone().ok_or(EncodeError::InvalidInternalState)?;
        let pps = self.delegate.pps.clone().ok_or(EncodeError::InvalidInternalState)?;

        let dpb_meta = DpbEntryMeta {
            poc: (self.counter as i32 * 2) & 0xffff,
            is_reference: IsReference::ShortTerm,
        };

        // Dummy slice, VAAPI backend will fill actual slice params.
        let slice = unsafe { std::mem::zeroed() };

        let request = BackendRequest {
            sps,
            pps,
            slice,
            input,
            input_meta,
            dpb_meta,
            ref_list_0: vec![],
            ref_list_1: vec![],
            intra_period: self.limit as u32,
            ip_period: 0,
            is_idr: idr,
            tunings: self.tunings.clone(),
            coded_output: Vec::new(),
        };
        Ok(request)
    }

    fn request_interframe(
        &mut self,
        input: Picture,
        input_meta: FrameMetadata,
    ) -> EncodeResult<BackendRequest<Picture, Reference>> {
        let mut ref_list_0 = vec![];
        for r in self.references.iter().rev() {
            ref_list_0.push(Rc::clone(r));
        }
        let sps = self.delegate.sps.clone().ok_or(EncodeError::InvalidInternalState)?;
        let pps = self.delegate.pps.clone().ok_or(EncodeError::InvalidInternalState)?;

        let dpb_meta = DpbEntryMeta {
            poc: (self.counter as i32 * 2) & 0xffff,
            is_reference: IsReference::ShortTerm,
        };

        let slice = unsafe { std::mem::zeroed() };

        let request = BackendRequest {
            sps,
            pps,
            slice,
            input,
            input_meta,
            dpb_meta,
            ref_list_0,
            ref_list_1: vec![],
            intra_period: self.limit as u32,
            ip_period: 0,
            is_idr: false,
            tunings: self.tunings.clone(),
            coded_output: Vec::new(),
        };
        self.references.clear();
        Ok(request)
    }

    fn try_tunings(&self, _tunings: &Tunings) -> EncodeResult<()> {
        Ok(())
    }

    fn apply_tunings(&mut self, _tunings: &Tunings) -> EncodeResult<()> {
        Ok(())
    }
}
