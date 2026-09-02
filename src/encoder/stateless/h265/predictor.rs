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
        // TODO: generate real SPS/PPS via H265 synthesizer (not yet available).
        // Use safe defaults instead of unsafe zeroed to avoid UB.
        let sps = Rc::new(Sps::default());
        // Minimal valid PPS referencing the SPS
        let pps = Rc::new(Pps {
            pic_parameter_set_id: 0,
            seq_parameter_set_id: sps.seq_parameter_set_id,
            dependent_slice_segments_enabled_flag: false,
            output_flag_present_flag: false,
            num_extra_slice_header_bits: 0,
            sign_data_hiding_enabled_flag: false,
            cabac_init_present_flag: false,
            num_ref_idx_l0_default_active_minus1: 0,
            num_ref_idx_l1_default_active_minus1: 0,
            init_qp_minus26: 0,
            constrained_intra_pred_flag: false,
            transform_skip_enabled_flag: false,
            cu_qp_delta_enabled_flag: false,
            diff_cu_qp_delta_depth: 0,
            cb_qp_offset: 0,
            cr_qp_offset: 0,
            slice_chroma_qp_offsets_present_flag: false,
            weighted_pred_flag: false,
            weighted_bipred_flag: false,
            transquant_bypass_enabled_flag: false,
            tiles_enabled_flag: false,
            entropy_coding_sync_enabled_flag: false,
            num_tile_columns_minus1: 0,
            num_tile_rows_minus1: 0,
            uniform_spacing_flag: true,
            column_width_minus1: [0; 19],
            row_height_minus1: [0; 21],
            loop_filter_across_tiles_enabled_flag: false,
            loop_filter_across_slices_enabled_flag: false,
            deblocking_filter_control_present_flag: false,
            deblocking_filter_override_enabled_flag: false,
            deblocking_filter_disabled_flag: false,
            beta_offset_div2: 0,
            tc_offset_div2: 0,
            scaling_list_data_present_flag: false,
            scaling_list: Default::default(),
            lists_modification_present_flag: false,
            log2_parallel_merge_level_minus2: 0,
            slice_segment_header_extension_present_flag: false,
            extension_present_flag: false,
            range_extension_flag: false,
            range_extension: Default::default(),
            scc_extension_flag: false,
            scc_extension: Default::default(),
            qp_bd_offset_y: 0,
            sps: Rc::clone(&sps),
        });
        self.delegate.sps = Some(sps);
        self.delegate.pps = Some(pps);
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

        // Dummy slice - safe empty slice (VAAPI backend ignores it for now)
        let slice = crate::codec::h265::parser::Slice {
            header: crate::codec::h265::parser::SliceHeader::default(),
            nalu: crate::codec::h264::nalu::Nalu {
                header: crate::codec::h265::parser::NaluHeader::default(),
                data: std::borrow::Cow::Borrowed(&[]),
                size: 0,
                offset: 0,
            },
        };

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

        let slice = crate::codec::h265::parser::Slice {
            header: crate::codec::h265::parser::SliceHeader::default(),
            nalu: crate::codec::h264::nalu::Nalu {
                header: crate::codec::h265::parser::NaluHeader::default(),
                data: std::borrow::Cow::Borrowed(&[]),
                size: 0,
                offset: 0,
            },
        };

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
        // Keep only last reference for low-delay (do not clear all, limit DPB)
        // Previously cleared all refs, losing reference for next P-frame.
        // Keep at most 1 reference (matching H264 predictor's single ref).
        if self.references.len() > 1 {
            let keep = self.references.len() - 1;
            self.references.drain(0..keep);
        }
        Ok(request)
    }

    fn try_tunings(&self, _tunings: &Tunings) -> EncodeResult<()> {
        Ok(())
    }

    fn apply_tunings(&mut self, _tunings: &Tunings) -> EncodeResult<()> {
        Ok(())
    }
}
