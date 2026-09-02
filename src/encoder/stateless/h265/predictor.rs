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
        // Just mirrors H264 LowDelayH264::new_sequence but for HEVC.
        let config = &self.delegate.config;
        let width = config.resolution.width as u16;
        let height = config.resolution.height as u16;

        let mut profile_tier = crate::codec::h265::parser::ProfileTierLevel::default();
        profile_tier.general_profile_idc = config.profile as u8;
        profile_tier.general_level_idc = config.level;
        profile_tier.general_tier_flag = false;
        profile_tier.general_progressive_source_flag = true;
        profile_tier.general_frame_only_constraint_flag = true;

        let sps = Rc::new(Sps {
            video_parameter_set_id: 0,
            max_sub_layers_minus1: 0,
            temporal_id_nesting_flag: true,
            profile_tier_level: profile_tier,
            seq_parameter_set_id: 0,
            chroma_format_idc: 1,
            separate_colour_plane_flag: false,
            pic_width_in_luma_samples: width,
            pic_height_in_luma_samples: height,
            conformance_window_flag: false,
            conf_win_left_offset: 0,
            conf_win_right_offset: 0,
            conf_win_top_offset: 0,
            conf_win_bottom_offset: 0,
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
            log2_max_pic_order_cnt_lsb_minus4: 4,
            sub_layer_ordering_info_present_flag: false,
            max_dec_pic_buffering_minus1: [1, 0, 0, 0, 0, 0, 0],
            max_num_reorder_pics: [0; 7],
            max_latency_increase_plus1: [0; 7],
            log2_min_luma_coding_block_size_minus3: 0,
            log2_diff_max_min_luma_coding_block_size: 3,
            log2_min_luma_transform_block_size_minus2: 0,
            log2_diff_max_min_luma_transform_block_size: 3,
            max_transform_hierarchy_depth_inter: 2,
            max_transform_hierarchy_depth_intra: 2,
            scaling_list_enabled_flag: false,
            scaling_list_data_present_flag: false,
            scaling_list: Default::default(),
            amp_enabled_flag: true,
            sample_adaptive_offset_enabled_flag: true,
            pcm_enabled_flag: false,
            pcm_sample_bit_depth_luma_minus1: 0,
            pcm_sample_bit_depth_chroma_minus1: 0,
            log2_min_pcm_luma_coding_block_size_minus3: 0,
            log2_diff_max_min_pcm_luma_coding_block_size: 0,
            pcm_loop_filter_disabled_flag: false,
            num_short_term_ref_pic_sets: 0,
            short_term_ref_pic_set: Vec::new(),
            long_term_ref_pics_present_flag: false,
            num_long_term_ref_pics_sps: 0,
            lt_ref_pic_poc_lsb_sps: [0; 32],
            used_by_curr_pic_lt_sps_flag: [false; 32],
            temporal_mvp_enabled_flag: true,
            strong_intra_smoothing_enabled_flag: true,
            vui_parameters_present_flag: false,
            vui_parameters: Default::default(),
            extension_present_flag: false,
            range_extension_flag: false,
            range_extension: Default::default(),
            scc_extension_flag: false,
            scc_extension: Default::default(),
            min_cb_log2_size_y: 3,
            ctb_log2_size_y: 6,
            ctb_size_y: 64,
            pic_height_in_ctbs_y: ((height as u32 + 63) / 64) as u32,
            pic_width_in_ctbs_y: ((width as u32 + 63) / 64) as u32,
            pic_size_in_ctbs_y: (((width as u32 + 63) / 64) * ((height as u32 + 63) / 64)) as u32,
            chroma_array_type: 1,
            wp_offset_half_range_y: 128,
            wp_offset_half_range_c: 128,
            max_tb_log2_size_y: 6,
            pic_size_in_samples_y: width as u32 * height as u32,
            vps: None,
        });

        let pps = Rc::new(Pps {
            pic_parameter_set_id: 0,
            seq_parameter_set_id: sps.seq_parameter_set_id,
            dependent_slice_segments_enabled_flag: false,
            output_flag_present_flag: false,
            num_extra_slice_header_bits: 0,
            sign_data_hiding_enabled_flag: true,
            cabac_init_present_flag: false,
            num_ref_idx_l0_default_active_minus1: 0,
            num_ref_idx_l1_default_active_minus1: 0,
            init_qp_minus26: 0,
            constrained_intra_pred_flag: false,
            transform_skip_enabled_flag: false,
            cu_qp_delta_enabled_flag: true,
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
            loop_filter_across_tiles_enabled_flag: true,
            loop_filter_across_slices_enabled_flag: false,
            deblocking_filter_control_present_flag: true,
            deblocking_filter_override_enabled_flag: false,
            deblocking_filter_disabled_flag: false,
            beta_offset_div2: 0,
            tc_offset_div2: 0,
            scaling_list_data_present_flag: false,
            scaling_list: Default::default(),
            lists_modification_present_flag: true,
            log2_parallel_merge_level_minus2: 2,
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

        let mut slice_header = crate::codec::h265::parser::SliceHeader::default();
        slice_header.type_ = crate::codec::h265::parser::SliceType::I;
        slice_header.pic_order_cnt_lsb = (dpb_meta.poc & 0xffff) as u16;
        let nalu_header = crate::codec::h265::parser::NaluHeader {
            type_: crate::codec::h265::parser::NaluType::IdrWRadl,
            nuh_layer_id: 0,
            nuh_temporal_id_plus1: 1,
        };
        let slice = crate::codec::h265::parser::Slice {
            header: slice_header,
            nalu: crate::codec::h264::nalu::Nalu {
                header: nalu_header,
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

        let mut slice_header = crate::codec::h265::parser::SliceHeader::default();
        slice_header.type_ = crate::codec::h265::parser::SliceType::P;
        slice_header.pic_order_cnt_lsb = (dpb_meta.poc & 0xffff) as u16;
        let nalu_header = crate::codec::h265::parser::NaluHeader {
            type_: crate::codec::h265::parser::NaluType::TrailR,
            nuh_layer_id: 0,
            nuh_temporal_id_plus1: 1,
        };
        let slice = crate::codec::h265::parser::Slice {
            header: slice_header,
            nalu: crate::codec::h264::nalu::Nalu {
                header: nalu_header,
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
