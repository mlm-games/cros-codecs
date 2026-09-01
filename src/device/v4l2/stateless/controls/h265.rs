// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

use v4l2r::bindings::v4l2_ctrl_hevc_decode_params;
use v4l2r::bindings::v4l2_ctrl_hevc_pps;
use v4l2r::bindings::v4l2_ctrl_hevc_scaling_matrix;
use v4l2r::bindings::v4l2_ctrl_hevc_slice_params;
use v4l2r::bindings::v4l2_ctrl_hevc_sps;
use v4l2r::bindings::v4l2_hevc_dpb_entry;
use v4l2r::controls::SafeExtControl;
use v4l2r::controls::codec::VideoHEVCMaxQp;
use v4l2r::controls::codec::VideoHEVCMinQp;

use crate::codec::h265::parser::Pps;
use crate::codec::h265::parser::Slice;
use crate::codec::h265::parser::Sps;
use crate::codec::h265::picture::PictureData;
use crate::codec::h265::picture::Reference;

// Re-export control wrappers for HEVC stateless
pub struct HevcSps;
impl v4l2r::controls::ExtControlTrait for HevcSps {
    const ID: u32 = v4l2r::bindings::V4L2_CID_STATELESS_HEVC_SPS;
    type PAYLOAD = v4l2_ctrl_hevc_sps;
}

pub struct HevcPps;
impl v4l2r::controls::ExtControlTrait for HevcPps {
    const ID: u32 = v4l2r::bindings::V4L2_CID_STATELESS_HEVC_PPS;
    type PAYLOAD = v4l2_ctrl_hevc_pps;
}

pub struct HevcSliceParams;
impl v4l2r::controls::ExtControlTrait for HevcSliceParams {
    const ID: u32 = v4l2r::bindings::V4L2_CID_STATELESS_HEVC_SLICE_PARAMS;
    type PAYLOAD = v4l2_ctrl_hevc_slice_params;
}

pub struct HevcScalingMatrix;
impl v4l2r::controls::ExtControlTrait for HevcScalingMatrix {
    const ID: u32 = v4l2r::bindings::V4L2_CID_STATELESS_HEVC_SCALING_MATRIX;
    type PAYLOAD = v4l2_ctrl_hevc_scaling_matrix;
}

pub struct HevcDecodeParams;
impl v4l2r::controls::ExtControlTrait for HevcDecodeParams {
    const ID: u32 = v4l2r::bindings::V4L2_CID_STATELESS_HEVC_DECODE_PARAMS;
    type PAYLOAD = v4l2_ctrl_hevc_decode_params;
}

pub struct HevcDecodeMode;
impl v4l2r::controls::ExtControlTrait for HevcDecodeMode {
    const ID: u32 = v4l2r::bindings::V4L2_CID_STATELESS_HEVC_DECODE_MODE;
    type PAYLOAD = i32;
}

pub struct HevcStartCode;
impl v4l2r::controls::ExtControlTrait for HevcStartCode {
    const ID: u32 = v4l2r::bindings::V4L2_CID_STATELESS_HEVC_START_CODE;
    type PAYLOAD = i32;
}

impl From<&Sps> for v4l2_ctrl_hevc_sps {
    fn from(sps: &Sps) -> Self {
        // Minimal but plausible mapping; flags zeroed for now.
        // Full spec mapping would include sps flags (separate colour plane etc.)
        // For correctness on real hardware, all fields should be filled per spec.
        Self {
            video_parameter_set_id: sps.video_parameter_set_id,
            seq_parameter_set_id: sps.seq_parameter_set_id,
            pic_width_in_luma_samples: sps.pic_width_in_luma_samples,
            pic_height_in_luma_samples: sps.pic_height_in_luma_samples,
            bit_depth_luma_minus8: sps.bit_depth_luma_minus8,
            bit_depth_chroma_minus8: sps.bit_depth_chroma_minus8,
            log2_max_pic_order_cnt_lsb_minus4: sps.log2_max_pic_order_cnt_lsb_minus4,
            sps_max_dec_pic_buffering_minus1: sps.max_dec_pic_buffering_minus1[0],
            sps_max_num_reorder_pics: sps.max_num_reorder_pics[0],
            sps_max_latency_increase_plus1: sps.max_latency_increase_plus1[0],
            log2_min_luma_coding_block_size_minus3: sps.log2_min_luma_coding_block_size_minus3,
            log2_diff_max_min_luma_coding_block_size: sps.log2_diff_max_min_luma_coding_block_size,
            log2_min_luma_transform_block_size_minus2: sps
                .log2_min_luma_transform_block_size_minus2,
            log2_diff_max_min_luma_transform_block_size: sps
                .log2_diff_max_min_luma_transform_block_size,
            max_transform_hierarchy_depth_inter: sps.max_transform_hierarchy_depth_inter,
            max_transform_hierarchy_depth_intra: sps.max_transform_hierarchy_depth_intra,
            pcm_sample_bit_depth_luma_minus1: sps.pcm_sample_bit_depth_luma_minus1,
            pcm_sample_bit_depth_chroma_minus1: sps.pcm_sample_bit_depth_chroma_minus1,
            log2_min_pcm_luma_coding_block_size_minus3: sps
                .log2_min_pcm_luma_coding_block_size_minus3,
            log2_diff_max_min_pcm_luma_coding_block_size: sps
                .log2_diff_max_min_pcm_luma_coding_block_size,
            num_short_term_ref_pic_sets: sps.num_short_term_ref_pic_sets as u8,
            num_long_term_ref_pics_sps: sps.num_long_term_ref_pics_sps as u8,
            chroma_format_idc: sps.chroma_format_idc,
            sps_max_sub_layers_minus1: sps.max_sub_layers_minus1,
            flags: 0,
            ..Default::default()
        }
    }
}

impl From<&Pps> for v4l2_ctrl_hevc_pps {
    fn from(pps: &Pps) -> Self {
        let mut flags: u64 = 0;
        if pps.dependent_slice_segments_enabled_flag {
            flags |= 1 << 0;
        }
        if pps.output_flag_present_flag {
            flags |= 1 << 1;
        }
        if pps.sign_data_hiding_enabled_flag {
            flags |= 1 << 2;
        }
        if pps.cabac_init_present_flag {
            flags |= 1 << 3;
        }
        if pps.constrained_intra_pred_flag {
            flags |= 1 << 4;
        }
        if pps.transform_skip_enabled_flag {
            flags |= 1 << 5;
        }
        if pps.cu_qp_delta_enabled_flag {
            flags |= 1 << 6;
        }
        if pps.slice_chroma_qp_offsets_present_flag {
            flags |= 1 << 7;
        }
        if pps.weighted_pred_flag {
            flags |= 1 << 8;
        }
        if pps.weighted_bipred_flag {
            flags |= 1 << 9;
        }
        if pps.transquant_bypass_enabled_flag {
            flags |= 1 << 10;
        }
        if pps.tiles_enabled_flag {
            flags |= 1 << 11;
        }
        if pps.entropy_coding_sync_enabled_flag {
            flags |= 1 << 12;
        }
        if pps.loop_filter_across_tiles_enabled_flag {
            flags |= 1 << 13;
        }
        if pps.loop_filter_across_slices_enabled_flag {
            flags |= 1 << 14;
        }
        if pps.deblocking_filter_override_enabled_flag {
            flags |= 1 << 15;
        }
        if pps.deblocking_filter_disabled_flag {
            flags |= 1 << 16;
        }
        if pps.lists_modification_present_flag {
            flags |= 1 << 17;
        }
        if pps.slice_segment_header_extension_present_flag {
            flags |= 1 << 18;
        }
        if pps.deblocking_filter_control_present_flag {
            flags |= 1 << 19;
        }

        Self {
            pic_parameter_set_id: pps.pic_parameter_set_id,
            num_extra_slice_header_bits: pps.num_extra_slice_header_bits,
            num_ref_idx_l0_default_active_minus1: pps.num_ref_idx_l0_default_active_minus1,
            num_ref_idx_l1_default_active_minus1: pps.num_ref_idx_l1_default_active_minus1,
            init_qp_minus26: pps.init_qp_minus26,
            diff_cu_qp_delta_depth: pps.diff_cu_qp_delta_depth,
            pps_cb_qp_offset: pps.cb_qp_offset,
            pps_cr_qp_offset: pps.cr_qp_offset,
            num_tile_columns_minus1: pps.num_tile_columns_minus1,
            num_tile_rows_minus1: pps.num_tile_rows_minus1,
            column_width_minus1: {
                let mut arr = [0u8; 20];
                for (i, v) in pps.column_width_minus1.iter().enumerate().take(20.min(pps.column_width_minus1.len())) {
                    arr[i] = *v as u8;
                }
                arr
            },
            row_height_minus1: {
                let mut arr = [0u8; 22];
                for (i, v) in pps.row_height_minus1.iter().enumerate().take(22.min(pps.row_height_minus1.len())) {
                    arr[i] = *v as u8;
                }
                arr
            },
            pps_beta_offset_div2: pps.beta_offset_div2,
            pps_tc_offset_div2: pps.tc_offset_div2,
            log2_parallel_merge_level_minus2: pps.log2_parallel_merge_level_minus2,
            flags,
            ..Default::default()
        }
    }
}

pub struct V4l2HevcDpbEntry {
    pub timestamp: u64,
    pub pic: std::rc::Rc<std::cell::RefCell<PictureData>>,
}

impl From<&V4l2HevcDpbEntry> for v4l2_hevc_dpb_entry {
    fn from(entry: &V4l2HevcDpbEntry) -> Self {
        let pic = entry.pic.borrow();
        let flags = match pic.reference() {
            Reference::LongTerm => 0x01, // V4L2_HEVC_DPB_ENTRY_LONG_TERM_REFERENCE
            Reference::ShortTerm => 0,
            Reference::None => 0,
        };
        // field_pic is 0 for frame
        Self {
            timestamp: entry.timestamp * 1000, // us -> ns
            flags,
            field_pic: 0,
            pic_order_cnt_val: pic.pic_order_cnt_val,
            ..Default::default()
        }
    }
}

#[derive(Default)]
pub struct V4l2CtrlHevcDecodeParams {
    handle: v4l2_ctrl_hevc_decode_params,
}

impl V4l2CtrlHevcDecodeParams {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_pic_order_cnt(&mut self, poc: i32) -> &mut Self {
        self.handle.pic_order_cnt_val = poc;
        self
    }

    pub fn set_dpb(&mut self, dpb: Vec<V4l2HevcDpbEntry>) -> &mut Self {
        self.handle.num_active_dpb_entries = dpb.len() as u8;
        for (i, e) in dpb.into_iter().enumerate().take(16) {
            self.handle.dpb[i] = v4l2_hevc_dpb_entry::from(&e);
        }
        self
    }

    pub fn set_flags(&mut self, flags: u64) -> &mut Self {
        self.handle.flags = flags;
        self
    }

    pub fn set_short_term_sets(&mut self, _slice: &Slice) -> &mut Self {
        // Simplified: sizes are derived from SPS; actual RPS parsing would need full SPS ext.
        self.handle.short_term_ref_pic_set_size = 0;
        self.handle.long_term_ref_pic_set_size = 0;
        self
    }
}

impl From<&V4l2CtrlHevcDecodeParams> for SafeExtControl<HevcDecodeParams> {
    fn from(params: &V4l2CtrlHevcDecodeParams) -> Self {
        SafeExtControl::<HevcDecodeParams>::from(params.handle)
    }
}

pub enum V4l2HevcDecodeMode {
    SliceBased = 0,
    FrameBased = 1,
}

impl From<V4l2HevcDecodeMode> for SafeExtControl<HevcDecodeMode> {
    fn from(mode: V4l2HevcDecodeMode) -> Self {
        SafeExtControl::<HevcDecodeMode>::from_value(mode as i32)
    }
}

pub enum V4l2HevcStartCode {
    None = 0,
    AnnexB = 1,
}

impl From<V4l2HevcStartCode> for SafeExtControl<HevcStartCode> {
    fn from(code: V4l2HevcStartCode) -> Self {
        SafeExtControl::<HevcStartCode>::from_value(code as i32)
    }
}
