// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
//! H.265 parameter set synthesis.
//!
//! The stateless VA-API encoder hands the driver bit-exact parameter sets as
//! packed headers, so the client writes them. The writers mirror the parser
//! field for field (`parser::parse_vps/parse_sps/parse_pps`); syntax the
//! encoder never produces is rejected with [`SynthesizerError::Unsupported`]
//! rather than emitted half-formed.

use std::fmt;
use std::io::Write;

use crate::codec::h264::nalu_writer::NaluWriter;
use crate::codec::h264::nalu_writer::NaluWriterError;
use crate::codec::h265::parser::NaluType;
use crate::codec::h265::parser::Pps;
use crate::codec::h265::parser::ProfileTierLevel;
use crate::codec::h265::parser::ShortTermRefPicSet;
use crate::codec::h265::parser::Sps;
use crate::codec::h265::parser::Vps;

#[derive(Debug)]
pub enum SynthesizerError {
    /// The structure uses syntax this writer does not produce.
    Unsupported(&'static str),
    NaluWriter(NaluWriterError),
}

impl fmt::Display for SynthesizerError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SynthesizerError::Unsupported(what) => write!(f, "cannot synthesize {what}"),
            SynthesizerError::NaluWriter(x) => write!(f, "{x}"),
        }
    }
}

impl From<NaluWriterError> for SynthesizerError {
    fn from(err: NaluWriterError) -> Self {
        SynthesizerError::NaluWriter(err)
    }
}

pub type SynthesizerResult<T> = Result<T, SynthesizerError>;

fn unsupported<T>(what: &'static str) -> SynthesizerResult<T> {
    Err(SynthesizerError::Unsupported(what))
}

/// Annex-B start code + 2-byte NAL header (layer 0, `TemporalId` 0).
fn annexb_header(nal_type: NaluType) -> [u8; 6] {
    let header = (u16::from(nal_type as u8) << 9) | 1;
    [0x00, 0x00, 0x00, 0x01, (header >> 8) as u8, header as u8]
}

struct Writer<W: Write>(NaluWriter<W>);

impl<W: Write> Writer<W> {
    fn new(writer: W, ep_enabled: bool) -> Self {
        Self(NaluWriter::new(writer, ep_enabled))
    }

    fn u<T: Into<u32>>(&mut self, bits: usize, value: T) -> SynthesizerResult<()> {
        self.0.write_f(bits, value)?;
        Ok(())
    }

    fn ue<T: Into<u32>>(&mut self, value: T) -> SynthesizerResult<()> {
        self.0.write_ue(value)?;
        Ok(())
    }

    fn se<T: Into<i32>>(&mut self, value: T) -> SynthesizerResult<()> {
        self.0.write_se(value)?;
        Ok(())
    }

    fn raw(&mut self, bytes: &[u8]) -> SynthesizerResult<()> {
        self.0.write_raw(bytes)?;
        Ok(())
    }

    /// `rbsp_trailing_bits()`
    fn trailing(&mut self) -> SynthesizerResult<()> {
        self.0.write_f(1, 1u32)?;
        self.0.flush()?;
        Ok(())
    }
}

/// `profile_tier_level()` (H.265 7.3.3), general part only: sub-layers are
/// rejected, the encoder writes a single layer.
fn profile_tier_level<W: Write>(
    w: &mut Writer<W>,
    ptl: &ProfileTierLevel,
    max_sub_layers_minus1: u8,
) -> SynthesizerResult<()> {
    if max_sub_layers_minus1 != 0 {
        return unsupported("sub-layer profile_tier_level");
    }

    w.u(2, ptl.general_profile_space)?;
    w.u(1, ptl.general_tier_flag)?;
    w.u(5, ptl.general_profile_idc)?;
    for flag in ptl.general_profile_compatibility_flag {
        w.u(1, flag)?;
    }
    w.u(1, ptl.general_progressive_source_flag)?;
    w.u(1, ptl.general_interlaced_source_flag)?;
    w.u(1, ptl.general_non_packed_constraint_flag)?;
    w.u(1, ptl.general_frame_only_constraint_flag)?;

    let compat = &ptl.general_profile_compatibility_flag;
    let idc = ptl.general_profile_idc;
    let c = |i: usize| idc == i as u8 || compat[i];

    if (4..=11).any(c) {
        w.u(1, ptl.general_max_12bit_constraint_flag)?;
        w.u(1, ptl.general_max_10bit_constraint_flag)?;
        w.u(1, ptl.general_max_8bit_constraint_flag)?;
        w.u(1, ptl.general_max_422chroma_constraint_flag)?;
        w.u(1, ptl.general_max_420chroma_constraint_flag)?;
        w.u(1, ptl.general_max_monochrome_constraint_flag)?;
        w.u(1, ptl.general_intra_constraint_flag)?;
        w.u(1, ptl.general_one_picture_only_constraint_flag)?;
        w.u(1, ptl.general_lower_bit_rate_constraint_flag)?;
        if c(5) || c(9) || c(10) || c(11) {
            w.u(1, ptl.general_max_14bit_constraint_flag)?;
            for _ in 0..33 {
                w.u(1, false)?;
            }
        } else {
            for _ in 0..34 {
                w.u(1, false)?;
            }
        }
    } else if c(2) {
        for _ in 0..7 {
            w.u(1, false)?;
        }
        w.u(1, ptl.general_one_picture_only_constraint_flag)?;
        for _ in 0..35 {
            w.u(1, false)?;
        }
    } else {
        for _ in 0..43 {
            w.u(1, false)?;
        }
    }

    // general_inbld_flag (or its reserved bit) closes the constraint block.
    if c(1) || c(2) || c(3) || c(4) || c(5) || c(9) || c(11) {
        w.u(1, ptl.general_inbld_flag)?;
    } else {
        w.u(1, false)?;
    }

    w.u(8, ptl.general_level_idc as u32)?;
    Ok(())
}

/// `st_ref_pic_set()` for one SPS set. The deltas are stored cumulative, so
/// each `delta_poc_*_minus1` is the step from the previous entry.
fn short_term_ref_pic_set<W: Write>(
    w: &mut Writer<W>,
    st: &ShortTermRefPicSet,
    idx: usize,
) -> SynthesizerResult<()> {
    if idx != 0 {
        w.u(1, st.inter_ref_pic_set_prediction_flag)?;
        if st.inter_ref_pic_set_prediction_flag {
            return unsupported("predicted short-term reference picture sets");
        }
    }

    w.ue(st.num_negative_pics)?;
    w.ue(st.num_positive_pics)?;

    let mut prev = 0i32;
    for i in 0..st.num_negative_pics as usize {
        let delta = st.delta_poc_s0[i];
        w.ue((prev - delta - 1).max(0) as u32)?;
        w.u(1, st.used_by_curr_pic_s0[i])?;
        prev = delta;
    }
    let mut prev = 0i32;
    for i in 0..st.num_positive_pics as usize {
        let delta = st.delta_poc_s1[i];
        w.ue((delta - prev - 1).max(0) as u32)?;
        w.u(1, st.used_by_curr_pic_s1[i])?;
        prev = delta;
    }

    Ok(())
}

fn vui_parameters<W: Write>(w: &mut Writer<W>, sps: &Sps) -> SynthesizerResult<()> {
    let vui = &sps.vui_parameters;

    w.u(1, vui.aspect_ratio_info_present_flag)?;
    if vui.aspect_ratio_info_present_flag {
        w.u(8, vui.aspect_ratio_idc)?;
        if vui.aspect_ratio_idc == 255 {
            w.u(16, vui.sar_width)?;
            w.u(16, vui.sar_height)?;
        }
    }

    w.u(1, vui.overscan_info_present_flag)?;
    if vui.overscan_info_present_flag {
        w.u(1, vui.overscan_appropriate_flag)?;
    }

    w.u(1, vui.video_signal_type_present_flag)?;
    if vui.video_signal_type_present_flag {
        w.u(3, vui.video_format)?;
        w.u(1, vui.video_full_range_flag)?;
        w.u(1, vui.colour_description_present_flag)?;
        if vui.colour_description_present_flag {
            w.u(8, vui.colour_primaries)?;
            w.u(8, vui.transfer_characteristics)?;
            w.u(8, vui.matrix_coeffs)?;
        }
    }

    w.u(1, vui.chroma_loc_info_present_flag)?;
    if vui.chroma_loc_info_present_flag {
        w.ue(vui.chroma_sample_loc_type_top_field)?;
        w.ue(vui.chroma_sample_loc_type_bottom_field)?;
    }

    w.u(1, vui.neutral_chroma_indication_flag)?;
    w.u(1, vui.field_seq_flag)?;
    w.u(1, vui.frame_field_info_present_flag)?;

    w.u(1, vui.default_display_window_flag)?;
    if vui.default_display_window_flag {
        w.ue(vui.def_disp_win_left_offset)?;
        w.ue(vui.def_disp_win_right_offset)?;
        w.ue(vui.def_disp_win_top_offset)?;
        w.ue(vui.def_disp_win_bottom_offset)?;
    }

    w.u(1, vui.timing_info_present_flag)?;
    if vui.timing_info_present_flag {
        w.u(32, vui.num_units_in_tick)?;
        w.u(32, vui.time_scale)?;
        w.u(1, vui.poc_proportional_to_timing_flag)?;
        if vui.poc_proportional_to_timing_flag {
            w.ue(vui.num_ticks_poc_diff_one_minus1)?;
        }
        w.u(1, vui.hrd_parameters_present_flag)?;
        if vui.hrd_parameters_present_flag {
            return unsupported("VUI HRD parameters");
        }
    }

    w.u(1, vui.bitstream_restriction_flag)?;
    if vui.bitstream_restriction_flag {
        w.u(1, vui.tiles_fixed_structure_flag)?;
        w.u(1, vui.motion_vectors_over_pic_boundaries_flag)?;
        w.u(1, vui.restricted_ref_pic_lists_flag)?;
        w.ue(vui.min_spatial_segmentation_idc)?;
        w.ue(vui.max_bytes_per_pic_denom)?;
        w.ue(vui.max_bits_per_min_cu_denom)?;
        w.ue(vui.log2_max_mv_length_horizontal)?;
        w.ue(vui.log2_max_mv_length_vertical)?;
    }

    Ok(())
}

/// Writes `vps` (with start code) to `writer`.
pub fn write_vps<W: Write>(vps: &Vps, writer: W, ep_enabled: bool) -> SynthesizerResult<()> {
    let mut w = Writer::new(writer, ep_enabled);
    w.raw(&annexb_header(NaluType::VpsNut))?;

    w.u(4, vps.video_parameter_set_id)?;
    w.u(1, vps.base_layer_internal_flag)?;
    w.u(1, vps.base_layer_available_flag)?;
    w.u(6, vps.max_layers_minus1)?;
    w.u(3, vps.max_sub_layers_minus1)?;
    w.u(1, vps.temporal_id_nesting_flag)?;
    w.u(16, 0xffffu32)?;
    profile_tier_level(&mut w, &vps.profile_tier_level, vps.max_sub_layers_minus1)?;

    w.u(1, vps.sub_layer_ordering_info_present_flag)?;
    let start = if vps.sub_layer_ordering_info_present_flag {
        0
    } else {
        vps.max_sub_layers_minus1 as usize
    };
    for i in start..=vps.max_sub_layers_minus1 as usize {
        w.ue(vps.max_dec_pic_buffering_minus1[i])?;
        w.ue(vps.max_num_reorder_pics[i])?;
        w.ue(vps.max_latency_increase_plus1[i])?;
    }

    w.u(6, vps.max_layer_id)?;
    w.ue(vps.num_layer_sets_minus1)?;
    if vps.num_layer_sets_minus1 != 0 {
        return unsupported("VPS layer sets");
    }

    w.u(1, vps.timing_info_present_flag)?;
    if vps.timing_info_present_flag {
        return unsupported("VPS timing info");
    }
    w.u(1, vps.extension_flag)?;
    if vps.extension_flag {
        return unsupported("VPS extensions");
    }

    w.trailing()
}

/// Writes `sps` (with start code) to `writer`.
pub fn write_sps<W: Write>(sps: &Sps, writer: W, ep_enabled: bool) -> SynthesizerResult<()> {
    let mut w = Writer::new(writer, ep_enabled);
    w.raw(&annexb_header(NaluType::SpsNut))?;

    w.u(4, sps.video_parameter_set_id)?;
    w.u(3, sps.max_sub_layers_minus1)?;
    w.u(1, sps.temporal_id_nesting_flag)?;
    profile_tier_level(&mut w, &sps.profile_tier_level, sps.max_sub_layers_minus1)?;

    w.ue(sps.seq_parameter_set_id)?;
    w.ue(sps.chroma_format_idc)?;
    if sps.chroma_format_idc == 3 {
        w.u(1, sps.separate_colour_plane_flag)?;
    }
    w.ue(sps.pic_width_in_luma_samples)?;
    w.ue(sps.pic_height_in_luma_samples)?;
    w.u(1, sps.conformance_window_flag)?;
    if sps.conformance_window_flag {
        w.ue(sps.conf_win_left_offset)?;
        w.ue(sps.conf_win_right_offset)?;
        w.ue(sps.conf_win_top_offset)?;
        w.ue(sps.conf_win_bottom_offset)?;
    }
    w.ue(sps.bit_depth_luma_minus8)?;
    w.ue(sps.bit_depth_chroma_minus8)?;
    w.ue(sps.log2_max_pic_order_cnt_lsb_minus4)?;

    w.u(1, sps.sub_layer_ordering_info_present_flag)?;
    let start = if sps.sub_layer_ordering_info_present_flag {
        0
    } else {
        sps.max_sub_layers_minus1 as usize
    };
    for i in start..=sps.max_sub_layers_minus1 as usize {
        w.ue(sps.max_dec_pic_buffering_minus1[i])?;
        w.ue(sps.max_num_reorder_pics[i])?;
        w.ue(sps.max_latency_increase_plus1[i])?;
    }

    w.ue(sps.log2_min_luma_coding_block_size_minus3)?;
    w.ue(sps.log2_diff_max_min_luma_coding_block_size)?;
    w.ue(sps.log2_min_luma_transform_block_size_minus2)?;
    w.ue(sps.log2_diff_max_min_luma_transform_block_size)?;
    w.ue(sps.max_transform_hierarchy_depth_inter)?;
    w.ue(sps.max_transform_hierarchy_depth_intra)?;

    w.u(1, sps.scaling_list_enabled_flag)?;
    if sps.scaling_list_enabled_flag {
        return unsupported("SPS scaling lists");
    }
    w.u(1, sps.amp_enabled_flag)?;
    w.u(1, sps.sample_adaptive_offset_enabled_flag)?;
    w.u(1, sps.pcm_enabled_flag)?;
    if sps.pcm_enabled_flag {
        return unsupported("SPS PCM");
    }

    w.ue(sps.num_short_term_ref_pic_sets)?;
    for (idx, st) in sps.short_term_ref_pic_set.iter().enumerate() {
        short_term_ref_pic_set(&mut w, st, idx)?;
    }
    w.u(1, sps.long_term_ref_pics_present_flag)?;
    if sps.long_term_ref_pics_present_flag {
        return unsupported("SPS long-term reference pictures");
    }
    w.u(1, sps.temporal_mvp_enabled_flag)?;
    w.u(1, sps.strong_intra_smoothing_enabled_flag)?;

    w.u(1, sps.vui_parameters_present_flag)?;
    if sps.vui_parameters_present_flag {
        vui_parameters(&mut w, sps)?;
    }

    w.u(1, sps.extension_present_flag)?;
    if sps.extension_present_flag {
        return unsupported("SPS extensions");
    }

    w.trailing()
}

/// Writes `pps` (with start code) to `writer`.
pub fn write_pps<W: Write>(pps: &Pps, writer: W, ep_enabled: bool) -> SynthesizerResult<()> {
    let mut w = Writer::new(writer, ep_enabled);
    w.raw(&annexb_header(NaluType::PpsNut))?;

    w.ue(pps.pic_parameter_set_id)?;
    w.ue(pps.seq_parameter_set_id)?;
    w.u(1, pps.dependent_slice_segments_enabled_flag)?;
    w.u(1, pps.output_flag_present_flag)?;
    w.u(3, pps.num_extra_slice_header_bits)?;
    w.u(1, pps.sign_data_hiding_enabled_flag)?;
    w.u(1, pps.cabac_init_present_flag)?;
    w.ue(pps.num_ref_idx_l0_default_active_minus1)?;
    w.ue(pps.num_ref_idx_l1_default_active_minus1)?;
    w.se(pps.init_qp_minus26)?;
    w.u(1, pps.constrained_intra_pred_flag)?;
    w.u(1, pps.transform_skip_enabled_flag)?;
    w.u(1, pps.cu_qp_delta_enabled_flag)?;
    if pps.cu_qp_delta_enabled_flag {
        w.ue(pps.diff_cu_qp_delta_depth)?;
    }
    w.se(pps.cb_qp_offset)?;
    w.se(pps.cr_qp_offset)?;
    w.u(1, pps.slice_chroma_qp_offsets_present_flag)?;
    w.u(1, pps.weighted_pred_flag)?;
    w.u(1, pps.weighted_bipred_flag)?;
    w.u(1, pps.transquant_bypass_enabled_flag)?;
    w.u(1, pps.tiles_enabled_flag)?;
    if pps.tiles_enabled_flag {
        return unsupported("PPS tiles");
    }
    w.u(1, pps.entropy_coding_sync_enabled_flag)?;
    w.u(1, pps.loop_filter_across_slices_enabled_flag)?;
    w.u(1, pps.deblocking_filter_control_present_flag)?;
    if pps.deblocking_filter_control_present_flag {
        w.u(1, pps.deblocking_filter_override_enabled_flag)?;
        w.u(1, pps.deblocking_filter_disabled_flag)?;
        if !pps.deblocking_filter_disabled_flag {
            w.se(pps.beta_offset_div2)?;
            w.se(pps.tc_offset_div2)?;
        }
    }
    w.u(1, pps.scaling_list_data_present_flag)?;
    if pps.scaling_list_data_present_flag {
        return unsupported("PPS scaling lists");
    }
    w.u(1, pps.lists_modification_present_flag)?;
    w.ue(pps.log2_parallel_merge_level_minus2)?;
    w.u(1, pps.slice_segment_header_extension_present_flag)?;
    w.u(1, pps.extension_present_flag)?;
    if pps.extension_present_flag {
        return unsupported("PPS extensions");
    }

    w.trailing()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::rc::Rc;

    use crate::codec::h265::parser::Level;
    use crate::codec::h265::parser::Nalu;
    use crate::codec::h265::parser::Parser;
    use crate::codec::h265::parser::Profile;
    use crate::codec::h265::parser::ProfileTierLevel;
    use crate::codec::h265::parser::ShortTermRefPicSet;
    use crate::codec::h265::parser::VuiParams;

    fn round_trip() {
        let ptl = ProfileTierLevel {
            general_profile_idc: Profile::Main as u8,
            general_level_idc: Level::L4,
            general_progressive_source_flag: true,
            general_frame_only_constraint_flag: true,
            ..Default::default()
        };

        let vps = Vps {
            video_parameter_set_id: 0,
            base_layer_internal_flag: true,
            base_layer_available_flag: true,
            max_layers_minus1: 0,
            max_sub_layers_minus1: 0,
            temporal_id_nesting_flag: true,
            profile_tier_level: ptl.clone(),
            max_dec_pic_buffering_minus1: [1, 0, 0, 0, 0, 0, 0],
            ..Default::default()
        };

        let sps = Sps {
            video_parameter_set_id: 0,
            max_sub_layers_minus1: 0,
            temporal_id_nesting_flag: true,
            profile_tier_level: ptl,
            seq_parameter_set_id: 0,
            chroma_format_idc: 1,
            pic_width_in_luma_samples: 640,
            pic_height_in_luma_samples: 360,
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
            log2_max_pic_order_cnt_lsb_minus4: 4,
            max_dec_pic_buffering_minus1: [1, 0, 0, 0, 0, 0, 0],
            log2_min_luma_coding_block_size_minus3: 0,
            log2_diff_max_min_luma_coding_block_size: 3,
            log2_min_luma_transform_block_size_minus2: 0,
            log2_diff_max_min_luma_transform_block_size: 3,
            max_transform_hierarchy_depth_inter: 2,
            max_transform_hierarchy_depth_intra: 2,
            amp_enabled_flag: true,
            sample_adaptive_offset_enabled_flag: true,
            temporal_mvp_enabled_flag: true,
            strong_intra_smoothing_enabled_flag: true,
            vui_parameters_present_flag: true,
            vui_parameters: VuiParams {
                video_signal_type_present_flag: true,
                video_format: 5,
                video_full_range_flag: false,
                colour_description_present_flag: true,
                colour_primaries: 2,
                transfer_characteristics: 2,
                matrix_coeffs: 1,
                ..Default::default()
            },
            num_short_term_ref_pic_sets: 1,
            short_term_ref_pic_set: vec![{
                let mut st = ShortTermRefPicSet::default();
                st.num_negative_pics = 1;
                st.delta_poc_s0[0] = -1;
                st.used_by_curr_pic_s0[0] = true;
                st.num_delta_pocs = 1;
                st
            }],
            min_cb_log2_size_y: 3,
            ctb_log2_size_y: 6,
            ctb_size_y: 64,
            pic_height_in_ctbs_y: 6,
            pic_width_in_ctbs_y: 10,
            pic_size_in_ctbs_y: 60,
            max_tb_log2_size_y: 5,
            pic_size_in_samples_y: 640 * 360,
            chroma_array_type: 1,
            wp_offset_half_range_y: 128,
            wp_offset_half_range_c: 128,
            ..Default::default()
        };

        // VPS
        let mut vps_bytes = Vec::new();
        write_vps(&vps, &mut vps_bytes, false).expect("write vps");
        std::fs::write("/tmp/opencode/vps_dump.bin", &vps_bytes).unwrap();
        let mut parser = Parser::default();
        let parsed = {
            let mut cursor = Cursor::new(&vps_bytes[..]);
            let nalu = Nalu::next(&mut cursor).expect("vps nalu");
            parser.parse_vps(&nalu).expect("parse vps").clone()
        };
        assert_eq!(parsed, vps, "VPS round-trip");

        // SPS
        let mut sps_bytes = Vec::new();
        write_sps(&sps, &mut sps_bytes, true).expect("write sps");
        let mut parser = Parser::default();
        let parsed_sps_rc = {
            let mut cursor = Cursor::new(&sps_bytes[..]);
            let nalu = Nalu::next(&mut cursor).expect("sps nalu");
            Rc::new(parser.parse_sps(&nalu).expect("parse sps").clone())
        };
        assert_eq!(*parsed_sps_rc, sps, "SPS round-trip");
        assert_eq!(parsed_sps_rc.vui_parameters.matrix_coeffs, 1, "VUI colour is carried");

        // PPS (built against the parsed SPS, like the decoder would)
        let pps = Pps {
            pic_parameter_set_id: 0,
            seq_parameter_set_id: 0,
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
            sps: Rc::clone(&parsed_sps_rc),
        };
        let mut pps_bytes = Vec::new();
        write_pps(&pps, &mut pps_bytes, true).expect("write pps");
        let mut parser = Parser::default();
        {
            let mut c = Cursor::new(&sps_bytes[..]);
            let nalu = Nalu::next(&mut c).expect("sps nalu");
            parser.parse_sps(&nalu).expect("parse sps for pps");
        }
        let parsed_pps = {
            let mut c = Cursor::new(&pps_bytes[..]);
            let nalu = Nalu::next(&mut c).expect("pps nalu");
            parser.parse_pps(&nalu).expect("parse pps").clone()
        };
        {
            // The wire carries no scaling lists (the flag is false); the parser
            // fills the spec's flat defaults, so compare everything else.
            parsed_pps.scaling_list = Default::default();
            let a = format!("{parsed_pps:?}");
            let b = format!("{pps:?}");
            if a != b {
                let i = a.bytes().zip(b.bytes()).position(|(x, y)| x != y).unwrap_or(0);
                let lo = i.saturating_sub(80);
                let hi = (i + 120).min(a.len());
                println!("PPS differs at {i}");
                println!("parsed: ...{}...", &a[lo..hi.min(a.len())]);
                println!("expect: ...{}...", &b[lo..hi.min(b.len())]);
                panic!("PPS mismatch: {a} vs {b}");
            }
        }
    }
}
