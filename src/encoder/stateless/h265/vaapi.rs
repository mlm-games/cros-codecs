// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

use std::rc::Rc;
use std::sync::Arc;

use anyhow::Context;
use libva::BufferType;
use libva::Display;
use libva::EncCodedBuffer;
use libva::EncPictureParameterBufferHEVC;
use libva::EncSequenceParameterBufferHEVC;
use libva::EncSliceParameterBufferHEVC;
use libva::HEVCEncPicFields;
use libva::HevcEncPicSccFields;
use libva::HevcEncSliceFields;
use libva::Picture;
use libva::PictureHEVC;
use libva::Surface;
use libva::SurfaceMemoryDescriptor;
use libva::VA_INVALID_ID;

use crate::BlockingMode;
use crate::Fourcc;
use crate::Resolution;
use crate::backend::vaapi::encoder::CodedOutputPromise;
use crate::backend::vaapi::encoder::Reconstructed;
use crate::backend::vaapi::encoder::VaapiBackend;
use crate::codec::h265::parser::Profile;
use crate::encoder::EncodeResult;
use crate::encoder::h265::EncoderConfig;
use crate::encoder::stateless::StatelessBackendResult;
use crate::encoder::stateless::StatelessVideoEncoderBackend;
use crate::encoder::stateless::h265::BackendRequest;
use crate::encoder::stateless::h265::StatelessEncoder;
use crate::encoder::stateless::h265::StatelessH265EncoderBackend;
use crate::video_frame::VideoFrame;

impl<M, H> StatelessVideoEncoderBackend<crate::encoder::h265::H265> for VaapiBackend<M, H>
where
    M: SurfaceMemoryDescriptor,
    H: std::borrow::Borrow<Surface<M>> + 'static,
{
    type Picture = H;
    type Reconstructed = Reconstructed;
    type CodedPromise = CodedOutputPromise<M, H>;
    type ReconPromise = crate::encoder::stateless::ReadyPromise<Self::Reconstructed>;
}

impl<M, H> VaapiBackend<M, H>
where
    M: SurfaceMemoryDescriptor,
    H: std::borrow::Borrow<Surface<M>> + 'static,
{
    fn build_hevc_seq_param(
        config: &EncoderConfig,
        sps: &crate::codec::h265::parser::Sps,
        bits_per_second: u32,
        intra_period: u32,
        ip_period: u32,
    ) -> BufferType {
        use libva::HEVCEncSeqFields;
        use libva::HevcEncVuiFields;
        use libva::HevcEncSeqSccFields;

        let seq_fields = HEVCEncSeqFields::new(
            sps.chroma_format_idc as u32,
            sps.separate_colour_plane_flag as u32,
            sps.bit_depth_luma_minus8 as u32,
            sps.bit_depth_chroma_minus8 as u32,
            sps.scaling_list_enabled_flag as u32,
            sps.strong_intra_smoothing_enabled_flag as u32,
            sps.amp_enabled_flag as u32,
            sps.sample_adaptive_offset_enabled_flag as u32,
            sps.pcm_enabled_flag as u32,
            sps.pcm_loop_filter_disabled_flag as u32,
            sps.temporal_mvp_enabled_flag as u32,
            0,
            0,
        );

        let vui_fields = if sps.vui_parameters_present_flag {
            Some(HevcEncVuiFields::new(
                sps.vui_parameters.aspect_ratio_info_present_flag as u32,
                0,
                0,
                sps.vui_parameters.timing_info_present_flag as u32,
                sps.vui_parameters.bitstream_restriction_flag as u32,
                0,
                sps.vui_parameters.motion_vectors_over_pic_boundaries_flag as u32,
                0,
                sps.vui_parameters.log2_max_mv_length_horizontal as u32,
                sps.vui_parameters.log2_max_mv_length_vertical as u32,
            ))
        } else {
            None
        };

        let scc_fields = HevcEncSeqSccFields::new(0);

        BufferType::EncSequenceParameter(libva::EncSequenceParameter::HEVC(
            EncSequenceParameterBufferHEVC::new(
                sps.profile_tier_level.general_profile_idc,
                sps.profile_tier_level.general_level_idc as u8,
                sps.profile_tier_level.general_tier_flag as u8,
                intra_period,
                intra_period,
                ip_period,
                bits_per_second,
                sps.pic_width_in_luma_samples,
                sps.pic_height_in_luma_samples,
                &seq_fields,
                sps.log2_min_luma_coding_block_size_minus3,
                sps.log2_diff_max_min_luma_coding_block_size,
                sps.log2_min_luma_transform_block_size_minus2,
                sps.log2_diff_max_min_luma_transform_block_size,
                sps.max_transform_hierarchy_depth_inter,
                sps.max_transform_hierarchy_depth_intra,
                sps.pcm_sample_bit_depth_luma_minus1 as u32,
                sps.pcm_sample_bit_depth_chroma_minus1 as u32,
                sps.log2_min_pcm_luma_coding_block_size_minus3 as u32,
                sps.log2_diff_max_min_pcm_luma_coding_block_size as u32,
                vui_fields,
                0,
                0,
                0,
                sps.vui_parameters.num_units_in_tick,
                sps.vui_parameters.time_scale,
                0,
                0,
                0,
                &scc_fields,
            ),
        ))
    }

    fn build_hevc_pic_param(
        &self,
        request: &BackendRequest<H, Reconstructed>,
        coded_buf: &EncCodedBuffer,
        recon: &Reconstructed,
    ) -> BufferType {
        use crate::encoder::stateless::h265::IsReference;

        let pps = &request.pps;
        let slice = &request.slice;

        let is_idr = request.is_idr;
        let coding_type = match slice.header.type_ {
            crate::codec::h265::parser::SliceType::I => 2,
            crate::codec::h265::parser::SliceType::P => 1,
            crate::codec::h265::parser::SliceType::B => 0,
        };
        let reference_flag = (request.dpb_meta.is_reference != IsReference::No) as u32;

        let pic_fields = HEVCEncPicFields::new(
            is_idr as u32,
            coding_type,
            reference_flag,
            pps.dependent_slice_segments_enabled_flag as u32,
            pps.sign_data_hiding_enabled_flag as u32,
            pps.constrained_intra_pred_flag as u32,
            pps.transform_skip_enabled_flag as u32,
            pps.cu_qp_delta_enabled_flag as u32,
            pps.weighted_pred_flag as u32,
            pps.weighted_bipred_flag as u32,
            pps.transquant_bypass_enabled_flag as u32,
            pps.tiles_enabled_flag as u32,
            pps.entropy_coding_sync_enabled_flag as u32,
            pps.loop_filter_across_tiles_enabled_flag as u32,
            pps.loop_filter_across_slices_enabled_flag as u32,
            pps.scaling_list_data_present_flag as u32,
            0,
            0,
            0,
        );
        let scc_fields = HevcEncPicSccFields::new(pps.scc_extension_flag as u16);

        // Build reference frames array (15 entries)
        let mut reference_frames: [PictureHEVC; 15] = std::array::from_fn(|_| {
            PictureHEVC::new(VA_INVALID_ID, 0, 0)
        });
        for (i, entry) in request.ref_list_0.iter().enumerate().take(15) {
            let pic = PictureHEVC::new(entry.recon_pic.surface_id(), entry.meta.poc, 0);
            reference_frames[i] = pic;
        }

        let decoded_curr_pic = PictureHEVC::new(recon.surface_id(), request.dpb_meta.poc, 0);

        let column_width_minus1 = {
            let mut arr = [0u8; 19];
            for (i, v) in pps.column_width_minus1.iter().enumerate().take(19) {
                arr[i] = *v as u8;
            }
            arr
        };
        let row_height_minus1 = {
            let mut arr = [0u8; 21];
            for (i, v) in pps.row_height_minus1.iter().enumerate().take(21) {
                arr[i] = *v as u8;
            }
            arr
        };

        let pic_param = EncPictureParameterBufferHEVC::new(
            decoded_curr_pic,
            reference_frames,
            coded_buf.id(),
            0, // collocated_ref_pic_index
            0, // last_picture
            (pps.init_qp_minus26 + 26) as u8,
            pps.diff_cu_qp_delta_depth,
            pps.cb_qp_offset,
            pps.cr_qp_offset,
            pps.num_tile_columns_minus1,
            pps.num_tile_rows_minus1,
            column_width_minus1,
            row_height_minus1,
            pps.log2_parallel_merge_level_minus2,
            64, // ctu_max_bitsize_allowed (default)
            pps.num_ref_idx_l0_default_active_minus1,
            pps.num_ref_idx_l1_default_active_minus1,
            pps.pic_parameter_set_id,
            slice.nalu.header.type_ as u8,
            &pic_fields,
            0, // hierarchical_level_plus1
            0, // va_byte_reserved
            &scc_fields,
        );
        BufferType::EncPictureParameter(libva::EncPictureParameter::HEVC(pic_param))
    }

    fn build_hevc_slice_param(
        &self,
        request: &BackendRequest<H, Reconstructed>,
    ) -> BufferType {
        let slice = &request.slice;

        let mut ref_pic_list0: [PictureHEVC; 15] = std::array::from_fn(|_| PictureHEVC::new(VA_INVALID_ID, 0, 0));
        let mut ref_pic_list1: [PictureHEVC; 15] = std::array::from_fn(|_| PictureHEVC::new(VA_INVALID_ID, 0, 0));

        for (i, entry) in request.ref_list_0.iter().enumerate().take(15) {
            ref_pic_list0[i] = PictureHEVC::new(entry.recon_pic.surface_id(), entry.meta.poc, 0);
        }
        for (i, entry) in request.ref_list_1.iter().enumerate().take(15) {
            ref_pic_list1[i] = PictureHEVC::new(entry.recon_pic.surface_id(), entry.meta.poc, 0);
        }

        let long_slice_flags = HevcEncSliceFields::new(
            1, // last_slice_of_pic
            slice.header.dependent_slice_segment_flag as u32,
            slice.header.colour_plane_id as u32,
            slice.header.temporal_mvp_enabled_flag as u32,
            slice.header.sao_luma_flag as u32,
            slice.header.sao_chroma_flag as u32,
            slice.header.num_ref_idx_active_override_flag as u32,
            slice.header.mvd_l1_zero_flag as u32,
            slice.header.cabac_init_flag as u32,
            slice.header.deblocking_filter_disabled_flag as u32,
            slice.header.loop_filter_across_slices_enabled_flag as u32,
            slice.header.collocated_from_l0_flag as u32,
        );

        let sps = &request.sps;
        let pic_width_in_ctbs = ((sps.pic_width_in_luma_samples as u32 + 63) / 64) as u32;
        let pic_height_in_ctbs = ((sps.pic_height_in_luma_samples as u32 + 63) / 64) as u32;
        let num_ctu_in_slice = pic_width_in_ctbs * pic_height_in_ctbs;

        let slice_param = EncSliceParameterBufferHEVC::new(
            slice.header.segment_address,
            num_ctu_in_slice,
            slice.header.type_ as u8,
            pps_pic_parameter_set_id(request),
            slice.header.num_ref_idx_l0_active_minus1,
            slice.header.num_ref_idx_l1_active_minus1,
            ref_pic_list0,
            ref_pic_list1,
            0, // luma_log2_weight_denom
            0, // delta_chroma_log2_weight_denom
            [0; 15],
            [0; 15],
            [[0; 2]; 15],
            [[0; 2]; 15],
            [0; 15],
            [0; 15],
            [[0; 2]; 15],
            [[0; 2]; 15],
            slice.header.five_minus_max_num_merge_cand,
            slice.header.qp_delta,
            slice.header.cb_qp_offset,
            slice.header.cr_qp_offset,
            slice.header.beta_offset_div2,
            slice.header.tc_offset_div2,
            &long_slice_flags,
            0,
            0,
        );
        BufferType::EncSliceParameter(libva::EncSliceParameter::HEVC(slice_param))
    }
}

fn pps_pic_parameter_set_id<H, R>(request: &BackendRequest<H, R>) -> u8 {
    request.pps.pic_parameter_set_id
}

impl<M, H> StatelessH265EncoderBackend for VaapiBackend<M, H>
where
    M: SurfaceMemoryDescriptor,
    H: std::borrow::Borrow<Surface<M>> + 'static,
{
    fn encode_slice(
        &mut self,
        request: BackendRequest<Self::Picture, Self::Reconstructed>,
    ) -> StatelessBackendResult<(Self::ReconPromise, Self::CodedPromise)> {
        let coded_buf = self.new_coded_buffer(&request.tunings.rate_control)?;
        let recon = self.new_scratch_picture()?;

        let bits_per_second = request.tunings.rate_control.bitrate_target().unwrap_or(0) as u32;
        // Use actual SPS dimensions and profile/level from SPS, not hardcoded values
        let seq_param = Self::build_hevc_seq_param(
            &EncoderConfig {
                resolution: crate::Resolution {
                    width: request.sps.pic_width_in_luma_samples as u32,
                    height: request.sps.pic_height_in_luma_samples as u32,
                },
                profile: Profile::try_from(request.sps.profile_tier_level.general_profile_idc)
                    .unwrap_or(Profile::Main),
                level: request.sps.profile_tier_level.general_level_idc,
                pred_structure: crate::encoder::PredictionStructure::LowDelay {
                    limit: request.intra_period as u16,
                },
                initial_tunings: request.tunings.clone(),
            },
            &request.sps,
            bits_per_second,
            request.intra_period,
            request.ip_period,
        );

        let pic_param = self.build_hevc_pic_param(&request, &coded_buf, &recon);
        let slice_param = self.build_hevc_slice_param(&request);

        let mut picture = Picture::new(
            request.dpb_meta.poc as u64,
            Rc::clone(self.context()),
            request.input,
        );

        picture.add_buffer(self.context().create_buffer(seq_param)?);
        picture.add_buffer(self.context().create_buffer(pic_param)?);
        picture.add_buffer(self.context().create_buffer(slice_param)?);

        use crate::backend::vaapi::encoder::tunings_to_libva_rc;
        let rc_param = tunings_to_libva_rc::<1, 51>(&request.tunings)?;
        let rc_param = BufferType::EncMiscParameter(libva::EncMiscParameter::RateControl(rc_param));
        picture.add_buffer(self.context().create_buffer(rc_param)?);

        let framerate_param = BufferType::EncMiscParameter(libva::EncMiscParameter::FrameRate(
            libva::EncMiscParameterFrameRate::new(request.tunings.framerate, 0),
        ));
        picture.add_buffer(self.context().create_buffer(framerate_param)?);

        let picture = picture.begin().context("picture begin")?;
        let picture = picture.render().context("picture render")?;
        let picture = picture.end().context("picture end")?;

        let references: Vec<std::rc::Rc<dyn std::any::Any>> = vec![];

        let reference_promise = crate::encoder::stateless::ReadyPromise::from(recon);
        let bitstream_promise =
            CodedOutputPromise::new(picture, references, coded_buf, request.coded_output);

        Ok((reference_promise, bitstream_promise))
    }
}

fn h265_va_profile_and_rc(
    config: &EncoderConfig,
) -> EncodeResult<(libva::VAProfile::Type, u32)> {
    let va_profile = match config.profile {
        Profile::Main => libva::VAProfile::VAProfileHEVCMain,
        Profile::Main10 => libva::VAProfile::VAProfileHEVCMain10,
        _ => return Err(crate::encoder::stateless::StatelessBackendError::UnsupportedProfile.into()),
    };
    let bitrate_control = match config.initial_tunings.rate_control {
        crate::encoder::RateControl::ConstantBitrate(_) => libva::VA_RC_CBR,
        crate::encoder::RateControl::VariableBitrate { .. } => libva::VA_RC_VBR,
        crate::encoder::RateControl::ConstantQuality(_) => libva::VA_RC_CQP,
    };
    Ok((va_profile, bitrate_control))
}

impl<V: VideoFrame> StatelessEncoder<V, VaapiBackend<V::MemDescriptor, Surface<V::MemDescriptor>>> {
    pub fn new_vaapi(
        display: Arc<Display>,
        config: EncoderConfig,
        fourcc: Fourcc,
        coded_size: Resolution,
        low_power: bool,
        blocking_mode: BlockingMode,
    ) -> EncodeResult<Self> {
        let (va_profile, bitrate_control) = h265_va_profile_and_rc(&config)?;
        let backend =
            VaapiBackend::new(display, va_profile, fourcc, coded_size, bitrate_control, low_power)?;
        Self::new_h265(backend, config, blocking_mode)
    }
}

impl<D: SurfaceMemoryDescriptor, S: std::borrow::Borrow<Surface<D>> + 'static>
    StatelessEncoder<S, VaapiBackend<D, S>>
{
    pub fn new_native_vaapi(
        display: Arc<Display>,
        config: EncoderConfig,
        fourcc: Fourcc,
        coded_size: Resolution,
        low_power: bool,
        blocking_mode: BlockingMode,
    ) -> EncodeResult<Self> {
        let (va_profile, bitrate_control) = h265_va_profile_and_rc(&config)?;
        let backend =
            VaapiBackend::new(display, va_profile, fourcc, coded_size, bitrate_control, low_power)?;
        Self::new_h265(backend, config, blocking_mode)
    }
}
