// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

use std::rc::Rc;
use std::sync::Arc;

use anyhow::Context;
use libva::BufferType;
use libva::Display;
use libva::EncCodedBuffer;
use libva::EncSequenceParameterBufferHEVC;
use libva::Picture;
use libva::Surface;
use libva::SurfaceMemoryDescriptor;
use libva::VAProfile;

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
        let seq_param = Self::build_hevc_seq_param(
            &EncoderConfig {
                resolution: crate::Resolution {
                    width: request.sps.pic_width_in_luma_samples as u32,
                    height: request.sps.pic_height_in_luma_samples as u32,
                },
                profile: Profile::Main,
                level: crate::codec::h265::parser::Level::L4,
                pred_structure: crate::encoder::PredictionStructure::LowDelay { limit: 30 },
                initial_tunings: request.tunings.clone(),
            },
            &request.sps,
            bits_per_second,
            request.intra_period,
            request.ip_period,
        );

        let mut picture = Picture::new(
            request.dpb_meta.poc as u64,
            Rc::clone(self.context()),
            request.input,
        );

        picture.add_buffer(self.context().create_buffer(seq_param)?);

        // For now, only sequence header; picture/slice params would be added for full impl.
        // This stub allows compilation and demonstrates VAAPI H.265 path.
        // TODO: Add HEVC picture and slice params (EncPictureParameterBufferHEVC, EncSliceParameterBufferHEVC)
        // when full predictor is implemented.

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

impl<V: VideoFrame> StatelessEncoder<V, VaapiBackend<V::MemDescriptor, Surface<V::MemDescriptor>>> {
    pub fn new_vaapi(
        display: Arc<Display>,
        config: EncoderConfig,
        fourcc: Fourcc,
        coded_size: Resolution,
        low_power: bool,
        blocking_mode: BlockingMode,
    ) -> EncodeResult<Self> {
        let va_profile = match config.profile {
            Profile::Main => VAProfile::VAProfileHEVCMain,
            Profile::Main10 => VAProfile::VAProfileHEVCMain10,
            _ => return Err(crate::encoder::stateless::StatelessBackendError::UnsupportedProfile.into()),
        };

        let bitrate_control = match config.initial_tunings.rate_control {
            crate::encoder::RateControl::ConstantBitrate(_) => libva::VA_RC_CBR,
            crate::encoder::RateControl::VariableBitrate { .. } => libva::VA_RC_VBR,
            crate::encoder::RateControl::ConstantQuality(_) => libva::VA_RC_CQP,
        };

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
        let va_profile = match config.profile {
            Profile::Main => VAProfile::VAProfileHEVCMain,
            Profile::Main10 => VAProfile::VAProfileHEVCMain10,
            _ => return Err(crate::encoder::stateless::StatelessBackendError::UnsupportedProfile.into()),
        };

        let bitrate_control = match config.initial_tunings.rate_control {
            crate::encoder::RateControl::ConstantBitrate(_) => libva::VA_RC_CBR,
            crate::encoder::RateControl::VariableBitrate { .. } => libva::VA_RC_VBR,
            crate::encoder::RateControl::ConstantQuality(_) => libva::VA_RC_CQP,
        };

        let backend =
            VaapiBackend::new(display, va_profile, fourcc, coded_size, bitrate_control, low_power)?;

        Self::new_h265(backend, config, blocking_mode)
    }
}
