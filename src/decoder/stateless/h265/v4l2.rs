// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

use std::cell::RefCell;
use std::rc::Rc;

use v4l2r::bindings::v4l2_ctrl_hevc_scaling_matrix;
use v4l2r::bindings::v4l2_ctrl_hevc_slice_params;
use v4l2r::controls::SafeExtControl;

use crate::Fourcc;
use crate::Rect;
use crate::Resolution;
use crate::backend::v4l2::decoder::ADDITIONAL_REFERENCE_FRAME_BUFFER;
use crate::backend::v4l2::decoder::V4l2StreamInfo;
use crate::backend::v4l2::decoder::stateless::V4l2Picture;
use crate::backend::v4l2::decoder::stateless::V4l2StatelessDecoderBackend;
use crate::backend::v4l2::decoder::stateless::V4l2StatelessDecoderHandle;
use crate::codec::h265::dpb::Dpb;
use crate::codec::h265::parser::Pps;
use crate::codec::h265::parser::Slice;
use crate::codec::h265::parser::Sps;
use crate::codec::h265::picture::PictureData;
use crate::decoder::BlockingMode;
use crate::decoder::DecodedHandle;
use crate::decoder::stateless::NewPictureError;
use crate::decoder::stateless::NewPictureResult;
use crate::decoder::stateless::NewStatelessDecoderError;
use crate::decoder::stateless::StatelessBackendResult;
use crate::decoder::stateless::StatelessDecoder;
use crate::decoder::stateless::StatelessDecoderBackend;
use crate::decoder::stateless::StatelessDecoderBackendPicture;
use crate::decoder::stateless::h265::H265;
use crate::decoder::stateless::h265::RefPicListEntry;
use crate::decoder::stateless::h265::RefPicSet;
use crate::decoder::stateless::h265::StatelessH265DecoderBackend;
use crate::device::v4l2::stateless::controls::h265::HevcDecodeParams;
use crate::device::v4l2::stateless::controls::h265::HevcPps;
use crate::device::v4l2::stateless::controls::h265::HevcScalingMatrix;
use crate::device::v4l2::stateless::controls::h265::HevcSliceParams;
use crate::device::v4l2::stateless::controls::h265::HevcSps;
use crate::device::v4l2::stateless::controls::h265::V4l2CtrlHevcDecodeParams;
use crate::device::v4l2::stateless::controls::h265::V4l2HevcDecodeMode;
use crate::device::v4l2::stateless::controls::h265::V4l2HevcDpbEntry;
use crate::device::v4l2::stateless::controls::h265::V4l2HevcStartCode;
use crate::video_frame::VideoFrame;

impl V4l2StreamInfo for &Sps {
    fn min_num_frames(&self) -> usize {
        std::cmp::min(self.max_dpb_size() as usize, 16) + ADDITIONAL_REFERENCE_FRAME_BUFFER
    }

    fn coded_size(&self) -> Resolution {
        Resolution::from((self.width() as u32, self.height() as u32))
    }

    fn visible_rect(&self) -> Rect {
        let rect = self.visible_rectangle();
        Rect {
            x: rect.min.x,
            y: rect.min.y,
            width: rect.max.x.saturating_sub(rect.min.x),
            height: rect.max.y.saturating_sub(rect.min.y),
        }
    }
}

impl<V: VideoFrame> StatelessDecoderBackendPicture<H265> for V4l2StatelessDecoderBackend<V> {
    type Picture = Rc<RefCell<V4l2Picture<V>>>;
}

impl<V: VideoFrame> StatelessH265DecoderBackend for V4l2StatelessDecoderBackend<V> {
    fn new_sequence(&mut self, sps: &Sps) -> StatelessBackendResult<()> {
        self.new_sequence(sps, Fourcc::from(b"HEVC"))
    }

    fn new_picture(
        &mut self,
        timestamp: u64,
        alloc_cb: &mut dyn FnMut() -> Option<
            <<Self as StatelessDecoderBackend>::Handle as DecodedHandle>::Frame,
        >,
    ) -> NewPictureResult<Self::Picture> {
        let frame = alloc_cb().ok_or(NewPictureError::OutOfOutputBuffers)?;
        let request_buffer = match self.device.alloc_request(timestamp, frame) {
            Ok(buffer) => buffer,
            _ => return Err(NewPictureError::OutOfOutputBuffers),
        };
        let picture = Rc::new(RefCell::new(V4l2Picture::new(request_buffer.clone())));
        request_buffer
            .as_ref()
            .borrow_mut()
            .set_picture_ref(Rc::<RefCell<V4l2Picture<V>>>::downgrade(&picture));
        Ok(picture)
    }

    fn begin_picture(
        &mut self,
        picture: &mut Self::Picture,
        picture_data: &PictureData,
        sps: &Sps,
        pps: &Pps,
        dpb: &Dpb<Self::Handle>,
        _rps: &RefPicSet<Self::Handle>,
        slice: &Slice,
    ) -> StatelessBackendResult<()> {
        let mut dpb_entries = Vec::<V4l2HevcDpbEntry>::new();
        let mut ref_pictures = Vec::<Rc<RefCell<V4l2Picture<V>>>>::new();

        for entry in dpb.entries() {
            let handle = &entry.1;
            ref_pictures.push(handle.picture.clone());
            dpb_entries.push(V4l2HevcDpbEntry {
                timestamp: handle.picture.borrow().timestamp(),
                pic: entry.0.clone(),
            });
        }

        // Also include current picture's references if needed? For HEVC the DPB already contains references.

        let hevc_sps =
            SafeExtControl::<HevcSps>::from(v4l2r::bindings::v4l2_ctrl_hevc_sps::from(sps));
        let hevc_pps =
            SafeExtControl::<HevcPps>::from(v4l2r::bindings::v4l2_ctrl_hevc_pps::from(pps));

        // Scaling matrix - kernel expects it if scaling_list_enabled_flag is set.
        let scaling_matrix = v4l2_ctrl_hevc_scaling_matrix::default();
        let hevc_scaling = SafeExtControl::<HevcScalingMatrix>::from(scaling_matrix);

        let mut decode_params = V4l2CtrlHevcDecodeParams::new();
        let mut flags: u64 = 0;
        if picture_data.nalu_type.is_irap() {
            flags |= u64::from(v4l2r::bindings::V4L2_HEVC_DECODE_PARAM_FLAG_IRAP_PIC);
        }
        if picture_data.nalu_type.is_idr() {
            flags |= u64::from(v4l2r::bindings::V4L2_HEVC_DECODE_PARAM_FLAG_IDR_PIC);
        }
        if picture_data.no_output_of_prior_pics_flag {
            flags |= u64::from(v4l2r::bindings::V4L2_HEVC_DECODE_PARAM_FLAG_NO_OUTPUT_OF_PRIOR);
        }

        decode_params
            .set_pic_order_cnt(picture_data.pic_order_cnt_val)
            .set_dpb(dpb_entries)
            .set_flags(flags)
            .set_short_term_sets(slice);

        // Convert to SafeExtControl
        let hevc_decode_params_ctrl = SafeExtControl::<HevcDecodeParams>::from(&decode_params);

        let mut picture = picture.borrow_mut();
        let request = picture.request();
        let mut request = request.as_ref().borrow_mut();

        // Collect ref pictures to keep them alive during decode
        picture.set_ref_pictures(ref_pictures);

        request
            .ioctl(hevc_sps)?
            .ioctl(hevc_pps)?
            .ioctl(hevc_scaling)?
            .ioctl(hevc_decode_params_ctrl)?
            .ioctl(V4l2HevcDecodeMode::FrameBased)?
            .ioctl(V4l2HevcStartCode::AnnexB)?;

        Ok(())
    }

    fn decode_slice(
        &mut self,
        picture: &mut Self::Picture,
        slice: &Slice,
        _sps: &Sps,
        _pps: &Pps,
        _ref_pic_list0: &[Option<RefPicListEntry<Self::Handle>>; 16],
        _ref_pic_list1: &[Option<RefPicListEntry<Self::Handle>>; 16],
    ) -> StatelessBackendResult<()> {
        // Build slice params control
        let mut slice_params = v4l2_ctrl_hevc_slice_params::default();
        // Use checked mul to avoid overflow on large slices
        slice_params.bit_size = (slice.nalu.size as u32).checked_mul(8).unwrap_or(u32::MAX);
        slice_params.data_byte_offset = 0;
        slice_params.nal_unit_type = slice.nalu.header.type_ as u8;
        slice_params.nuh_temporal_id_plus1 = slice.nalu.header.nuh_temporal_id_plus1;
        slice_params.slice_type = slice.header.type_ as u8;
        slice_params.slice_pic_order_cnt = slice.header.pic_order_cnt_lsb as i32;
        slice_params.num_ref_idx_l0_active_minus1 = slice.header.num_ref_idx_l0_active_minus1;
        slice_params.num_ref_idx_l1_active_minus1 = slice.header.num_ref_idx_l1_active_minus1;
        slice_params.collocated_ref_idx = slice.header.collocated_ref_idx;
        slice_params.five_minus_max_num_merge_cand = slice.header.five_minus_max_num_merge_cand;
        slice_params.slice_qp_delta = slice.header.qp_delta;
        slice_params.slice_cb_qp_offset = slice.header.cb_qp_offset;
        slice_params.slice_cr_qp_offset = slice.header.cr_qp_offset;
        slice_params.slice_beta_offset_div2 = slice.header.beta_offset_div2;
        slice_params.slice_tc_offset_div2 = slice.header.tc_offset_div2;

        let mut flags: u64 = 0;
        if slice.header.sao_luma_flag {
            flags |= u64::from(v4l2r::bindings::V4L2_HEVC_SLICE_PARAMS_FLAG_SLICE_SAO_LUMA);
        }
        if slice.header.sao_chroma_flag {
            flags |= u64::from(v4l2r::bindings::V4L2_HEVC_SLICE_PARAMS_FLAG_SLICE_SAO_CHROMA);
        }
        if slice.header.temporal_mvp_enabled_flag {
            flags |= u64::from(v4l2r::bindings::V4L2_HEVC_SLICE_PARAMS_FLAG_SLICE_TEMPORAL_MVP_ENABLED);
        }
        if slice.header.mvd_l1_zero_flag {
            flags |= u64::from(v4l2r::bindings::V4L2_HEVC_SLICE_PARAMS_FLAG_MVD_L1_ZERO);
        }
        if slice.header.cabac_init_flag {
            flags |= u64::from(v4l2r::bindings::V4L2_HEVC_SLICE_PARAMS_FLAG_CABAC_INIT);
        }
        if slice.header.collocated_from_l0_flag {
            flags |= u64::from(v4l2r::bindings::V4L2_HEVC_SLICE_PARAMS_FLAG_COLLOCATED_FROM_L0);
        }
        if slice.header.use_integer_mv_flag {
            flags |= u64::from(v4l2r::bindings::V4L2_HEVC_SLICE_PARAMS_FLAG_USE_INTEGER_MV);
        }
        if slice.header.deblocking_filter_disabled_flag {
            flags |= u64::from(v4l2r::bindings::V4L2_HEVC_SLICE_PARAMS_FLAG_SLICE_DEBLOCKING_FILTER_DISABLED);
        }
        if slice.header.loop_filter_across_slices_enabled_flag {
            flags |= u64::from(v4l2r::bindings::V4L2_HEVC_SLICE_PARAMS_FLAG_SLICE_LOOP_FILTER_ACROSS_SLICES_ENABLED);
        }
        if slice.header.dependent_slice_segment_flag {
            flags |= u64::from(v4l2r::bindings::V4L2_HEVC_SLICE_PARAMS_FLAG_DEPENDENT_SLICE_SEGMENT);
        }
        slice_params.flags = flags;

        let hevc_slice = SafeExtControl::<HevcSliceParams>::from(slice_params);

        let request = picture.borrow_mut().request();
        let mut request = request.as_ref().borrow_mut();

        request.ioctl(hevc_slice)?;

        const START_CODE: [u8; 3] = [0, 0, 1];
        request.write(&START_CODE);
        request.write(slice.nalu.as_ref());
        Ok(())
    }

    fn submit_picture(&mut self, picture: Self::Picture) -> StatelessBackendResult<Self::Handle> {
        let request = picture.borrow_mut().request();
        let mut request = request.as_ref().borrow_mut();
        request.submit()?;
        Ok(V4l2StatelessDecoderHandle {
            picture: picture.clone(),
            stream_info: self.stream_info.clone(),
        })
    }
}

impl<V: VideoFrame> StatelessDecoder<H265, V4l2StatelessDecoderBackend<V>> {
    pub fn new_v4l2(blocking_mode: BlockingMode) -> Result<Self, NewStatelessDecoderError> {
        Self::new(V4l2StatelessDecoderBackend::new()?, blocking_mode)
    }
}
