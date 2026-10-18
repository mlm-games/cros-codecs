// Copyright 2023 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

pub mod dpb;
pub mod nalu;
pub mod nalu_writer;
pub mod parser;
pub mod picture;
pub mod synthesizer;

use std::borrow::Cow;
use std::io::Cursor;

use crate::codec::h264::parser::Nalu;
use crate::codec::h264::parser::NaluType;
use crate::codec::h264::parser::Parser;
use crate::codec::h264::parser::Sps;
use crate::codec::h264::synthesizer::Synthesizer;

/// Colour signalling carried in the SPS VUI.
///
/// A stream without `video_signal_type` is treated as limited-range by
/// decoders; a caller that feeds full-range samples must say so or the stream
/// is mislabelled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorSignal {
    /// Samples are full-range (0..255); limited (16..235 / 16..240) when false.
    pub full_range: bool,
    /// ITU-T H.273 `matrix_coefficients`, 2 = unspecified.
    pub matrix_coefficients: u8,
}

impl Default for ColorSignal {
    fn default() -> Self {
        // What a decoder assumes when the VUI carries no signal.
        Self { full_range: false, matrix_coefficients: 2 }
    }
}

fn apply_colour(sps: &mut Sps, color: ColorSignal) {
    sps.vui_parameters_present_flag = true;
    sps.vui_parameters.video_signal_type_present_flag = true;
    sps.vui_parameters.video_format = 5; // unspecified
    sps.vui_parameters.video_full_range_flag = color.full_range;
    if color.matrix_coefficients != 2 {
        sps.vui_parameters.colour_description_present_flag = true;
        sps.vui_parameters.colour_primaries = 2; // unspecified
        sps.vui_parameters.transfer_characteristics = 2; // unspecified
        sps.vui_parameters.matrix_coefficients = color.matrix_coefficients;
    }
}

/// Adds `color` to the stream's SPS and returns the rewritten Annex-B stream,
/// or `None` when it has no SPS. Everything but the SPS NAL itself, including
/// inter-NAL padding, is copied through unchanged.
///
/// Encoders that write no VUI leave a stream at the decoder default (limited
/// range, unspecified coefficients); this relabels such streams without
/// re-encoding.
pub fn set_sps_colour(stream: &[u8], color: ColorSignal) -> Option<Vec<u8>> {
    let mut parser = Parser::default();
    let mut cursor = Cursor::new(stream);
    let mut out = Vec::with_capacity(stream.len() + 32);
    let mut copied = 0usize;
    let mut rewritten = false;

    while let Ok(nalu) = Nalu::next(&mut cursor) {
        if nalu.header.type_ != NaluType::Sps {
            continue;
        }
        let Ok(sps) = parser.parse_sps(&nalu) else {
            continue;
        };
        let mut sps = (**sps).clone();
        apply_colour(&mut sps, color);

        let mut bytes = Vec::new();
        if Synthesizer::<Sps, _>::synthesize(nalu.header.ref_idc, &sps, &mut bytes, true).is_err() {
            continue;
        }

        let start = nalu.data.as_ptr() as usize - stream.as_ptr() as usize;
        out.extend_from_slice(&stream[copied..start]);
        out.extend_from_slice(&nalu.data[..nalu.offset]); // the original start code
        out.extend_from_slice(&bytes);
        copied = start + nalu.data.len();
        rewritten = true;
    }

    if !rewritten {
        return None;
    }
    out.extend_from_slice(&stream[copied..]);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::h264::parser::Level;
    use crate::codec::h264::parser::Profile;
    use crate::codec::h264::parser::SpsBuilder;

    fn test_sps() -> Sps {
        SpsBuilder::new()
            .seq_parameter_set_id(0)
            .profile_idc(Profile::Main)
            .level_idc(Level::L4)
            .chroma_format_idc(1)
            .resolution(640, 360)
            .max_frame_num(30)
            .pic_order_cnt_type(0)
            .max_pic_order_cnt_lsb(60)
            .max_num_ref_frames(1)
            .frame_mbs_only_flag(true)
            .direct_8x8_inference_flag(true)
            .bit_depth_luma(8)
            .bit_depth_chroma(8)
            .aspect_ratio(1, 1)
            .timing_info(1, 60, false)
            .max_num_reorder_frames(0)
            .max_dec_frame_buffering(1)
            .build()
    }

    fn annex_b() -> Vec<u8> {
        let sps = test_sps();
        let mut stream = Vec::new();
        Synthesizer::<Sps, _>::synthesize(3, &sps, &mut stream, true).unwrap();
        // A PPS and a slice around it, so the copy path is covered too.
        stream.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x68, 0xCE, 0x3C, 0x80]);
        stream.extend_from_slice(&[0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x00, 0x00, 0x03, 0x02]);
        stream
    }

    #[test]
    fn rewrite_preserves_everything_but_the_sps() {
        let stream = annex_b();
        let color = ColorSignal { full_range: false, matrix_coefficients: 1 };
        let rewritten = set_sps_colour(&stream, color).expect("sps present");

        // Non-SPS bytes are preserved verbatim.
        assert!(rewritten.ends_with(&stream[stream.len() - 10..]));

        // The SPS that comes out parses to exactly the input SPS with the
        // colour fields applied.
        let mut parser = Parser::default();
        let mut cursor = Cursor::new(&rewritten[..]);
        let nalu = Nalu::next(&mut cursor).unwrap();
        let parsed = parser.parse_sps(&nalu).unwrap().clone();
        let mut expected = test_sps();
        apply_colour(&mut expected, color);
        assert_eq!(parsed, expected);
        assert!(parsed.vui_parameters.video_signal_type_present_flag);
        assert!(!parsed.vui_parameters.video_full_range_flag);
        assert_eq!(parsed.vui_parameters.matrix_coefficients, 1);

        // A stream the parser cannot see an SPS in is left alone.
        assert!(set_sps_colour(&[0x00, 0x00, 0x01, 0x41, 0x9A], color).is_none());
    }
}
