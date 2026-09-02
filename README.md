# nuxodecs (fork of cros-codecs)

[<img alt="crates.io" src="https://img.shields.io/crates/v/nuxodecs">](https://crates.io/crates/nuxodecs)
[<img alt="docs.rs" src="https://img.shields.io/docsrs/nuxodecs">](https://docs.rs/nuxodecs/latest/nuxodecs/)

> Published as [`nuxodecs`](https://crates.io/crates/nuxodecs), fork of [`cros-codecs`](https://github.com/chromeos/cros-codecs) for my hw-codecs crate.

Original readme below:

A lightweight, simple, low-dependency, and hopefully safe crate for
hardware-accelerated video decoding and encoding on Linux.

It is developed for use in ChromeOS (particularly
[crosvm](https://github.com/google/crosvm)), but has no dependency to ChromeOS
and should be usable anywhere.

## Current features

- Simple decoder API,
- VAAPI decoder support (using
  [cros-libva](https://github.com/mlm-games/cros-libva)) for H.264, H.265, VP8,
  VP9 and AV1,
- VAAPI encoder support for H.264, H.265, VP9 and AV1,
- Stateful V4L2 encoder support for H.264, H.265, VP8 and VP9,
- Stateless V4L2 decoder support for H.264, H.265, VP8, VP9 and AV1.

## Planned features

- Stateful V4L2 decoder support.

## Non-goals

- Support for systems other than Linux.

## Example programs

The `ccdec` example program can decode an encoded stream and write the decoded
frames to a file. As such it can be used for testing purposes.

```shell
$ cargo build --examples
$ ./target/debug/examples/ccdec --help
Usage: ccdec <input> [--output <output>] --input-format <input-format> [--output-format <output-format>] [--compute-md5 <compute-md5>]

Simple player using cros-codecs

Positional Arguments:
  input             input file

Options:
  --output          output file to write the decoded frames to
  --input-format    input format to decode from.
  --output-format   pixel format to decode into. Default: i420
  --compute-md5     whether to display the MD5 of the decoded stream, and at
                    which granularity (stream or frame)
  --help            display usage information
```

## Testing

Fluster can be used for testing, using the `ccdec` example program described
above. [This branch](https://github.com/Gnurou/fluster/tree/cros-codecs)
contains support for cros-codecs testing. Just make sure the `ccdec` binary is
in your `PATH`, and run Fluster using one of the `ccdec` decoders, e.g.

```shell
python fluster.py run -d ccdec-H.264 -ts JVT-AVC_V1
```

## Credits

The majority of the code in the initial commit has been written by Daniel
Almeida as a VAAPI backend for crosvm, before being split into this crate.
