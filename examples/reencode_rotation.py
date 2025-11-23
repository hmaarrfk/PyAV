#!/usr/bin/env python3
"""
Re-encode the first N frames of a video with a new EXIF orientation.

This script uses PyAV to read a video file, apply a rotation matrix based on
EXIF orientation, and write the first N frames to a new video file.
"""

import click
import pathlib
import logging
import tqdm

import av
import av.logging
from av.sidedata import set_rotation


@click.command()
@click.argument('video_file', type=click.Path(exists=True, path_type=pathlib.Path))
@click.option(
    '--exif-orientation',
    type=click.IntRange(1, 8),
    default=1,
    help='EXIF orientation value (1-8), default is 1 (no rotation)'
)
@click.option(
    '--n-frames',
    type=click.IntRange(min=1),
    default=30,
    help='Number of frames to re-encode (at least 1), default is 30'
)
@click.option(
    '--force',
    is_flag=True,
    default=False,
    help='Overwrite output file if it already exists'
)
@click.option(
    '--verbose',
    is_flag=True,
    default=False,
    help='Enable verbose logging for better error messages'
)
def reencode_rotation(video_file, exif_orientation, n_frames, force, verbose):
    """Re-encode the first N frames of a video with a new EXIF orientation.

    VIDEO_FILE: Path to the input video file
    """
    if verbose:
        av.logging.set_level(av.logging.VERBOSE)
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

    if not video_file.suffix.lower() == '.mp4':
        click.echo(f"Warning: Input file is not .mp4: {video_file.suffix}", err=True)

    output_path = video_file.parent / f"{video_file.stem}_rotated_{exif_orientation}{video_file.suffix}"

    if output_path.exists() and not force:
        click.echo(f"Error: Output file already exists: {output_path}", err=True)
        click.echo("Use --force to overwrite", err=True)
        raise click.Abort()

    if output_path.exists() and force:
        click.echo(f"Output file exists, will be overwritten: {output_path}")
        output_path.unlink()

    click.echo(f"Reading video: {video_file}")
    click.echo(f"Re-encoding first {n_frames} frames with EXIF orientation {exif_orientation}")
    click.echo(f"Output will be written to: {output_path}")

    try:
        input_container = av.open(str(video_file))
        input_stream = input_container.streams.video[0]
    except Exception as e:
        click.echo(f"Error opening input file: {e}", err=True)
        raise click.Abort()

    try:
        output_container = av.open(str(output_path), mode='w', format='mp4')
    except Exception as e:
        click.echo(f"Error creating output file: {e}", err=True)
        input_container.close()
        raise click.Abort()
    output_stream = output_container.add_stream('libx264', rate=input_stream.average_rate or 30)
    output_stream.width = input_stream.width
    output_stream.height = input_stream.height
    output_stream.pix_fmt = 'yuv420p'

    set_rotation(output_stream, exif_orientation)

    output_container.start_encoding()

    frame_count = 0
    try:
        for i, frame in tqdm.tqdm(zip(
            range(n_frames),
            input_container.decode(video=0)
        ), desc="Re-encoding frames", total=n_frames):
            for packet in output_stream.encode(frame):
                output_container.mux(packet)
        for packet in output_stream.encode():
            output_container.mux(packet)
    except Exception as e:
        import traceback
        traceback.print_exc()
        click.echo(f"Error during encoding: {e}", err=True)
        try:
            output_container.close()
        except:
            pass
        try:
            input_container.close()
        except:
            pass
        if output_path.exists():
            output_path.unlink()
        raise click.Abort()
    finally:
        output_container.close()
        input_container.close()

    click.echo(f"Successfully created: {output_path}")


if __name__ == '__main__':
    reencode_rotation()
