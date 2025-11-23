"""
Tests for rotation utilities.
"""

import numpy as np
import pytest

import av
from av import VideoFrame
from av.sidedata import get_rotation, set_rotation

from .common import TestCase


class TestDefaultDisplayMatrix(TestCase):
    def test_default_display_matrix_identity(self) -> None:
        """Test that by default, a video has no display matrix (identity transformation)."""
        output_path = self.sandboxed("test_default_matrix.mp4")

        output_container = av.open(output_path, mode='w', format='mp4')
        stream = output_container.add_stream('libx264', rate=30)
        stream.width = 320
        stream.height = 240
        stream.pix_fmt = 'yuv420p'

        output_container.start_encoding()

        for i in range(10):
            frame = VideoFrame(320, 240, 'yuv420p')
            for packet in stream.encode(frame):
                output_container.mux(packet)

        for packet in stream.encode():
            output_container.mux(packet)

        output_container.close()

        input_container = av.open(output_path)
        for i, decoded_frame in enumerate(input_container.decode(video=0)):
            if i >= 10:
                break

            matrix = get_rotation(decoded_frame)
            if matrix is not None:
                identity = np.eye(3, dtype=np.float64)
                np.testing.assert_allclose(matrix, identity, atol=1e-5)

        input_container.close()


def test_exif_rotation_matrices() -> None:
    """Test all 8 standard EXIF rotation matrices can be written and read back."""
    matrix_exif1 = np.eye(3, dtype=np.float64)
    matrix_exif8 = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ], dtype=np.float64)
    matrix_exif3 = matrix_exif8 @ matrix_exif8
    matrix_exif6 = matrix_exif3 @ matrix_exif8

    matrix_exif2 = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=np.float64)
    matrix_exif7 = matrix_exif2 @ matrix_exif8
    matrix_exif4 = matrix_exif7 @ matrix_exif8
    matrix_exif5 = matrix_exif4 @ matrix_exif8

    exif_matrices = {
        1: matrix_exif1,
        2: matrix_exif2,
        3: matrix_exif3,
        4: matrix_exif4,
        5: matrix_exif5,
        6: matrix_exif6,
        7: matrix_exif7,
        8: matrix_exif8,
    }

    for exif_orientation, expected_matrix in exif_matrices.items():
        frame = VideoFrame(320, 240, 'yuv420p')

        set_rotation(frame, exif_orientation)

        retrieved_matrix = get_rotation(frame)

        assert retrieved_matrix is not None, f"Failed to retrieve matrix for EXIF {exif_orientation}"
        np.testing.assert_allclose(
            retrieved_matrix,
            expected_matrix,
            atol=1e-5,
            err_msg=f"Matrix mismatch for EXIF orientation {exif_orientation}"
        )


class TestDisplayMatrixEncoding(TestCase):
    """Test display matrix preservation during encoding/decoding."""

    def test_exif_rotation_matrices_encoding(self) -> None:
        """Test that all 8 EXIF rotation matrices are preserved during encoding/decoding.

        The display matrix is set on the stream before encoding, which ensures it's
        written to the container metadata. During decoding, PyAV copies it from
        the stream's codecpar to decoded frames.
        """
        matrix_exif1 = np.eye(3, dtype=np.float64)
        matrix_exif8 = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], dtype=np.float64)
        matrix_exif3 = matrix_exif8 @ matrix_exif8
        matrix_exif6 = matrix_exif3 @ matrix_exif8

        matrix_exif2 = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=np.float64)
        matrix_exif7 = matrix_exif2 @ matrix_exif8
        matrix_exif4 = matrix_exif7 @ matrix_exif8
        matrix_exif5 = matrix_exif4 @ matrix_exif8

        exif_matrices = {
            1: matrix_exif1,
            2: matrix_exif2,
            3: matrix_exif3,
            4: matrix_exif4,
            5: matrix_exif5,
            6: matrix_exif6,
            7: matrix_exif7,
            8: matrix_exif8,
        }

        for exif_orientation, expected_matrix in exif_matrices.items():
            frame = VideoFrame(320, 240, 'yuv420p')

            output_path = self.sandboxed(f"exif_{exif_orientation}.mp4")
            output_container = av.open(output_path, mode='w', format='mp4')
            stream = output_container.add_stream('libx264', rate=30)
            stream.width = frame.width
            stream.height = frame.height
            stream.pix_fmt = 'yuv420p'

            set_rotation(stream, exif_orientation)

            output_container.start_encoding()

            for packet in stream.encode(frame):
                output_container.mux(packet)

            for packet in stream.encode():
                output_container.mux(packet)

            output_container.close()

            input_container = av.open(output_path)
            decoded_frame = next(input_container.decode(video=0))
            input_container.close()

            retrieved_matrix = get_rotation(decoded_frame)

            assert retrieved_matrix is not None, (
                f"Failed to retrieve matrix for EXIF {exif_orientation} after encoding. "
                "Display matrix should be preserved when set on stream and copied to frames during decode."
            )

            np.testing.assert_allclose(
                retrieved_matrix,
                expected_matrix,
                atol=1e-5,
                err_msg=f"Matrix mismatch for EXIF orientation {exif_orientation} after encoding"
            )

def test_set_rotation_with_matrix() -> None:
    """Test that set_rotation works with numpy matrix arrays."""
    frame = VideoFrame(320, 240, 'yuv420p')

    test_matrices = [
        np.eye(3, dtype=np.float64),
        np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], dtype=np.float64),
        np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
        ], dtype=np.float64),
    ]

    for matrix in test_matrices:
        set_rotation(frame, matrix)
        retrieved = get_rotation(frame)
        assert retrieved is not None
        np.testing.assert_allclose(matrix, retrieved, atol=1e-5)


def test_set_rotation_with_float32() -> None:
    """Test that set_rotation works with float32 arrays."""
    frame = VideoFrame(320, 240, 'yuv420p')
    matrix = np.eye(3, dtype=np.float32)
    set_rotation(frame, matrix)
    retrieved = get_rotation(frame)
    assert retrieved is not None
    np.testing.assert_allclose(matrix, retrieved, atol=1e-5)


def test_set_rotation_invalid_exif() -> None:
    """Test that set_rotation raises error for invalid EXIF orientation."""
    frame = VideoFrame(320, 240, 'yuv420p')

    with pytest.raises(ValueError, match="EXIF orientation must be 1-8"):
        set_rotation(frame, 0)

    with pytest.raises(ValueError, match="EXIF orientation must be 1-8"):
        set_rotation(frame, 9)


def test_set_rotation_invalid_matrix_shape() -> None:
    """Test that set_rotation raises error for invalid matrix shapes."""
    frame = VideoFrame(320, 240, 'yuv420p')

    with pytest.raises(ValueError, match="Matrix must be 3x3"):
        set_rotation(frame, np.eye(2))

    with pytest.raises(ValueError, match="Matrix must be 3x3"):
        set_rotation(frame, np.eye(4))


def test_set_rotation_invalid_type() -> None:
    """Test that set_rotation raises error for invalid types."""
    frame = VideoFrame(320, 240, 'yuv420p')

    with pytest.raises(TypeError, match="rotation must be"):
        set_rotation(frame, "invalid")

    with pytest.raises(TypeError, match="rotation must be"):
        set_rotation(frame, [1, 2, 3])


def test_set_rotation_stream_only() -> None:
    """Test that setting rotation on stream only (without frames) works.

    This verifies that setting rotation on the stream before encoding
    is sufficient, and it will be copied to frames during decoding.
    """
    from .common import TestCase

    output_path = TestCase().sandboxed("test_stream_only_rotation.mp4")
    output_container = av.open(output_path, mode='w', format='mp4')
    stream = output_container.add_stream('libx264', rate=30)
    stream.width = 320
    stream.height = 240
    stream.pix_fmt = 'yuv420p'

    set_rotation(stream, 3)

    output_container.start_encoding()
    frame = VideoFrame(320, 240, 'yuv420p')
    for packet in stream.encode(frame):
        output_container.mux(packet)
    for packet in stream.encode():
        output_container.mux(packet)
    output_container.close()

    input_container = av.open(output_path)
    decoded_frame = next(input_container.decode(video=0))
    input_container.close()

    matrix = get_rotation(decoded_frame)
    assert matrix is not None
    expected = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1],
    ], dtype=np.float64)
    np.testing.assert_allclose(matrix, expected, atol=1e-5)
