"""
Tests for side data functionality underlying display matrix feature.
"""

import numpy as np
import pytest

import av
from av import VideoFrame
from av.sidedata.sidedata import Type, get_rotation_from_stream, set_rotation_on_stream
from av.sidedata.display import _matrix_to_bytes

from .common import TestCase


class TestSideDataContainer(TestCase):
    """Test SideDataContainer.add() method."""

    def test_add_new_side_data(self) -> None:
        """Test adding new side data to a frame."""
        frame = VideoFrame(320, 240, 'yuv420p')
        test_data = b'test side data'

        frame.side_data.add(Type.DISPLAYMATRIX, test_data)

        assert len(frame.side_data) == 1
        assert Type.DISPLAYMATRIX in frame.side_data
        assert bytes(frame.side_data[Type.DISPLAYMATRIX]) == test_data

    def test_add_side_data_string_type(self) -> None:
        """Test adding side data using string type name."""
        frame = VideoFrame(320, 240, 'yuv420p')
        test_data = b'test data'

        frame.side_data.add('DISPLAYMATRIX', test_data)

        assert Type.DISPLAYMATRIX in frame.side_data
        assert bytes(frame.side_data['DISPLAYMATRIX']) == test_data

    def test_replace_existing_side_data(self) -> None:
        """Test replacing existing side data with same size."""
        frame = VideoFrame(320, 240, 'yuv420p')
        initial_data = b'initial data'
        replacement_data = b'replacement '

        frame.side_data.add(Type.DISPLAYMATRIX, initial_data)
        assert bytes(frame.side_data[Type.DISPLAYMATRIX]) == initial_data

        frame.side_data.add(Type.DISPLAYMATRIX, replacement_data)
        assert len(frame.side_data) == 1
        assert bytes(frame.side_data[Type.DISPLAYMATRIX]) == replacement_data

    def test_replace_side_data_different_size_error(self) -> None:
        """Test that replacing with different size raises ValueError."""
        frame = VideoFrame(320, 240, 'yuv420p')
        initial_data = b'initial'
        replacement_data = b'replacement data'

        frame.side_data.add(Type.DISPLAYMATRIX, initial_data)

        with pytest.raises(ValueError, match="Existing side data has size"):
            frame.side_data.add(Type.DISPLAYMATRIX, replacement_data)

    def test_add_invalid_type_string(self) -> None:
        """Test that invalid type string raises ValueError."""
        frame = VideoFrame(320, 240, 'yuv420p')

        with pytest.raises(ValueError, match="Unknown side data type"):
            frame.side_data.add('INVALID_TYPE', b'data')

    def test_add_invalid_type_object(self) -> None:
        """Test that invalid type object raises TypeError."""
        frame = VideoFrame(320, 240, 'yuv420p')

        with pytest.raises(TypeError, match="data_type must be Type enum or string"):
            frame.side_data.add(123, b'data')

    def test_add_invalid_data_type(self) -> None:
        """Test that non-bytes data raises TypeError."""
        frame = VideoFrame(320, 240, 'yuv420p')

        with pytest.raises(TypeError, match="data_bytes must be bytes"):
            frame.side_data.add(Type.DISPLAYMATRIX, "not bytes")

    def test_add_multiple_side_data_types(self) -> None:
        """Test adding multiple different side data types."""
        frame = VideoFrame(320, 240, 'yuv420p')

        frame.side_data.add(Type.DISPLAYMATRIX, b'matrix data' * 3)
        frame.side_data.add('REPLAYGAIN', b'replaygain data')

        assert len(frame.side_data) == 2
        assert Type.DISPLAYMATRIX in frame.side_data
        assert 'REPLAYGAIN' in frame.side_data

    def test_refresh_after_add(self) -> None:
        """Test that _refresh() updates the container after adding."""
        frame = VideoFrame(320, 240, 'yuv420p')

        assert len(frame.side_data) == 0

        frame.side_data.add(Type.DISPLAYMATRIX, b'test' * 9)
        assert len(frame.side_data) == 1
        assert Type.DISPLAYMATRIX in frame.side_data

        frame.side_data.add('REPLAYGAIN', b'test2')
        assert len(frame.side_data) == 2


class TestStreamSideData(TestCase):
    """Test stream-level side data functions."""

    def test_set_rotation_on_stream(self) -> None:
        """Test setting rotation on a stream."""
        output_path = self.sandboxed("test_stream_side_data.mp4")
        output_container = av.open(output_path, mode='w', format='mp4')
        stream = output_container.add_stream('libx264', rate=30)
        stream.width = 320
        stream.height = 240
        stream.pix_fmt = 'yuv420p'

        matrix_bytes = _matrix_to_bytes(np.eye(3, dtype=np.float64))
        set_rotation_on_stream(stream, matrix_bytes)

        retrieved = get_rotation_from_stream(stream)
        assert retrieved is not None
        np.testing.assert_allclose(retrieved, np.eye(3), atol=1e-5)

        output_container.close()

    def test_get_rotation_from_stream_none(self) -> None:
        """Test getting rotation from stream with no rotation returns None."""
        output_path = self.sandboxed("test_stream_no_rotation.mp4")
        output_container = av.open(output_path, mode='w', format='mp4')
        stream = output_container.add_stream('libx264', rate=30)
        stream.width = 320
        stream.height = 240
        stream.pix_fmt = 'yuv420p'

        retrieved = get_rotation_from_stream(stream)
        assert retrieved is None

        output_container.close()

    def test_set_rotation_on_stream_replace(self) -> None:
        """Test replacing existing rotation on stream."""
        output_path = self.sandboxed("test_stream_replace.mp4")
        output_container = av.open(output_path, mode='w', format='mp4')
        stream = output_container.add_stream('libx264', rate=30)
        stream.width = 320
        stream.height = 240
        stream.pix_fmt = 'yuv420p'

        matrix1 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        matrix_bytes1 = _matrix_to_bytes(matrix1)
        set_rotation_on_stream(stream, matrix_bytes1)

        retrieved1 = get_rotation_from_stream(stream)
        np.testing.assert_allclose(retrieved1, matrix1, atol=1e-5)

        matrix2 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
        matrix_bytes2 = _matrix_to_bytes(matrix2)
        set_rotation_on_stream(stream, matrix_bytes2)

        retrieved2 = get_rotation_from_stream(stream)
        np.testing.assert_allclose(retrieved2, matrix2, atol=1e-5)

        output_container.close()

    def test_set_rotation_on_stream_wrong_size_error(self) -> None:
        """Test that setting rotation with wrong size raises ValueError."""
        output_path = self.sandboxed("test_stream_wrong_size.mp4")
        output_container = av.open(output_path, mode='w', format='mp4')
        stream = output_container.add_stream('libx264', rate=30)
        stream.width = 320
        stream.height = 240
        stream.pix_fmt = 'yuv420p'

        matrix_bytes = _matrix_to_bytes(np.eye(3, dtype=np.float64))
        set_rotation_on_stream(stream, matrix_bytes)

        wrong_size_data = b'wrong size'
        with pytest.raises(ValueError, match="Existing side data has size"):
            set_rotation_on_stream(stream, wrong_size_data)

        output_container.close()

    def test_set_rotation_on_stream_invalid_type(self) -> None:
        """Test that setting rotation on non-stream raises TypeError."""
        frame = VideoFrame(320, 240, 'yuv420p')
        matrix_bytes = _matrix_to_bytes(np.eye(3, dtype=np.float64))

        with pytest.raises(TypeError, match="stream must be a Stream"):
            set_rotation_on_stream(frame, matrix_bytes)

    def test_set_rotation_on_stream_preserved_after_start_encoding(self) -> None:
        """Test that rotation set on stream is preserved after start_encoding."""
        output_path = self.sandboxed("test_stream_preserved.mp4")
        output_container = av.open(output_path, mode='w', format='mp4')
        stream = output_container.add_stream('libx264', rate=30)
        stream.width = 320
        stream.height = 240
        stream.pix_fmt = 'yuv420p'

        matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        matrix_bytes = _matrix_to_bytes(matrix)
        set_rotation_on_stream(stream, matrix_bytes)

        retrieved_before = get_rotation_from_stream(stream)
        np.testing.assert_allclose(retrieved_before, matrix, atol=1e-5)

        output_container.start_encoding()

        retrieved_after = get_rotation_from_stream(stream)
        np.testing.assert_allclose(retrieved_after, matrix, atol=1e-5)

        output_container.close()
