"""
Utilities for setting rotation on video frames using display matrix side data.
"""

import struct

import numpy as np


_EXIF_MATRICES = None


def _get_exif_matrices():
    global _EXIF_MATRICES
    if _EXIF_MATRICES is None:
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

        _EXIF_MATRICES = {
            1: matrix_exif1,
            2: matrix_exif2,
            3: matrix_exif3,
            4: matrix_exif4,
            5: matrix_exif5,
            6: matrix_exif6,
            7: matrix_exif7,
            8: matrix_exif8,
        }
    return _EXIF_MATRICES


def _matrix_to_bytes(matrix: np.ndarray) -> bytes:
    matrix = np.asarray(matrix, dtype=np.float64)

    if matrix.shape != (3, 3):
        raise ValueError(f"Matrix must be 3x3, got shape {matrix.shape}")

    result = bytearray()

    for i in range(2):
        for j in range(3):
            value = matrix[i, j]
            fixed_point = int(value * (1 << 16))
            result.extend(struct.pack('>i', fixed_point))

    for j in range(3):
        value = matrix[2, j]
        fixed_point = int(value * (1 << 30))
        result.extend(struct.pack('>i', fixed_point))

    return bytes(result)


def _bytes_to_matrix(matrix_bytes: bytes) -> np.ndarray:
    if len(matrix_bytes) != 36:
        raise ValueError(f"Display matrix must be 36 bytes, got {len(matrix_bytes)}")

    matrix = np.zeros((3, 3), dtype=np.float64)

    for i in range(2):
        for j in range(3):
            idx = i * 3 + j
            fixed_point = struct.unpack('>i', matrix_bytes[idx * 4:(idx + 1) * 4])[0]
            matrix[i, j] = fixed_point / (1 << 16)

    for j in range(3):
        idx = 6 + j
        fixed_point = struct.unpack('>i', matrix_bytes[idx * 4:(idx + 1) * 4])[0]
        matrix[2, j] = fixed_point / (1 << 30)

    return matrix


def set_rotation(target, rotation) -> None:
    """
    Set the rotation on a video frame or stream.

    For encoding, set rotation on the stream before encoding starts to ensure
    it's preserved in the container metadata.

    :param target: A VideoFrame or VideoStream to set the rotation on
    :param rotation: Either a 3x3 numpy array (float32 or float64) representing the
                     transformation matrix, or an integer EXIF orientation value (1-8)
    """
    from av.sidedata.sidedata import Type, set_rotation_on_stream

    if isinstance(rotation, (int, np.integer)):
        exif_orientation = int(rotation)
        if exif_orientation < 1 or exif_orientation > 8:
            raise ValueError(f"EXIF orientation must be 1-8, got {exif_orientation}")
        matrices = _get_exif_matrices()
        matrix = matrices[exif_orientation]
    elif isinstance(rotation, np.ndarray):
        matrix = rotation
    else:
        raise TypeError(
            f"rotation must be a 3x3 numpy array or EXIF orientation (1-8), "
            f"got {type(rotation)}"
        )

    matrix_bytes = _matrix_to_bytes(matrix)

    try:
        target.side_data.add(Type.DISPLAYMATRIX, matrix_bytes)
    except AttributeError:
        try:
            set_rotation_on_stream(target, matrix_bytes)
        except (AttributeError, TypeError) as e:
            raise TypeError(
                f"target must be a VideoFrame or VideoStream, got {type(target)}"
            ) from e


def get_rotation(target) -> np.ndarray | None:
    """
    Get the rotation matrix from a video frame or stream.

    :param target: A VideoFrame or VideoStream to get the rotation matrix from
    :return: A 3x3 numpy array (float64) representing the transformation matrix, or None if not present
    """
    from av.sidedata.sidedata import Type, get_rotation_from_stream

    try:
        side_data = target.side_data[Type.DISPLAYMATRIX]
        return _bytes_to_matrix(bytes(side_data))
    except (KeyError, AttributeError):
        if hasattr(target, 'ptr'):
            try:
                return get_rotation_from_stream(target)
            except (AttributeError, TypeError):
                pass
        return None
