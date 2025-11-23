from libc.stdint cimport int32_t
from libc.string cimport memcpy

import cython
cimport libav as lib

from collections.abc import Mapping
from enum import Enum

from av.sidedata.motionvectors import MotionVectors
from av.bytesource cimport ByteSource, bytesource
from av.stream cimport Stream


cdef object _cinit_bypass_sentinel = object()


class Type(Enum):
    """
    Enum class representing different types of frame data in audio/video processing.
    Values are mapped to corresponding AV_FRAME_DATA constants from FFmpeg.

    From: https://github.com/FFmpeg/FFmpeg/blob/master/libavutil/frame.h
    """
    PANSCAN = lib.AV_FRAME_DATA_PANSCAN
    A53_CC = lib.AV_FRAME_DATA_A53_CC
    STEREO3D = lib.AV_FRAME_DATA_STEREO3D
    MATRIXENCODING = lib.AV_FRAME_DATA_MATRIXENCODING
    DOWNMIX_INFO = lib.AV_FRAME_DATA_DOWNMIX_INFO
    REPLAYGAIN = lib.AV_FRAME_DATA_REPLAYGAIN
    DISPLAYMATRIX = lib.AV_FRAME_DATA_DISPLAYMATRIX
    AFD = lib.AV_FRAME_DATA_AFD
    MOTION_VECTORS = lib.AV_FRAME_DATA_MOTION_VECTORS
    SKIP_SAMPLES = lib.AV_FRAME_DATA_SKIP_SAMPLES
    AUDIO_SERVICE_TYPE = lib.AV_FRAME_DATA_AUDIO_SERVICE_TYPE
    MASTERING_DISPLAY_METADATA = lib.AV_FRAME_DATA_MASTERING_DISPLAY_METADATA
    GOP_TIMECODE = lib.AV_FRAME_DATA_GOP_TIMECODE
    SPHERICAL = lib.AV_FRAME_DATA_SPHERICAL
    CONTENT_LIGHT_LEVEL = lib.AV_FRAME_DATA_CONTENT_LIGHT_LEVEL
    ICC_PROFILE = lib.AV_FRAME_DATA_ICC_PROFILE
    S12M_TIMECODE = lib.AV_FRAME_DATA_S12M_TIMECODE
    DYNAMIC_HDR_PLUS = lib.AV_FRAME_DATA_DYNAMIC_HDR_PLUS
    REGIONS_OF_INTEREST = lib.AV_FRAME_DATA_REGIONS_OF_INTEREST
    VIDEO_ENC_PARAMS = lib.AV_FRAME_DATA_VIDEO_ENC_PARAMS
    SEI_UNREGISTERED = lib.AV_FRAME_DATA_SEI_UNREGISTERED
    FILM_GRAIN_PARAMS = lib.AV_FRAME_DATA_FILM_GRAIN_PARAMS
    DETECTION_BBOXES = lib.AV_FRAME_DATA_DETECTION_BBOXES
    DOVI_RPU_BUFFER = lib.AV_FRAME_DATA_DOVI_RPU_BUFFER
    DOVI_METADATA = lib.AV_FRAME_DATA_DOVI_METADATA
    DYNAMIC_HDR_VIVID = lib.AV_FRAME_DATA_DYNAMIC_HDR_VIVID
    AMBIENT_VIEWING_ENVIRONMENT = lib.AV_FRAME_DATA_AMBIENT_VIEWING_ENVIRONMENT
    VIDEO_HINT = lib.AV_FRAME_DATA_VIDEO_HINT


cdef SideData wrap_side_data(Frame frame, int index):
    if frame.ptr.side_data[index].type == lib.AV_FRAME_DATA_MOTION_VECTORS:
        return MotionVectors(_cinit_bypass_sentinel, frame, index)
    else:
        return SideData(_cinit_bypass_sentinel, frame, index)


cdef int get_display_rotation(Frame frame):
    for i in range(frame.ptr.nb_side_data):
        if frame.ptr.side_data[i].type == lib.AV_FRAME_DATA_DISPLAYMATRIX:
            return int(lib.av_display_rotation_get(<const int32_t *>frame.ptr.side_data[i].data))
    return 0


cdef class SideData(Buffer):
    def __init__(self, sentinel, Frame frame, int index):
        if sentinel is not _cinit_bypass_sentinel:
            raise RuntimeError("cannot manually instantiate SideData")
        self.frame = frame
        self.ptr = frame.ptr.side_data[index]
        self.metadata = wrap_dictionary(self.ptr.metadata)

    cdef size_t _buffer_size(self):
        return self.ptr.size

    cdef void* _buffer_ptr(self):
        return self.ptr.data

    cdef bint _buffer_writable(self):
        return False

    def __repr__(self):
        return f"<av.sidedata.{self.__class__.__name__} {self.ptr.size} bytes of {self.type} at 0x{<unsigned int>self.ptr.data:0x}>"

    @property
    def type(self):
        return Type(self.ptr.type)


cdef class _SideDataContainer:
    def __init__(self, Frame frame):
        self.frame = frame
        self._refresh()

    def __len__(self):
        return len(self._by_index)

    def __iter__(self):
        return iter(self._by_index)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._by_index[key]
        if isinstance(key, str):
            return self._by_type[Type[key]]
        return self._by_type[key]

    def _refresh(self):
        """Refresh the side data container after adding new side data."""
        self._by_index = []
        self._by_type = {}

        cdef int i
        cdef SideData data
        for i in range(self.frame.ptr.nb_side_data):
            data = wrap_side_data(self.frame, i)
            self._by_index.append(data)
            self._by_type[data.type] = data

    def add(self, data_type, data_bytes):
        """
        Add side data to this frame.

        :param data_type: The type of side data (from :class:`Type` enum or string name)
        :param data_bytes: The side data as bytes
        :type data_bytes: bytes

        If side data of this type already exists, it will be replaced.
        """
        cdef lib.AVFrameSideDataType dtype
        cdef lib.AVFrameSideData* side_data_ptr

        if isinstance(data_type, str):
            try:
                type_enum = Type[data_type]
                dtype = type_enum.value
            except KeyError:
                raise ValueError(f"Unknown side data type: {data_type}")
        elif isinstance(data_type, Type):
            dtype = data_type.value
        else:
            raise TypeError(f"data_type must be Type enum or string, got {type(data_type)}")

        if not isinstance(data_bytes, bytes):
            raise TypeError(f"data_bytes must be bytes, got {type(data_bytes)}")

        self.frame.make_writable()

        cdef ByteSource source = bytesource(data_bytes)

        existing_side_data = lib.av_frame_get_side_data(self.frame.ptr, dtype)
        if existing_side_data != cython.NULL:
            if existing_side_data.size != source.length:
                raise ValueError(
                    f"Existing side data has size {existing_side_data.size}, "
                    f"but new data has size {source.length}"
                )
            memcpy(existing_side_data.data, source.ptr, source.length)
        else:
            side_data_ptr = lib.av_frame_new_side_data(
                self.frame.ptr,
                dtype,
                source.length
            )
            if side_data_ptr == cython.NULL:
                raise RuntimeError("Failed to allocate side data")
            memcpy(side_data_ptr.data, source.ptr, source.length)

        self._refresh()


class SideDataContainer(_SideDataContainer, Mapping):
    pass


cdef set_rotation_on_stream_c(object stream_obj, matrix_bytes):
    """
    Set display matrix side data on a stream's codecpar and codec context (Cython implementation).

    This ensures the rotation is preserved in container metadata during encoding.
    """
    cdef Stream stream = stream_obj
    cdef lib.AVPacketSideData* side_data_ptr
    cdef ByteSource source = bytesource(matrix_bytes)
    cdef lib.AVCodecParameters* codecpar = stream.ptr.codecpar
    cdef lib.AVCodecContext* codec_ctx = stream.codec_context.ptr if stream.codec_context else NULL

    existing = lib.av_packet_side_data_get(
        codecpar.coded_side_data,
        codecpar.nb_coded_side_data,
        lib.AV_PKT_DATA_DISPLAYMATRIX
    )

    if existing != cython.NULL:
        if existing.size != source.length:
            raise ValueError(
                f"Existing side data has size {existing.size}, "
                f"but new data has size {source.length}"
            )
        memcpy(existing.data, source.ptr, source.length)
    else:
        side_data_ptr = lib.av_packet_side_data_new(
            &codecpar.coded_side_data,
            &codecpar.nb_coded_side_data,
            lib.AV_PKT_DATA_DISPLAYMATRIX,
            source.length,
            0
        )
        if side_data_ptr == cython.NULL:
            raise RuntimeError("Failed to allocate side data")
        memcpy(side_data_ptr.data, source.ptr, source.length)

    if codec_ctx != NULL:
        existing_ctx = lib.av_packet_side_data_get(
            codec_ctx.coded_side_data,
            codec_ctx.nb_coded_side_data,
            lib.AV_PKT_DATA_DISPLAYMATRIX
        )
        if existing_ctx != cython.NULL:
            if existing_ctx.size != source.length:
                raise ValueError(
                    f"Existing codec context side data has size {existing_ctx.size}, "
                    f"but new data has size {source.length}"
                )
            memcpy(existing_ctx.data, source.ptr, source.length)
        else:
            side_data_ptr = lib.av_packet_side_data_new(
                &codec_ctx.coded_side_data,
                &codec_ctx.nb_coded_side_data,
                lib.AV_PKT_DATA_DISPLAYMATRIX,
                source.length,
                0
            )
            if side_data_ptr == cython.NULL:
                raise RuntimeError("Failed to allocate side data on codec context")
            memcpy(side_data_ptr.data, source.ptr, source.length)


def set_rotation_on_stream(stream, matrix_bytes):
    """Python wrapper for set_rotation_on_stream_c."""
    try:
        set_rotation_on_stream_c(stream, matrix_bytes)
    except TypeError:
        raise TypeError(
            f"stream must be a Stream, got {type(stream)}"
        )


def get_rotation_from_stream(stream):
    """Get display matrix from stream's codecpar."""
    cdef Stream stream_base = stream
    cdef lib.AVCodecParameters* codecpar = stream_base.ptr.codecpar
    cdef const lib.AVPacketSideData* existing

    existing = lib.av_packet_side_data_get(
        codecpar.coded_side_data,
        codecpar.nb_coded_side_data,
        lib.AV_PKT_DATA_DISPLAYMATRIX
    )

    if existing == cython.NULL or existing.size != 36:
        return None

    cdef bytes matrix_bytes = (<char*>existing.data)[:existing.size]
    from av.sidedata.display import _bytes_to_matrix
    return _bytes_to_matrix(matrix_bytes)
