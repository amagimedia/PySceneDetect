from scenedetect.video_stream import VideoStream
from scenedetect.frame_timecode import FrameTimecode
from typing import Union, Tuple, Optional, Iterable, List
import numpy as np

import os
import sys
from pathlib import Path
from datetime import timedelta
from urllib.parse import urlparse
import cv2
import m3u8
import subprocess
from queue import Queue, Empty
from threading import Thread
import selectors
import time

def video_length_opencv(
    inp_media: str
) -> Tuple[str, float, float, float, float]:
    """Video duration in HH:MM:SS

    Args:
        inp_file (str): Input video location

    Returns:
        str: Video duration as HH:MM:SS
    """

    if not inp_media:
        print(f"enter valid input for video meta extraction")
        sys.exit()
    video = cv2.VideoCapture(inp_media)

    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    ## get average fps
    fps = video.get(cv2.CAP_PROP_FPS)
    # fps = frame_count / duration
    print(f"Frame rate of the video : {fps}")

    seconds = frame_count / fps
    video_time = timedelta(seconds=seconds)
    # video_time = datetime.strftime('%H:%M:%S.%F', time.gmtime(seconds))[:-3]
    video_time = str(video_time)

    return video_time, fps, width, height, frame_count

class VideoReader_ffmpeg:
    """Video Reader object using FFmpeg"""

    @staticmethod
    def resize_frames(input_frames: np.ndarray, resize_ratio: float) -> np.ndarray:
        """Resize input frames from the video for further processing.

        Args:
            input_frames (np.ndarray): Input frames numpy array.
            resize_ratio (float): ratio to resize the height and width by.

        Returns:
            np.ndarray: resized numpy array.
        """
        resized_frms = np.array(
            [
                cv2.resize(
                    input_frames[frm_idx], None, fx=resize_ratio, fy=resize_ratio
                )
                for frm_idx in range(input_frames.shape[0])
            ]
        )

        return resized_frms

    def __init__(
        self,
        av_url: str,
        skip_duration_sec=0.0,
        duration_sec=-1.0,
        pipe_buffer_size=10**8,
    ):
        """Initialize video reader object

        Args:
            av_url (str): Input video location (S3 or local)
            skip_duration_sec (float, optional): Initial duration to skip.
                                                    Defaults to 0.0.
            duration_sec (float, optional): _description_. Defaults to -1.0.
        """
        av_url_parse = urlparse(av_url)
        self.local_av_file = None
        self.resize_ratio: Optional[float] = None

        # needed_params = ["MAX_VIDEO_FRAME_WIDTH"]
        # missing_params = set(needed_params).difference(run_cfg.keys())
        # self.MAX_VIDEO_FRAME_WIDTH = run_cfg["MAX_VIDEO_FRAME_WIDTH"]

        # if not len(missing_params) == 0:
        #     print(f"Missing params: {missing_params}")
        #     sys.exit()

        self.local_av_file = av_url_parse.path

        if Path(self.local_av_file).suffix == ".m3u8" and av_url_parse.scheme == "":
            m3u8_parser: m3u8.M3U8 = m3u8.load(self.local_av_file)
            self.local_av_file = (
                f"{Path(self.local_av_file).parent}/{m3u8_parser.playlists[0].uri}"
            )
        else:
            self.local_av_file = av_url

        # self.original_height, self.original_width, fps, duration, max_frames = list(map(json.loads(
        #     subprocess.check_output(
        #     ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', self.local_av_file])
        #     .decode('utf-8'))['streams'][0]
        #     .get, ['height', 'width', 'avg_frame_rate', 'duration', 'nb_frames']
        #     ))

        # if Path(self.local_av_file).suffix == ".m3u8":
        #     duration = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-i", self.local_av_file, "-skip_frame", "nokey", "-of", "default=noprint_wrappers=1:nokey=1"]).decode('utf-8')
        (
            self.video_duration,
            self.frame_rate,
            self.original_width,
            self.original_height,
            self.max_frames,
        ) = video_length_opencv(self.local_av_file)
        self.pipe_buffer_size = pipe_buffer_size
        self.max_frames = int(self.max_frames)
        # Resize width and height calculation
        aspect_ratio = self.original_width / self.original_height
        # self.src_frm_width = self.MAX_VIDEO_FRAME_WIDTH
        # self.src_frm_height = int(self.src_frm_width / aspect_ratio)
        self.src_frm_width = int(self.original_width)
        self.src_frm_height = int(self.original_height)
        self.src_frm_depth = 3
        self.src_frm_dims = (self.src_frm_height, self.src_frm_width)

        # Duration as HH:MM:SS.MS
        # self.video_duration = timedelta(seconds=np.round(float(duration), 3)).__str__()

        # Frame rate
        # num, denom = fps.split("/")
        # self.frame_rate = np.round(float(num) / float(denom), 3)

        # if self.src_frm_width > self.MAX_VIDEO_FRAME_WIDTH:
        #     self.resize_ratio = self.MAX_VIDEO_FRAME_WIDTH / self.src_frm_width

        # # Max frames
        # if self.max_frames is None:
        #     self.max_frames = int(float(self.video_duration) * self.frame_rate)
        # else:
        #     self.max_frames = int(self.max_frames)

    def frames_iterator(self, batch_size=1) -> Iterable[np.ndarray]:
        """Generate Iterator for frames based on batch size.

        Args:
            batch_size (int, optional): Defaults to 1.
        Returns:
            Iterable[np.ndarray]: a batch of frames to iterate through.

        Yields:
            Iterator[Iterable[np.ndarray]]: a batch of frames.
        """

        frm_buffer: List = []
        buffer_idx: int = 0
        frm_idx = 0

        for frame in self.get_frames():
            buffer_idx += 1
            frm_idx += 1
            frm_buffer.append(frame)

            if buffer_idx == batch_size or frm_idx == self.max_frames:
                yield np.array(frm_buffer)
                buffer_idx = 0
                frm_buffer = []

    def enqueue_frame(self, read_from, err, poll, q, size, timeout=15):
        sel = selectors.DefaultSelector()
        sel.register(read_from, selectors.EVENT_READ, "read")
        sel.register(err, selectors.EVENT_READ, "err")

        buf = bytearray()

        start = time.time()

        while (time.time() - start) < timeout:
            selected = sel.select(timeout=3)
            if selected is None:
                print("No data to read")
                continue

            for key, _ in selected:
                if key.data == "err":
                    # Also dump ffmpeg logging to VK logs
                    print(key.fileobj.read().decode("utf-8", "ignore"))
                else:
                    r = key.fileobj.read()  # ignoring size as fd is slow file
                    if r is None:  # Non blocking read returned empty, writer not ready
                        # This situation should not happen as
                        # read() is being called after selector returns
                        # There must be some data or even empty string.
                        continue
                    if r == b"":  # Writer end is closed
                        if len(buf) > 0 and len(buf) != size:
                            raise Exception(
                                f"Frame bytes mismatch {len(buf)} vs {size}"
                            )
                        return 0

                    buf.extend(r)

                    # Flush as many frames as read. Partial frames kept in buf
                    while len(buf) >= size:
                        q.put(buf[:size])
                        buf = buf[size:]

                    start = time.time()

        if (time.time() - start) >= timeout:
            raise Exception("Frame read timeout")

        return 0

    def get_frames(
        self, start_time_offset=None, num_frames=None
    ) -> Iterable[np.ndarray]:
        """Generate Iterator for individual frames.

        Returns:
            Iterable[np.ndarray]: frames to iterate through.

        Yields:
            Iterator[Iterable[np.ndarray]]: a single frame
        """
        proc_def = [
            "ffmpeg",
            "-nostats",
            "-nostdin",
            "-hide_banner",
            "-i",
            self.local_av_file,
            "-s",
            f"{self.src_frm_width}x{self.src_frm_height}",
        ]
        segment_args = [
            "-ss",
            f"{start_time_offset}",
            "-frames:v",
            f"{num_frames}",
        ]
        proc_def += segment_args if start_time_offset is not None else []
        proc_def += [
            "-an",
            "-f",
            "image2pipe",
            "-pix_fmt",
            "rgb24",
            "-vcodec",
            "rawvideo",
            "-",
        ]
        proc = subprocess.Popen(
            proc_def,
            bufsize=self.pipe_buffer_size,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        q = Queue(maxsize=20)

        os.set_blocking(proc.stdout.fileno(), False)
        os.set_blocking(proc.stderr.fileno(), False)

        reader = Thread(
            target=self.enqueue_frame,
            args=(
                proc.stdout,
                proc.stderr,
                proc.poll,
                q,
                self.src_frm_height * self.src_frm_width * self.src_frm_depth,
                60,
            ),
        )
        reader.daemon = True

        # No need to explicitly stop
        # We expect ffmpeg to either error out or exit successfully

        reader.start()

        wait_timeout = 0
        wait_thresh = 60  # Max wait time

        while reader.is_alive() or not q.empty():

            try:
                raw_image = q.get(timeout=1)  # 1 sec for a frame is already toomuch
            except Empty:
                wait_timeout += 1
                if wait_timeout >= wait_thresh:
                    raise Exception("Waited too long reading a frame")
                continue

            wait_timeout = 0

            frame = np.frombuffer(raw_image, dtype=np.uint8).reshape(
                (self.src_frm_height, self.src_frm_width, self.src_frm_depth)
            )
            yield frame

    def seek_frames(self, start_time_offset, num_frames) -> np.ndarray:
        """Get batch of frames starting from start_time_offset

        Returns:
            np.ndarray: batch of frames.
        """
        return np.array(list(self.get_frames(start_time_offset, num_frames)))

class VideoStream_ffmpeg(VideoStream):

    def __init__(self, path_or_url, name = "") -> None:
        super().__init__()

        self._path = path_or_url
        self.vr = VideoReader_ffmpeg(
            av_url=path_or_url,
            # run_cfg={
            #     "MAX_VIDEO_FRAME_WIDTH": 640
            # }
        )

        self._name = '' if name is None else name
        self._frame = None
        self._frame_idx = 0
        self._duration_frames = self.vr.max_frames
        self.duration_secs = self.vr.max_frames / self.vr.frame_rate
        self.frame_iterator = self.frames_iterator()

    BACKEND_NAME = 'ffmpeg'
    """Unique name used to identify this backend."""

    @property
    def path(self) -> Union[bytes, str]:
        """Video path."""
        return self._path
    
    @property
    def name(self) -> Union[bytes, str]:
        """Name of the video, without extension."""
        return self._name


    @property
    def is_seekable(self) -> bool:
        """True if seek() is allowed, False otherwise."""
        return True
    
    @property
    def frame_size(self) -> Tuple[int, int]:
        """Size of each video frame in pixels as a tuple of (width, height)."""
        return (self.vr.src_frm_width, self.vr.src_frm_height)
    
    @property
    def duration(self) -> FrameTimecode:
        """Duration of the video as a FrameTimecode."""
        return self.base_timecode + self._duration_frames
    
    @property
    def frame_rate(self) -> float:
        """Frame rate in frames/sec."""
        return self.vr.frame_rate
    
    @property
    def position(self) -> FrameTimecode:
        """Current position within stream as FrameTimecode.

        This can be interpreted as presentation time stamp, thus frame 1 corresponds
        to the presentation time 0.  Returns 0 even if `frame_number` is 1."""
        if self._frame is None:
            return self.base_timecode
        return FrameTimecode(self._frame_idx, self.vr.frame_rate)

    @property
    def position_ms(self) -> float:
        """Current position within stream as a float of the presentation time in
        milliseconds. The first frame has a PTS of 0."""
        if self._frame is None:
            return 0.0
        return self._frame.time * 1000.0

    @property
    def frame_number(self) -> int:
        """Current position within stream as the frame number.

        Will return 0 until the first frame is `read`."""
        if self._frame is not None:
            return self.position.frame_num + 1
        return 0
    
    @property
    def aspect_ratio(self) -> float:
        """Pixel aspect ratio as a float (1.0 represents square pixels)."""
        return self.vr.src_frm_width / self.vr.src_frm_height
    
    def frames_iterator(self):
        for frame in self.vr.get_frames():
            self._frame_idx += 1
            yield frame
    
    def read(self, decode: bool = True, advance: bool = True) -> Union[np.ndarray, bool]:
        has_advanced = False
        if advance:
            try:
                last_frame = self._frame
                self._frame = next(self.frame_iterator)
            except RecursionError:
                return False
            except Exception as e:
                self._frame = last_frame
                return self.read(decode, advance=True)
            
            has_advanced = True
        if decode:
            return self._frame
        return has_advanced
    
    def reset(self):
        """ Close and re-open the VideoStream (should be equivalent to calling `seek(0)`). """
        self.frame_iterator = self.vr.frames_iterator()
        self._frame = None

    def seek(self, target: Union[FrameTimecode, float, int]) -> None:
        """Seek to the given timecode. If given as a frame number, represents the current seek
        pointer (e.g. if seeking to 0, the next frame decoded will be the first frame of the video).

        For 1-based indices (first frame is frame #1), the target frame number needs to be converted
        to 0-based by subtracting one. For example, if we want to seek to the first frame, we call
        seek(0) followed by read(). If we want to seek to the 5th frame, we call seek(4) followed
        by read(), at which point frame_number will be 5.

        May not be supported on all input codecs (see `is_seekable`).

        Arguments:
            target: Target position in video stream to seek to.
                If float, interpreted as time in seconds.
                If int, interpreted as frame number.
        Raises:
            ValueError: `target` is not a valid value (i.e. it is negative).
        """
        if target < 0:
            raise ValueError("Target cannot be negative!")
        beginning = (target == 0)
        target = (self.base_timecode + target)
        if target >= 1:
            target = target - 1
        target_pts = int(
            (self.base_timecode + target).get_seconds())
        self._frame = None
        self.vr.seek_frames(start_time_offset=target_pts, num_frames=1)
        if not beginning:
            self.read(decode=False, advance=True)
        while self.position < target:
            if self.read(decode=False, advance=True) is False:
                break