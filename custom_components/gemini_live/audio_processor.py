"""Audio processing utilities for Gemini Live Audio integration."""
from __future__ import annotations

import base64
import io
import logging
import struct
import wave

from .const import AUDIO_INPUT_SAMPLE_RATE, AUDIO_OUTPUT_SAMPLE_RATE

_LOGGER = logging.getLogger(__name__)


class AudioProcessor:
    """Audio processing utilities."""

    @staticmethod
    def resample(audio_data: bytes, from_rate: int, to_rate: int) -> bytes:
        """Resample audio data to target sample rate.
        
        Simple linear resampling for 16-bit PCM audio.
        """
        if from_rate == to_rate:
            return audio_data

        # Convert bytes to samples
        samples = struct.unpack(f"<{len(audio_data) // 2}h", audio_data)

        # Calculate ratio
        ratio = to_rate / from_rate
        new_length = int(len(samples) * ratio)

        # Simple linear interpolation
        resampled = []
        for i in range(new_length):
            src_idx = i / ratio
            idx = int(src_idx)
            frac = src_idx - idx

            if idx + 1 < len(samples):
                value = samples[idx] * (1 - frac) + samples[idx + 1] * frac
            else:
                value = samples[idx] if idx < len(samples) else 0

            resampled.append(int(value))

        return struct.pack(f"<{len(resampled)}h", *resampled)

    @staticmethod
    def encode_base64(audio_data: bytes) -> str:
        """Encode audio data to base64 string."""
        return base64.b64encode(audio_data).decode("utf-8")

    @staticmethod
    def decode_base64(audio_string: str) -> bytes:
        """Decode base64 string to audio data."""
        return base64.b64decode(audio_string)

    @staticmethod
    def pcm_to_wav(
        pcm_data: bytes,
        sample_rate: int = AUDIO_OUTPUT_SAMPLE_RATE,
        channels: int = 1,
        sample_width: int = 2,
    ) -> bytes:
        """Convert raw PCM data to WAV format."""
        buffer = io.BytesIO()

        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)

        return buffer.getvalue()

    @staticmethod
    def wav_to_pcm(wav_data: bytes) -> tuple[bytes, int, int, int]:
        """Convert WAV data to raw PCM.
        
        Returns: (pcm_data, sample_rate, channels, sample_width)
        """
        buffer = io.BytesIO(wav_data)

        with wave.open(buffer, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            pcm_data = wav_file.readframes(wav_file.getnframes())

        return pcm_data, sample_rate, channels, sample_width

    @staticmethod
    def normalize_audio_for_input(audio_data: bytes, sample_rate: int = None) -> bytes:
        """Normalize audio for Gemini API input (16kHz 16-bit PCM mono)."""
        if sample_rate and sample_rate != AUDIO_INPUT_SAMPLE_RATE:
            audio_data = AudioProcessor.resample(
                audio_data,
                sample_rate,
                AUDIO_INPUT_SAMPLE_RATE,
            )
        return audio_data

    @staticmethod
    def normalize_audio_from_output(audio_data: bytes) -> bytes:
        """Output from Gemini is already 24kHz 16-bit PCM mono."""
        return audio_data

    @staticmethod
    def create_silence(duration_ms: int, sample_rate: int = AUDIO_OUTPUT_SAMPLE_RATE) -> bytes:
        """Create silence audio data."""
        num_samples = int(sample_rate * duration_ms / 1000)
        return struct.pack(f"<{num_samples}h", *([0] * num_samples))

    @staticmethod
    def get_audio_duration_ms(audio_data: bytes, sample_rate: int = AUDIO_OUTPUT_SAMPLE_RATE) -> int:
        """Calculate audio duration in milliseconds."""
        num_samples = len(audio_data) // 2  # 16-bit = 2 bytes per sample
        return int(num_samples * 1000 / sample_rate)


class AudioBuffer:
    """Buffer for accumulating audio data."""

    def __init__(self, sample_rate: int = AUDIO_OUTPUT_SAMPLE_RATE) -> None:
        """Initialize the buffer."""
        self._buffer = io.BytesIO()
        self._sample_rate = sample_rate
        self._lock = None  # Will be set to asyncio.Lock() when needed

    @property
    def sample_rate(self) -> int:
        """Return the sample rate."""
        return self._sample_rate

    def write(self, data: bytes) -> None:
        """Write audio data to the buffer."""
        self._buffer.write(data)

    def read(self, num_bytes: int = -1) -> bytes:
        """Read audio data from the buffer."""
        self._buffer.seek(0)
        data = self._buffer.read(num_bytes)
        # Remove read data from buffer
        remaining = self._buffer.read()
        self._buffer = io.BytesIO()
        self._buffer.write(remaining)
        return data

    def read_all(self) -> bytes:
        """Read all data from the buffer."""
        self._buffer.seek(0)
        data = self._buffer.read()
        self._buffer = io.BytesIO()
        return data

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer = io.BytesIO()

    @property
    def size(self) -> int:
        """Return the current buffer size in bytes."""
        pos = self._buffer.tell()
        self._buffer.seek(0, 2)  # Seek to end
        size = self._buffer.tell()
        self._buffer.seek(pos)
        return size

    @property
    def duration_ms(self) -> int:
        """Return the current buffer duration in milliseconds."""
        return AudioProcessor.get_audio_duration_ms(
            b"\x00" * self.size,
            self._sample_rate,
        )

    def get_wav(self) -> bytes:
        """Get buffer contents as WAV."""
        pcm_data = self.read_all()
        return AudioProcessor.pcm_to_wav(pcm_data, self._sample_rate)


class AudioChunker:
    """Chunk audio data for streaming."""

    def __init__(
        self,
        chunk_size_ms: int = 100,
        sample_rate: int = AUDIO_INPUT_SAMPLE_RATE,
    ) -> None:
        """Initialize the chunker."""
        self._chunk_size_ms = chunk_size_ms
        self._sample_rate = sample_rate
        # 16-bit = 2 bytes per sample
        self._chunk_size_bytes = int(sample_rate * chunk_size_ms / 1000) * 2
        self._buffer = b""

    def add_audio(self, audio_data: bytes) -> list[bytes]:
        """Add audio and return complete chunks."""
        self._buffer += audio_data
        
        chunks = []
        while len(self._buffer) >= self._chunk_size_bytes:
            chunk = self._buffer[:self._chunk_size_bytes]
            self._buffer = self._buffer[self._chunk_size_bytes:]
            chunks.append(chunk)
        
        return chunks

    def flush(self) -> bytes | None:
        """Flush remaining data as final chunk."""
        if self._buffer:
            final = self._buffer
            self._buffer = b""
            return final
        return None
