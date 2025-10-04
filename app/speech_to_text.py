import whisper
import speech_recognition as sr
import tempfile
import os
import logging
from typing import Optional
import librosa
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechToTextProcessor:
    def __init__(self, use_whisper: bool = True):
        """Initialize the speech-to-text processor."""
        self.use_whisper = use_whisper
        self.whisper_model = None
        self.sr_recognizer = sr.Recognizer()

        if use_whisper:
            try:
                logger.info("Loading Whisper model...")
                self.whisper_model = whisper.load_model("base")
                logger.info("Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                self.use_whisper = False

    def transcribe_audio(self, audio_file_path: str) -> Optional[str]:
        """
        Transcribe audio file to text.

        Args:
            audio_file_path: Path to the audio file

        Returns:
            Transcribed text or None if failed
        """
        try:
            # Ensure audio file exists
            if not os.path.exists(audio_file_path):
                logger.error(f"Audio file not found: {audio_file_path}")
                return None

            # Try Whisper first
            if self.use_whisper and self.whisper_model:
                return self._transcribe_with_whisper(audio_file_path)

            # Fallback to SpeechRecognition
            return self._transcribe_with_sr(audio_file_path)

        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return None

    def _transcribe_with_whisper(self, audio_file_path: str) -> Optional[str]:
        """Transcribe using OpenAI Whisper."""
        try:
            logger.info("Transcribing with Whisper...")
            result = self.whisper_model.transcribe(audio_file_path)
            text = result["text"].strip()
            logger.info(f"Whisper transcription: {text}")
            return text
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return self._transcribe_with_sr(audio_file_path)

    def _transcribe_with_sr(self, audio_file_path: str) -> Optional[str]:
        """Transcribe using SpeechRecognition library."""
        try:
            logger.info("Transcribing with SpeechRecognition...")

            # Convert audio to WAV format if needed
            audio_data, sr_rate = librosa.load(audio_file_path, sr=16000)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_data, sr_rate)

                with sr.AudioFile(tmp_file.name) as source:
                    audio = self.sr_recognizer.record(source)

                # Try Google Speech Recognition
                text = self.sr_recognizer.recognize_google(audio)
                logger.info(f"SpeechRecognition transcription: {text}")

                # Clean up temp file
                os.unlink(tmp_file.name)

                return text

        except sr.UnknownValueError:
            logger.error("Speech Recognition could not understand audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Could not request results from Google Speech Recognition service: {e}")
            return None
        except Exception as e:
            logger.error(f"SpeechRecognition transcription failed: {e}")
            return None

    def transcribe_bytes(self, audio_bytes: bytes, format: str = "wav") -> Optional[str]:
        """
        Transcribe audio from bytes.

        Args:
            audio_bytes: Audio data as bytes
            format: Audio format (wav, mp3, etc.)

        Returns:
            Transcribed text or None if failed
        """
        try:
            # Save bytes to temporary file
            with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file.flush()

                # Transcribe the temporary file
                result = self.transcribe_audio(tmp_file.name)

                # Clean up
                os.unlink(tmp_file.name)

                return result

        except Exception as e:
            logger.error(f"Error transcribing audio bytes: {e}")
            return None
