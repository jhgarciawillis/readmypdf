from gtts import gTTS
import io
import os
import tempfile
import streamlit as st
from pydub import AudioSegment
import logging
from config import Config

class AudioGenerator:
    @staticmethod
    @st.cache_data
    def generate_character_audio(text, language, rate=1.0, pitch=1.0):
        try:
            tts = gTTS(text=text, lang=language, slow=False)
            audio_io = io.BytesIO()
            tts.write_to_fp(audio_io)
            audio_io.seek(0)
            
            # Load the audio using pydub
            audio = AudioSegment.from_file(audio_io, format="mp3")
            
            # Adjust speed (rate)
            if rate != 1.0:
                audio = audio._spawn(audio.raw_data, overrides={
                    "frame_rate": int(audio.frame_rate * rate)
                })
            
            # Adjust pitch
            if pitch != 1.0:
                octaves = pitch - 1
                new_sample_rate = int(audio.frame_rate * (2.0 ** octaves))
                audio = audio._spawn(audio.raw_data, overrides={
                    "frame_rate": new_sample_rate
                })
            
            # Export the modified audio to a new BytesIO object
            modified_audio_io = io.BytesIO()
            audio.export(modified_audio_io, format="mp3")
            modified_audio_io.seek(0)
            
            return modified_audio_io.getvalue()
        except Exception as e:
            logging.error(f"Error generating audio: {str(e)}")
            raise

    @staticmethod
    @st.cache_data
    def split_audio_into_chunks(audio_data, chunk_size=60):  # chunk_size in seconds
        try:
            # Load the audio data
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
            
            # Split the audio into chunks
            chunk_length_ms = chunk_size * 1000  # Convert to milliseconds
            chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
            
            # Convert chunks back to bytes
            chunk_bytes = []
            for chunk in chunks:
                chunk_io = io.BytesIO()
                chunk.export(chunk_io, format="mp3")
                chunk_bytes.append(chunk_io.getvalue())
            
            return chunk_bytes
        except Exception as e:
            logging.error(f"Error splitting audio into chunks: {str(e)}")
            raise

    @staticmethod
    @st.cache_data
    def merge_audio_files(audio_files):
        try:
            combined = AudioSegment.empty()
            for audio_file in audio_files:
                segment = AudioSegment.from_file(io.BytesIO(audio_file), format="mp3")
                combined += segment
            
            output = io.BytesIO()
            combined.export(output, format="mp3")
            return output.getvalue()
        except Exception as e:
            logging.error(f"Error merging audio files: {str(e)}")
            raise

    @staticmethod
    @st.cache_data
    def apply_audio_effects(audio_data, volume_change=0, fade_in=0, fade_out=0):
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
            
            # Adjust volume
            if volume_change != 0:
                audio = audio + volume_change
            
            # Apply fade in
            if fade_in > 0:
                audio = audio.fade_in(duration=fade_in * 1000)
            
            # Apply fade out
            if fade_out > 0:
                audio = audio.fade_out(duration=fade_out * 1000)
            
            output = io.BytesIO()
            audio.export(output, format="mp3")
            return output.getvalue()
        except Exception as e:
            logging.error(f"Error applying audio effects: {str(e)}")
            raise

    @staticmethod
    def get_audio_duration(audio_data):
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
            return len(audio) / 1000.0  # Duration in seconds
        except Exception as e:
            logging.error(f"Error getting audio duration: {str(e)}")
            raise

if __name__ == "__main__":
    # This block can be used for testing the AudioGenerator class
    logging.basicConfig(level=logging.INFO)
    st.set_page_config(page_title="Audio Generator Test", layout="wide")
    st.title("Audio Generator Test")

    text_input = st.text_area("Enter text to convert to speech:", height=150)
    language = st.selectbox("Select language", options=['en', 'es', 'fr'])
    rate = st.slider("Speech Rate", 0.5, 2.0, 1.0, 0.1)
    pitch = st.slider("Speech Pitch", 0.5, 2.0, 1.0, 0.1)

    if st.button("Generate Audio"):
        if text_input:
            with st.spinner("Generating audio..."):
                try:
                    audio_data = AudioGenerator.generate_character_audio(text_input, language, rate, pitch)
                    st.audio(audio_data, format='audio/mp3')
                    
                    duration = AudioGenerator.get_audio_duration(audio_data)
                    st.write(f"Audio duration: {duration:.2f} seconds")

                    chunks = AudioGenerator.split_audio_into_chunks(audio_data, chunk_size=10)
                    st.write(f"Number of 10-second chunks: {len(chunks)}")

                    modified_audio = AudioGenerator.apply_audio_effects(audio_data, volume_change=5, fade_in=1, fade_out=1)
                    st.audio(modified_audio, format='audio/mp3')
                    st.write("Audio with effects applied (volume +5dB, 1s fade in/out)")

                    st.success("Audio generated and processed successfully!")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter some text to convert to speech.")