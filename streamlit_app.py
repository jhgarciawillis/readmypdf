import streamlit as st
import requests
import json
import spacy
import fitz
import os
import tempfile
import boto3

from pathlib import Path
from boto3 import Session
from tqdm import tqdm
from google.cloud import language_v1, speech, vision

# Handle potential import errors
try:
    import easyocr
    import cv2
except ImportError as e:
    st.error(f"Error importing required libraries: {e}")
    st.error("Please make sure all required libraries are installed correctly.")
    st.stop()

class Config:
    AUDIO_OUTPUT_FORMAT = 'mp3'
    AUDIO_SAMPLE_RATE = '24000'
    SUPPORTED_LANGUAGES = {
        'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
        'it': 'Italian', 'ru': 'Russian', 'zh-CN': 'Chinese (Simplified)',
        'ja': 'Japanese', 'ar': 'Arabic', 'hi': 'Hindi', 'pt': 'Portuguese', 'nl': 'Dutch'
    }

class AWSClient:
    def __init__(self):
        self.aws_access_key = st.secrets["AWS_ACCESS_KEY"]
        self.aws_secret_key = st.secrets["AWS_SECRET_KEY"]
        self.aws_region = st.secrets["AWS_REGION"]
        self.session = Session(
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
            region_name=self.aws_region
        )

    def get_comprehend_client(self):
        return boto3.client('comprehend', region_name=self.aws_region,
                            aws_access_key_id=self.aws_access_key,
                            aws_secret_access_key=self.aws_secret_key)

    def get_polly_client(self):
        return self.session.client('polly')

class GoogleClient:
    def __init__(self):
        self.credentials = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials

    def get_language_client(self):
        return language_v1.LanguageServiceClient()

    def get_vision_client(self):
        return vision.ImageAnnotatorClient()

class PDFProcessor:
    @staticmethod
    @st.cache_data
    def convert_pdf_to_images(pdf_file, output_folder):
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page_index in range(len(pdf_document)):
            page = pdf_document[page_index]
            image = page.get_pixmap()
            image.save(f"{output_folder}/page_{page_index}.png")
        pdf_document.close()

    @staticmethod
    def detect_margins(image_path, margin_threshold=10):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        margin_contours = [contour for contour in contours if cv2.contourArea(contour) > margin_threshold]

        if margin_contours:
            min_x, min_y, max_x, max_y = cv2.boundingRect(max(margin_contours, key=cv2.contourArea))
            cropped_image = image[min_y:max_y, min_x:max_x]
            return cropped_image
        else:
            return image

    @staticmethod
    @st.cache_data
    def extract_text_from_image(image, language_codes):
        reader = easyocr.Reader(language_codes)
        text = reader.read_from_cv2_image(image)
        return ' '.join([word[-2] for word in text])

    @staticmethod
    @st.cache_data
    def extract_text(pdf_file, start_page=1, end_page=None, language_codes=[]):
        text_content = ""
        try:
            with tempfile.TemporaryDirectory() as temp_folder:
                PDFProcessor.convert_pdf_to_images(pdf_file, temp_folder)
                
                pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
                num_pages = len(pdf_document)
                end_page = num_pages if end_page is None or end_page > num_pages else end_page

                progress_bar = st.progress(0)
                for page_num in range(start_page - 1, end_page):
                    image_path = f"{temp_folder}/page_{page_num}.png"
                    cropped_image = PDFProcessor.detect_margins(image_path)
                    text_content += PDFProcessor.extract_text_from_image(cropped_image, language_codes)
                    progress = (page_num - start_page + 2) / (end_page - start_page + 1)
                    progress_bar.progress(progress)

        except Exception as e:
            st.error(f"An error occurred while extracting text: {e}")
        finally:
            pdf_document.close()
        return text_content

# ... [rest of the classes remain unchanged] ...

def main():
    st.title("PDF to Audio Converter")

    aws_client = AWSClient()
    google_client = GoogleClient()

    pdf_file = st.file_uploader("Upload PDF file", type="pdf")
    
    if pdf_file:
        start_page = st.number_input("Start Page", min_value=1, value=1)
        language = st.selectbox("PDF Language", options=list(Config.SUPPORTED_LANGUAGES.keys()), format_func=lambda x: Config.SUPPORTED_LANGUAGES[x])
        audio_language = st.selectbox("Audio Language", options=list(Config.SUPPORTED_LANGUAGES.keys()), format_func=lambda x: Config.SUPPORTED_LANGUAGES[x])
        
        voices = VoiceManager.list_voices(language_code=audio_language, aws_client=aws_client)
        voice_id = st.selectbox("Select Voice", options=[v[0] for v in voices], format_func=lambda x: dict(voices)[x])
        
        engine = st.selectbox("Engine", options=["neural", "standard"])

        if st.button("Convert to Audio"):
            with st.spinner("Converting PDF to audio..."):
                audio_data = PDFToAudioConverter.convert(pdf_file, start_page, voice_id, engine, audio_language, aws_client, google_client)
            
            if audio_data:
                st.success(f"Audio generated for {len(audio_data)} characters.")
                for character, audio in audio_data.items():
                    st.audio(audio, format='audio/mp3')
            else:
                st.error("Failed to convert PDF to audio.")

if __name__ == "__main__":
    main()
