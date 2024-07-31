import streamlit as st
import requests
import json
import fitz
import os
import tempfile
import boto3
from PIL import Image
import pytesseract
from pathlib import Path
from boto3 import Session
from tqdm import tqdm
from google.cloud import language_v1, speech, vision

# Handle potential import errors
try:
    import spacy
except ImportError as e:
    st.error(f"Error importing spacy: {e}")
    st.error("Please make sure spacy and its dependencies are installed correctly.")
    st.stop()

class Config:
    AUDIO_OUTPUT_FORMAT = 'mp3'
    AUDIO_SAMPLE_RATE = '24000'
    SUPPORTED_LANGUAGES = {
        'en': 'eng',
        'es': 'spa',
        'fr': 'fra',
        'de': 'deu',
        'it': 'ita',
        'ru': 'rus',
        'zh-CN': 'chi_sim',
        'ja': 'jpn',
        'ar': 'ara',
        'hi': 'hin',
        'pt': 'por',
        'nl': 'nld'
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
            pix = page.get_pixmap()
            pix.save(f"{output_folder}/page_{page_index}.png")
        pdf_document.close()

    @staticmethod
    def detect_margins(image_path):
        with Image.open(image_path) as img:
            bg = Image.new(img.mode, img.size, img.getpixel((0,0)))
            diff = Image.new(img.mode, img.size)
            for x in range(img.width):
                for y in range(img.height):
                    pixel = img.getpixel((x, y))
                    if pixel != bg.getpixel((x, y)):
                        diff.putpixel((x, y), pixel)
            bbox = diff.getbbox()
        return bbox

    @staticmethod
    @st.cache_data
    def extract_text_from_image(image_path, language_code):
        return pytesseract.image_to_string(Image.open(image_path), lang=language_code)

    @staticmethod
    @st.cache_data
    def extract_text(pdf_file, start_page=1, end_page=None, language_code='eng'):
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
                    bbox = PDFProcessor.detect_margins(image_path)
                    if bbox:
                        with Image.open(image_path) as img:
                            cropped_img = img.crop(bbox)
                            cropped_path = f"{temp_folder}/cropped_page_{page_num}.png"
                            cropped_img.save(cropped_path)
                        text_content += PDFProcessor.extract_text_from_image(cropped_path, language_code)
                    else:
                        text_content += PDFProcessor.extract_text_from_image(image_path, language_code)
                    progress = (page_num - start_page + 2) / (end_page - start_page + 1)
                    progress_bar.progress(progress)

        except Exception as e:
            st.error(f"An error occurred while extracting text: {e}")
        finally:
            pdf_document.close()
        return text_content

class CharacterAnalyzer:
    @staticmethod
    @st.cache_data
    def detect_character_names(text, language_code):
        try:
            if language_code == 'en':
                nlp = spacy.load('en_core_web_sm')
            elif language_code == 'es':
                nlp = spacy.load('es_core_news_sm')
            elif language_code == 'fr':
                nlp = spacy.load('fr_core_news_sm')
            else:
                raise ValueError(f"Unsupported language code: {language_code}")

            doc = nlp(text)
            character_names = [entity.text for entity in doc.ents if entity.label_ == 'PERSON']
            return character_names
        except OSError:
            st.error(f"Language model for {language_code} not found. Please install it using 'python -m spacy download {language_code}_core_web_sm'")
            return []

    @staticmethod
    def detect_character_attributes(text, language_code, aws_client, google_client):
        comprehend_client = aws_client.get_comprehend_client()
        emotion_response = comprehend_client.detect_sentiment(Text=text, LanguageCode=language_code)
        emotion = emotion_response['Sentiment'].lower()
        
        language_client = google_client.get_language_client()
        document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT, language=language_code)
        entity_sentiment_response = language_client.analyze_entity_sentiment(document=document)
        
        characters = []
        for entity in entity_sentiment_response.entities:
            if entity.type_ == language_v1.Entity.Type.PERSON:
                vision_client = google_client.get_vision_client()
                image = vision.Image(content=entity.name.encode('utf-8'))
                response = vision_client.face_detection(image=image)
                face = response.face_annotations[0] if response.face_annotations else None
                
                gender = face.gender.lower() if face and face.gender else ""
                age = face.age_range.low if face and face.age_range else ""
                
                character = {
                    "name": entity.name,
                    "gender": gender,
                    "age": age,
                    "emotion": entity.sentiment.score,
                    "pace": ""
                }
                characters.append(character)
        
        return characters

class AudioGenerator:
    @staticmethod
    def generate_character_audio(text, character, voice_id, engine, language, aws_client):
        polly = aws_client.get_polly_client()

        voice_settings = {
            'Engine': engine,
            'LanguageCode': language,
            'VoiceId': voice_id,
            'TextType': 'ssml'
        }

        emotion_tag = f'<amazon:emotion name="{character["emotion"]}" intensity="medium">' if character["emotion"] else ''
        rate_tag = f'<prosody rate="{character["pace"]}">'

        ssml_text = f'<speak>{emotion_tag}{rate_tag}{text}</prosody></amazon:emotion></speak>'

        response = polly.synthesize_speech(
            Text=ssml_text,
            OutputFormat=Config.AUDIO_OUTPUT_FORMAT,
            SampleRate=Config.AUDIO_SAMPLE_RATE,
            **voice_settings
        )

        return response['AudioStream'].read()

class PDFToAudioConverter:
    @staticmethod
    def convert(pdf_file, start_page, voice_id, engine, language, aws_client, google_client):
        try:
            text = PDFProcessor.extract_text(pdf_file, start_page=start_page, language_code=Config.SUPPORTED_LANGUAGES[language])
            if not text:
                st.error("Failed to extract text from PDF.")
                return None

            character_names = CharacterAnalyzer.detect_character_names(text, language)
            characters = CharacterAnalyzer.detect_character_attributes(text, language, aws_client, google_client)

            for character in characters:
                if character['name'] in character_names:
                    character['name'] = character_names[character_names.index(character['name'])]

            audio_data = {}
            for character in characters:
                audio = AudioGenerator.generate_character_audio(text, character, voice_id, engine, language, aws_client)
                audio_data[character['name']] = audio

            return audio_data
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            return None

class VoiceManager:
    @staticmethod
    def list_voices(language_code, aws_client):
        polly = aws_client.get_polly_client()
        response = polly.describe_voices(LanguageCode=language_code)
        voices = response['Voices']
        return [(voice['Id'], f"{voice['Id']} ({voice['Gender']})") for voice in voices]

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
