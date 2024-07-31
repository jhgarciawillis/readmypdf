import streamlit as st
import PyPDF2
import requests
import json
import spacy
import fitz
import easyocr
import cv2
import os
import tempfile
import boto3

from pathlib import Path
from boto3 import Session
from tqdm import tqdm
from google.cloud import language_v1, speech, vision

# Configurations
AWS_ACCESS_KEY = st.secrets["AWS_ACCESS_KEY"]
AWS_SECRET_KEY = st.secrets["AWS_SECRET_KEY"]
AWS_REGION = st.secrets["AWS_REGION"]
GOOGLE_APPLICATION_CREDENTIALS = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]

AUDIO_OUTPUT_FORMAT = 'mp3'
AUDIO_SAMPLE_RATE = '24000'

SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'ru': 'Russian',
    'zh-CN': 'Chinese (Simplified)',
    'ja': 'Japanese',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'pt': 'Portuguese',
    'nl': 'Dutch'
}

@st.cache_data
def convert_pdf_to_images(pdf_file, output_folder):
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page_index in range(len(pdf_document)):
        page = pdf_document[page_index]
        image = page.get_pixmap()
        image.save(f"{output_folder}/page_{page_index}.png")
    pdf_document.close()

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

@st.cache_data
def extract_text_from_image(image, language_codes):
    reader = easyocr.Reader(language_codes)
    text = reader.read_from_cv2_image(image)
    return ' '.join([word[-2] for word in text])

@st.cache_data
def extract_text(pdf_file, start_page=1, end_page=None, language_codes=[]):
    text_content = ""
    try:
        with tempfile.TemporaryDirectory() as temp_folder:
            convert_pdf_to_images(pdf_file, temp_folder)
            
            pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
            num_pages = len(pdf_document)
            end_page = num_pages if end_page is None or end_page > num_pages else end_page

            progress_bar = st.progress(0)
            for page_num in range(start_page - 1, end_page):
                image_path = f"{temp_folder}/page_{page_num}.png"
                cropped_image = detect_margins(image_path)
                text_content += extract_text_from_image(cropped_image, language_codes)
                progress = (page_num - start_page + 2) / (end_page - start_page + 1)
                progress_bar.progress(progress)

    except Exception as e:
        st.error(f"An error occurred while extracting text: {e}")
    finally:
        pdf_document.close()
    return text_content

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

def detect_character_attributes(text, language_code):
    comprehend_client = boto3.client('comprehend', region_name=AWS_REGION,
                                     aws_access_key_id=AWS_ACCESS_KEY,
                                     aws_secret_access_key=AWS_SECRET_KEY)
    emotion_response = comprehend_client.detect_sentiment(Text=text, LanguageCode=language_code)
    emotion = emotion_response['Sentiment'].lower()
    
    client = language_v1.LanguageServiceClient()
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT, language=language_code)
    entity_sentiment_response = client.analyze_entity_sentiment(document=document)
    
    characters = []
    for entity in entity_sentiment_response.entities:
        if entity.type_ == language_v1.Entity.Type.PERSON:
            vision_client = vision.ImageAnnotatorClient()
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
                "pace": ""  # Pace detection removed as it required audio input
            }
            characters.append(character)
    
    return characters

def generate_character_audio(text, character, voice_id, engine, language):
    session = Session(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )
    polly = session.client('polly')

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
        OutputFormat=AUDIO_OUTPUT_FORMAT,
        SampleRate=AUDIO_SAMPLE_RATE,
        **voice_settings
    )

    return response['AudioStream'].read()

def pdf_to_audio(pdf_file, start_page=1, voice_id='Joanna', engine='neural', language='en-US'):
    try:
        text = extract_text(pdf_file, start_page=start_page, language_codes=[language])
        if not text:
            st.error("Failed to extract text from PDF.")
            return None

        character_names = detect_character_names(text, language)
        characters = detect_character_attributes(text, language)

        for character in characters:
            if character['name'] in character_names:
                character['name'] = character_names[character_names.index(character['name'])]

        audio_data = {}
        for character in characters:
            audio = generate_character_audio(text, character, voice_id, engine, language)
            audio_data[character['name']] = audio

        return audio_data
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

def list_voices(language_code='en-US'):
    session = Session(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )
    polly = session.client('polly')

    response = polly.describe_voices(LanguageCode=language_code)
    voices = response['Voices']

    return [(voice['Id'], f"{voice['Id']} ({voice['Gender']})") for voice in voices]

def main():
    st.title("PDF to Audio Converter")

    pdf_file = st.file_uploader("Upload PDF file", type="pdf")
    
    if pdf_file:
        start_page = st.number_input("Start Page", min_value=1, value=1)
        language = st.selectbox("PDF Language", options=list(SUPPORTED_LANGUAGES.keys()), format_func=lambda x: SUPPORTED_LANGUAGES[x])
        audio_language = st.selectbox("Audio Language", options=list(SUPPORTED_LANGUAGES.keys()), format_func=lambda x: SUPPORTED_LANGUAGES[x])
        
        voices = list_voices(language_code=audio_language)
        voice_id = st.selectbox("Select Voice", options=[v[0] for v in voices], format_func=lambda x: dict(voices)[x])
        
        engine = st.selectbox("Engine", options=["neural", "standard"])

        if st.button("Convert to Audio"):
            with st.spinner("Converting PDF to audio..."):
                audio_data = pdf_to_audio(pdf_file, start_page=start_page, voice_id=voice_id, engine=engine, language=audio_language)
            
            if audio_data:
                st.success(f"Audio generated for {len(audio_data)} characters.")
                for character, audio in audio_data.items():
                    st.audio(audio, format='audio/mp3')
            else:
                st.error("Failed to convert PDF to audio.")

if __name__ == "__main__":
    main()
