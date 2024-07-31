import PyPDF2
import requests
import json
import spacy
import argparse
import configparser
import fitz
import easyocr
import cv2
import os
import shutil

from pathlib import Path
from boto3 import Session
from playsound import playsound
from tqdm import tqdm
from google.cloud import language_v1, speech, vision

# Configurations
AWS_ACCESS_KEY = 'YOUR_ACCESS_KEY'
AWS_SECRET_KEY = 'YOUR_SECRET_KEY'
AWS_REGION = 'YOUR_REGION'
GOOGLE_APPLICATION_CREDENTIALS = 'PATH_TO_GOOGLE_CREDENTIALS_JSON'

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
    # Add more supported languages as needed
}

def debug_print(*args):
    print(*args)

def convert_pdf_to_images(pdf_path, output_folder):
    pdf_file = fitz.open(pdf_path)
    for page_index in range(len(pdf_file)):
        page = pdf_file[page_index]
        image = page.get_pixmap()
        image.save(f"{output_folder}/page_{page_index}.png")
    pdf_file.close()

def detect_margins(image_path, margin_threshold=10):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    margin_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > margin_threshold:
            margin_contours.append(contour)

    if margin_contours:
        min_x, min_y, max_x, max_y = cv2.boundingRect(max(margin_contours, key=cv2.contourArea))
        cropped_image = image[min_y:max_y, min_x:max_x]
        return cropped_image
    else:
        return image

def extract_text_from_image(image, language_codes):
    reader = easyocr.Reader(language_codes)
    text = reader.read_from_cv2_image(image)
    return ' '.join([word[-2] for word in text])

def extract_text(pdf_path, start_page=1, end_page=None, language_codes=[]):
    text_content = ""
    try:
        debug_print("Opening PDF file:", pdf_path)
        pdf_file = fitz.open(pdf_path)
        num_pages = len(pdf_file)
        end_page = num_pages if end_page is None or end_page > num_pages else end_page

        temp_folder = "temp_images"
        os.makedirs(temp_folder, exist_ok=True)
        convert_pdf_to_images(pdf_path, temp_folder)

        progress_update_step = max(1, round((end_page - start_page + 1) * 0.015))  # Calculate progress step for 1.5% updates
        with tqdm(total=end_page - start_page + 1, desc="Extracting text", unit="page") as pbar:
            for page_num in range(start_page - 1, end_page):
                image_path = f"{temp_folder}/page_{page_num}.png"
                cropped_image = detect_margins(image_path)
                text_content += extract_text_from_image(cropped_image, language_codes)

                if (page_num - start_page + 1) % progress_update_step == 0:
                    pbar.update(progress_update_step)
            pbar.update((end_page - start_page + 1) % progress_update_step)  # Ensure the progress bar completes.
    except FileNotFoundError:
        debug_print(f"The file {pdf_path} was not found.")
    except Exception as e:
        debug_print(f"An error occurred while extracting text: {e}")
    finally:
        pdf_file.close()  # Close the PDF file
        shutil.rmtree(temp_folder)  # Remove the temporary image folder
    return text_content

def detect_character_names(text, language_code):
    # Load the pre-trained spaCy model based on the language code
    if language_code == 'en':
        nlp = spacy.load('en_core_web_sm')
    elif language_code == 'es':
        nlp = spacy.load('es_core_news_sm')
    elif language_code == 'fr':
        nlp = spacy.load('fr_core_news_sm')
    else:
        raise ValueError(f"Unsupported language code: {language_code}")

    # Process the text with the spaCy model
    doc = nlp(text)

    # Extract the named entities of type PERSON
    character_names = [entity.text for entity in doc.ents if entity.label_ == 'PERSON']

    return character_names

def detect_character_attributes(text, language_code, aws_region, aws_access_key, aws_secret_key):
    # Using Amazon Comprehend for emotion detection
    comprehend_client = boto3.client('comprehend', region_name=aws_region,
                                     aws_access_key_id=aws_access_key,
                                     aws_secret_access_key=aws_secret_key)
    emotion_response = comprehend_client.detect_sentiment(Text=text, LanguageCode=language_code)
    emotion = emotion_response['Sentiment'].lower()
    
    # Using Google Cloud Natural Language API for entity sentiment analysis
    client = language_v1.LanguageServiceClient()
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT, language=language_code)
    entity_sentiment_response = client.analyze_entity_sentiment(document=document)
    
    characters = []
    for entity in entity_sentiment_response.entities:
        if entity.type_ == language_v1.Entity.Type.PERSON:
            # Using Google Cloud Vision API for gender and age detection
            vision_client = vision.ImageAnnotatorClient()
            image = vision.Image(content=entity.name.encode('utf-8'))
            response = vision_client.face_detection(image=image)
            face = response.face_annotations[0] if response.face_annotations else None
            
            gender = face.gender.lower() if face and face.gender else ""
            age = face.age_range.low if face and face.age_range else ""
            
            # Using Amazon Transcribe for pace detection
            transcribe_client = boto3.client('transcribe', region_name=aws_region,
                                             aws_access_key_id=aws_access_key,
                                             aws_secret_access_key=aws_secret_key)
            job_name = f"pace-detection-{entity.name}"
            transcribe_client.start_transcription_job(
                TranscriptionJobName=job_name,
                LanguageCode=language_code,
                MediaFormat=AUDIO_OUTPUT_FORMAT,
                Media={'MediaFileUri': 'YOUR_AUDIO_FILE_URI'},
                Settings={'ShowSpeakerLabels': True, 'MaxSpeakerLabels': 2}
            )
            while True:
                status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
                if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                    break
            
            if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
                transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
                transcript_data = requests.get(transcript_uri).json()
                pace = transcript_data['results']['speaker_labels'][0]['pace']
            else:
                pace = ""
            
            character = {
                "name": entity.name,
                "gender": gender,
                "age": age,
                "emotion": entity.sentiment.score,
                "pace": pace
            }
            characters.append(character)
    
    return characters

def generate_character_audio(text, character, voice_id, engine, language, aws_region, aws_access_key, aws_secret_key):
    # Set up Amazon Polly client
    session = Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )
    polly = session.client('polly')

    # Configure voice settings based on character attributes
    voice_settings = {
        'Engine': engine,
        'LanguageCode': language,
        'VoiceId': voice_id,
        'TextType': 'ssml'
    }

    # Set emotion and pace based on character attributes
    emotion_tag = f'<amazon:emotion name="{character["emotion"]}" intensity="medium">' if character["emotion"] else ''
    rate_tag = f'<prosody rate="{character["pace"]}">'

    # Generate SSML markup
    ssml_text = f'<speak>{emotion_tag}{rate_tag}{text}</prosody></amazon:emotion></speak>'

    # Synthesize speech
    response = polly.synthesize_speech(
        Text=ssml_text,
        OutputFormat=AUDIO_OUTPUT_FORMAT,
        SampleRate=AUDIO_SAMPLE_RATE,
        **voice_settings
    )

    return response['AudioStream'].read()

def pdf_to_audio(pdf_name, start_page=1, voice_id='Joanna', engine='neural', language='en-US',
                 aws_region=None, aws_access_key=None, aws_secret_key=None):
    try:
        debug_print("Converting PDF to audio...")
        pdf_path = Path(pdf_name)
        if not pdf_path.suffix:
            pdf_path = pdf_path.with_suffix('.pdf')
        debug_print("PDF Path:", pdf_path)

        text = extract_text(pdf_path, start_page=start_page, language_codes=[language])
        if not text:
            debug_print("Failed to extract text from PDF.")
            return None

        character_names = detect_character_names(text, language)
        characters = detect_character_attributes(text, language, aws_region, aws_access_key, aws_secret_key)

        # Assign the detected character names to the characters
        for character in characters:
            if character['name'] in character_names:
                character['name'] = character_names[character_names.index(character['name'])]

        # Generate audio for each character
        audio_paths = []
        for character in characters:
            audio_data = generate_character_audio(text, character, voice_id, engine, language,
                                                  aws_region, aws_access_key, aws_secret_key)

            audio_path = pdf_path.with_name(f"{pdf_path.stem}_{character['name']}_tts.{AUDIO_OUTPUT_FORMAT}")
            audio_path.unlink(missing_ok=True)  # Remove the file if it exists

            debug_print(f"Saving audio for character {character['name']}...")
            with open(str(audio_path), 'wb') as audio_file:
                audio_file.write(audio_data)

            audio_paths.append(audio_path)

        return audio_paths
    except Exception as e:
        debug_print(f"An unexpected error occurred: {e}")
        return None

def play_audio(audio_paths):
    for audio_path in audio_paths:
        try:
            debug_print(f"Playing audio file: {audio_path}")
            playsound(str(audio_path))
        except Exception as e:
            debug_print(f"Failed to play audio: {e}")

def list_voices(language_code='en-US', aws_region=None, aws_access_key=None, aws_secret_key=None):
    session = Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )
    polly = session.client('polly')

    response = polly.describe_voices(LanguageCode=language_code)
    voices = response['Voices']

    print(f"Available voices for language '{language_code}':")
    for voice in voices:
        print(f"Voice ID: {voice['Id']}, Gender: {voice['Gender']}, Age: {voice.get('Age', 'N/A')}")

def list_supported_languages():
    return SUPPORTED_LANGUAGES

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert PDF to Audio')
    parser.add_argument('pdf_name', type=str, help='Name of the PDF file')
    parser.add_argument('--start_page', type=int, default=1, help='PDF page number to start reading from')
    parser.add_argument('--voice_id', type=str, default='Joanna', help='Voice ID for the audio output')
    parser.add_argument('--engine', type=str, default='neural', help='Engine type for audio generation')
    parser.add_argument('--language', type=str, default='en-US', help='Language of the PDF')
    parser.add_argument('--audio_language', type=str, default='en-US', help='Language for audio output')
    args = parser.parse_args()

    debug_print("PDF Name:", args.pdf_name)
    debug_print("Start Page:", args.start_page)
    debug_print("Language:", args.language)
    debug_print("Audio Language:", args.audio_language)

    list_voices(language_code=args.audio_language, aws_region=AWS_REGION,
                aws_access_key=AWS_ACCESS_KEY, aws_secret_key=AWS_SECRET_KEY)

    audio_paths = pdf_to_audio(args.pdf_name, start_page=args.start_page, voice_id=args.voice_id,
                               engine=args.engine, language=args.audio_language,
                               aws_region=AWS_REGION, aws_access_key=AWS_ACCESS_KEY,
                               aws_secret_key=AWS_SECRET_KEY)
    if audio_paths:
        print(f"Audio saved for {len(audio_paths)} characters. Playing now...")
        play_audio(audio_paths)
    else:
        print("Failed to convert PDF to audio.")
