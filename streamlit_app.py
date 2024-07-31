import streamlit as st
import fitz
import os
import tempfile
from PIL import Image
import pytesseract
from gtts import gTTS
import io
import spacy

class Config:
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

class AudioGenerator:
    @staticmethod
    def generate_character_audio(text, language):
        tts = gTTS(text=text, lang=language)
        audio_io = io.BytesIO()
        tts.write_to_fp(audio_io)
        audio_io.seek(0)
        return audio_io.read()

class PDFToAudioConverter:
    @staticmethod
    def convert(pdf_file, start_page, language):
        try:
            text = PDFProcessor.extract_text(pdf_file, start_page=start_page, language_code=Config.SUPPORTED_LANGUAGES[language])
            if not text:
                st.error("Failed to extract text from PDF.")
                return None

            character_names = CharacterAnalyzer.detect_character_names(text, language)

            audio_data = {}
            for character in character_names:
                audio = AudioGenerator.generate_character_audio(text, language)
                audio_data[character] = audio

            return audio_data
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            return None

def main():
    st.title("PDF to Audio Converter")

    pdf_file = st.file_uploader("Upload PDF file", type="pdf")
    
    if pdf_file:
        start_page = st.number_input("Start Page", min_value=1, value=1)
        language = st.selectbox("PDF Language", options=list(Config.SUPPORTED_LANGUAGES.keys()), format_func=lambda x: Config.SUPPORTED_LANGUAGES[x])
        
        if st.button("Convert to Audio"):
            with st.spinner("Converting PDF to audio..."):
                audio_data = PDFToAudioConverter.convert(pdf_file, start_page, language)
            
            if audio_data:
                st.success(f"Audio generated for {len(audio_data)} characters.")
                for character, audio in audio_data.items():
                    st.audio(audio, format='audio/mp3')
            else:
                st.error("Failed to convert PDF to audio.")

if __name__ == "__main__":
    main()
