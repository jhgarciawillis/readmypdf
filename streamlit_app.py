import streamlit as st
import os
from pdf_processor import PDFProcessor
from text_analyzer import TextAnalyzer
from audio_generator import AudioGenerator
from config import Config
from ui_components import UIComponents

def main():
    st.set_page_config(page_title="Enhanced PDF to Audio Converter", layout="wide")
    st.title("Enhanced PDF to Audio Converter")

    # Sidebar for file selection and settings
    with st.sidebar:
        st.header("Settings")
        file_source = st.radio("Choose file source:", ("Upload", "Local File"))
        
        if file_source == "Upload":
            pdf_file = st.file_uploader("Upload PDF file", type="pdf")
        else:
            pdf_file = UIComponents.file_selector()

        language = st.selectbox("PDF Language", options=list(Config.SUPPORTED_LANGUAGES.keys()), 
                                format_func=lambda x: Config.SUPPORTED_LANGUAGES[x])

        rate = st.slider("Speech Rate", 0.5, 2.0, 1.0, 0.1)
        pitch = st.slider("Speech Pitch", 0.5, 2.0, 1.0, 0.1)

    if pdf_file:
        st.header("PDF Processing")
        
        if st.button("Convert to Audio"):
            with st.spinner("Processing PDF..."):
                try:
                    # Extract text from PDF
                    text = PDFProcessor.extract_text(pdf_file, language_code=Config.SUPPORTED_LANGUAGES[language])
                    
                    # Get table of contents
                    toc = PDFProcessor.get_table_of_contents(pdf_file)
                    
                    # Split text into chapters
                    chapters = TextAnalyzer.split_text_into_chapters(text)
                    
                    # Generate audio for each chapter
                    audio_data = {}
                    for chapter, chapter_text in chapters.items():
                        audio = AudioGenerator.generate_character_audio(chapter_text, language, rate, pitch)
                        audio_data[chapter] = audio

                    st.success("PDF processed and audio generated successfully!")

                    # Store processed data in session state
                    st.session_state['audio_data'] = audio_data
                    st.session_state['toc'] = toc
                    st.session_state['current_chapter'] = list(chapters.keys())[0]
                    st.session_state['current_position'] = 0

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    return

        # Display results if audio data is available
        if 'audio_data' in st.session_state:
            st.header("Audio Book")

            # Display table of contents
            col1, col2 = st.columns([1, 3])
            with col1:
                st.subheader("Table of Contents")
                UIComponents.table_of_contents_view(st.session_state['toc'], st.session_state['current_chapter'])

            with col2:
                st.subheader("Audio Player")
                # Audio controller
                current_audio = st.session_state['audio_data'][st.session_state['current_chapter']]
                UIComponents.audio_controller(current_audio)

                # Progress bar
                UIComponents.progress_bar(st.session_state['current_position'], len(current_audio))

                # Bookmarking
                if st.button("Add Bookmark"):
                    UIComponents.add_bookmark(st.session_state['current_chapter'], st.session_state['current_position'])

                st.subheader("Bookmarks")
                UIComponents.display_bookmarks()

    else:
        st.info("Please upload a PDF file or select a local file to get started.")

if __name__ == "__main__":
    main()