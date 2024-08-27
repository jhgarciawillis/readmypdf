import streamlit as st
import os
from config import Config
import base64
from io import BytesIO

class UIComponents:
    @staticmethod
    def file_selector(folder_path='.'):
        filenames = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
        selected_filename = st.selectbox('Select a PDF file', filenames)
        return os.path.join(folder_path, selected_filename)

    @staticmethod
    def audio_controller(audio_data, key=None):
        # Custom HTML for audio player with controls
        audio_str = f"data:audio/mp3;base64,{base64.b64encode(audio_data).decode()}"
        audio_html = f"""
            <audio id="audio-{key}" style="width:100%;">
                <source src="{audio_str}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
            <div style="display:flex; justify-content:space-between; width:100%;">
                <button onclick="document.getElementById('audio-{key}').play()">Play</button>
                <button onclick="document.getElementById('audio-{key}').pause()">Pause</button>
                <button onclick="document.getElementById('audio-{key}').currentTime-=10">-10s</button>
                <button onclick="document.getElementById('audio-{key}').currentTime+=10">+10s</button>
                <button onclick="document.getElementById('audio-{key}').playbackRate-=0.1">Slower</button>
                <button onclick="document.getElementById('audio-{key}').playbackRate+=0.1">Faster</button>
            </div>
        """
        st.components.v1.html(audio_html, height=100)

    @staticmethod
    def progress_bar(current_time, total_time):
        progress = current_time / total_time if total_time > 0 else 0
        st.progress(progress)
        st.write(f"Time: {current_time:.2f}s / {total_time:.2f}s")

    @staticmethod
    def add_bookmark(chapter, position):
        if 'bookmarks' not in st.session_state:
            st.session_state.bookmarks = []
        st.session_state.bookmarks.append((chapter, position))
        st.success(f"Bookmark added at {position:.2f}s in {chapter}")

    @staticmethod
    def display_bookmarks():
        if 'bookmarks' in st.session_state and st.session_state.bookmarks:
            for i, (chapter, position) in enumerate(st.session_state.bookmarks):
                st.write(f"{i+1}. {chapter} at {position:.2f}s")
                if st.button(f"Go to bookmark {i+1}"):
                    # Logic to jump to the bookmark position
                    pass
        else:
            st.info("No bookmarks added yet.")

    @staticmethod
    def table_of_contents_view(toc_data, current_chapter):
        st.subheader("Table of Contents")
        for chapter in toc_data:
            if chapter == current_chapter:
                st.markdown(f"**{chapter}**")
            else:
                if st.button(chapter):
                    # Logic to jump to the selected chapter
                    st.session_state.current_chapter = chapter

    @staticmethod
    def language_selector():
        languages = Config.get_supported_language_names()
        selected_language = st.selectbox("Select Language", list(languages.values()))
        return Config.get_language_code(selected_language)

    @staticmethod
    def audio_settings():
        col1, col2 = st.columns(2)
        with col1:
            rate = st.slider("Speech Rate", 0.5, 2.0, 1.0, 0.1)
        with col2:
            pitch = st.slider("Speech Pitch", 0.5, 2.0, 1.0, 0.1)
        return rate, pitch

    @staticmethod
    def pdf_page_selector(num_pages):
        col1, col2 = st.columns(2)
        with col1:
            start_page = st.number_input("Start Page", min_value=1, max_value=num_pages, value=1)
        with col2:
            end_page = st.number_input("End Page", min_value=start_page, max_value=num_pages, value=num_pages)
        return start_page, end_page

    @staticmethod
    def display_pdf_preview(pdf_file):
        base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

    @staticmethod
    def error_message(message):
        st.error(message)

    @staticmethod
    def success_message(message):
        st.success(message)

    @staticmethod
    def info_message(message):
        st.info(message)

    @staticmethod
    def warning_message(message):
        st.warning(message)

if __name__ == "__main__":
    st.set_page_config(page_title="UI Components Test", layout="wide")
    st.title("UI Components Test")

    # Test file selector
    st.subheader("File Selector")
    selected_file = UIComponents.file_selector()
    st.write(f"Selected file: {selected_file}")

    # Test audio controller
    st.subheader("Audio Controller")
    audio_file = st.file_uploader("Upload an audio file", type=['mp3'])
    if audio_file is not None:
        audio_bytes = audio_file.read()
        UIComponents.audio_controller(audio_bytes, key="test_audio")

    # Test progress bar
    st.subheader("Progress Bar")
    current_time = st.slider("Current Time", 0.0, 100.0, 50.0)
    UIComponents.progress_bar(current_time, 100.0)

    # Test bookmarks
    st.subheader("Bookmarks")
    if st.button("Add Bookmark"):
        UIComponents.add_bookmark("Test Chapter", current_time)
    UIComponents.display_bookmarks()

    # Test table of contents
    st.subheader("Table of Contents")
    test_toc = ["Chapter 1", "Chapter 2", "Chapter 3"]
    UIComponents.table_of_contents_view(test_toc, "Chapter 2")

    # Test language selector
    st.subheader("Language Selector")
    selected_lang = UIComponents.language_selector()
    st.write(f"Selected language code: {selected_lang}")

    # Test audio settings
    st.subheader("Audio Settings")
    rate, pitch = UIComponents.audio_settings()
    st.write(f"Rate: {rate}, Pitch: {pitch}")

    # Test PDF page selector
    st.subheader("PDF Page Selector")
    start, end = UIComponents.pdf_page_selector(100)
    st.write(f"Selected pages: {start} to {end}")

    # Test messages
    st.subheader("Messages")
    UIComponents.error_message("This is an error message")
    UIComponents.success_message("This is a success message")
    UIComponents.info_message("This is an info message")
    UIComponents.warning_message("This is a warning message")