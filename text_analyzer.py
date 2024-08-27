import spacy
import re
from collections import Counter
from config import Config
import streamlit as st
import logging

class TextAnalyzer:
    @staticmethod
    @st.cache_resource
    def load_nlp_model(language_code):
        try:
            if language_code == 'en':
                return spacy.load('en_core_web_sm')
            elif language_code == 'es':
                return spacy.load('es_core_news_sm')
            elif language_code == 'fr':
                return spacy.load('fr_core_news_sm')
            else:
                raise ValueError(f"Unsupported language code: {language_code}")
        except OSError:
            logging.error(f"Language model for {language_code} not found. Please install it using 'python -m spacy download {language_code}_core_web_sm'")
            return None

    @staticmethod
    @st.cache_data
    def detect_character_names(text, language_code):
        nlp = TextAnalyzer.load_nlp_model(language_code)
        if nlp is None:
            return []

        doc = nlp(text)
        character_names = [entity.text for entity in doc.ents if entity.label_ == 'PERSON']
        
        # Count occurrences and sort by frequency
        name_counts = Counter(character_names)
        sorted_names = sorted(name_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Filter out names that appear only once
        return [name for name, count in sorted_names if count > 1]

    @staticmethod
    @st.cache_data
    def split_text_into_chapters(text):
        # Simple chapter detection based on common chapter headings
        chapter_patterns = [
            r'Chapter \d+',
            r'CHAPTER \d+',
            r'\d+\. ',
            r'\d+\.',
            r'Part \d+',
            r'PART \d+'
        ]
        
        combined_pattern = '|'.join(chapter_patterns)
        chapters = re.split(combined_pattern, text)
        
        # Remove empty chapters and strip whitespace
        chapters = [chapter.strip() for chapter in chapters if chapter.strip()]
        
        # Create a dictionary with chapter titles
        chapter_dict = {}
        for i, chapter in enumerate(chapters):
            if i == 0 and not re.match(combined_pattern, text):
                chapter_dict['Introduction'] = chapter
            else:
                match = re.search(combined_pattern, text[text.index(chapter):])
                if match:
                    chapter_title = match.group(0)
                else:
                    chapter_title = f"Chapter {i}"
                chapter_dict[chapter_title] = chapter
        
        return chapter_dict

    @staticmethod
    @st.cache_data
    def extract_keywords(text, language_code, num_keywords=10):
        nlp = TextAnalyzer.load_nlp_model(language_code)
        if nlp is None:
            return []

        doc = nlp(text)
        keywords = [token.text for token in doc if not token.is_stop and not token.is_punct and token.pos_ in ['NOUN', 'PROPN', 'ADJ']]
        keyword_freq = Counter(keywords)
        return keyword_freq.most_common(num_keywords)

    @staticmethod
    @st.cache_data
    def sentiment_analysis(text, language_code):
        nlp = TextAnalyzer.load_nlp_model(language_code)
        if nlp is None:
            return None

        doc = nlp(text)
        
        # This is a very basic sentiment analysis. For more accurate results,
        # you might want to use a dedicated sentiment analysis library.
        positive_words = set(['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic'])
        negative_words = set(['bad', 'terrible', 'awful', 'horrible', 'poor', 'disappointing'])
        
        positive_count = sum(1 for token in doc if token.text.lower() in positive_words)
        negative_count = sum(1 for token in doc if token.text.lower() in negative_words)
        
        if positive_count > negative_count:
            return 'Positive'
        elif negative_count > positive_count:
            return 'Negative'
        else:
            return 'Neutral'

    @staticmethod
    @st.cache_data
    def summarize_text(text, language_code, num_sentences=3):
        nlp = TextAnalyzer.load_nlp_model(language_code)
        if nlp is None:
            return ""

        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]
        
        # Simple summarization by selecting the first few sentences
        # For more sophisticated summarization, consider using a dedicated library
        summary = " ".join(sentences[:num_sentences])
        return summary

if __name__ == "__main__":
    # This block can be used for testing the TextAnalyzer class
    logging.basicConfig(level=logging.INFO)
    st.set_page_config(page_title="Text Analyzer Test", layout="wide")
    st.title("Text Analyzer Test")

    text_input = st.text_area("Enter some text to analyze:", height=200)
    language = st.selectbox("Select language", options=['en', 'es', 'fr'])

    if st.button("Analyze Text"):
        if text_input:
            with st.spinner("Analyzing..."):
                try:
                    characters = TextAnalyzer.detect_character_names(text_input, language)
                    st.write("Detected Characters:", characters)

                    chapters = TextAnalyzer.split_text_into_chapters(text_input)
                    st.write("Chapters:", list(chapters.keys()))

                    keywords = TextAnalyzer.extract_keywords(text_input, language)
                    st.write("Keywords:", keywords)

                    sentiment = TextAnalyzer.sentiment_analysis(text_input, language)
                    st.write("Sentiment:", sentiment)

                    summary = TextAnalyzer.summarize_text(text_input, language)
                    st.write("Summary:", summary)

                    st.success("Text analyzed successfully!")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter some text to analyze.")