import os
from pathlib import Path

class Config:
    # Supported languages for OCR and text-to-speech
    SUPPORTED_LANGUAGES = {
        'en': 'eng',  # English
        'es': 'spa',  # Spanish
        'fr': 'fra',  # French
        'de': 'deu',  # German
        'it': 'ita',  # Italian
        'ru': 'rus',  # Russian
        'zh-CN': 'chi_sim',  # Simplified Chinese
        'ja': 'jpn',  # Japanese
        'ar': 'ara',  # Arabic
        'hi': 'hin',  # Hindi
        'pt': 'por',  # Portuguese
        'nl': 'nld'   # Dutch
    }

    # Application settings
    APP_NAME = "PDF to Audio Converter"
    VERSION = "1.0.0"

    # File paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    CACHE_DIR = BASE_DIR / '.cache'
    LOG_DIR = BASE_DIR / 'logs'
    TEMP_DIR = BASE_DIR / 'temp'
    OUTPUT_DIR = BASE_DIR / 'output'

    # Ensure directories exist
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Logging configuration
    LOG_FILE = LOG_DIR / 'app.log'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_LEVEL = 'INFO'

    # PDF processing settings
    MAX_PDF_SIZE = 50 * 1024 * 1024  # 50 MB
    DEFAULT_START_PAGE = 1
    DEFAULT_END_PAGE = None  # None means process all pages

    # Audio settings
    DEFAULT_AUDIO_FORMAT = 'mp3'
    DEFAULT_AUDIO_QUALITY = '192k'
    MAX_AUDIO_LENGTH = 10 * 60  # 10 minutes

    # Text processing settings
    MAX_TEXT_LENGTH = 5000  # characters
    SUMMARY_LENGTH = 3  # sentences

    # UI settings
    THEME = 'light'
    PRIMARY_COLOR = '#F63366'
    BACKGROUND_COLOR = '#FFFFFF'
    SECONDARY_BACKGROUND_COLOR = '#F0F2F6'
    TEXT_COLOR = '#262730'
    FONT = 'sans serif'

    # API keys (replace with your actual keys or use environment variables)
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-openai-api-key')
    GOOGLE_CLOUD_API_KEY = os.getenv('GOOGLE_CLOUD_API_KEY', 'your-google-cloud-api-key')

    # Performance settings
    CACHE_TTL = 3600  # 1 hour
    MAX_WORKERS = 4  # for concurrent processing

    @classmethod
    def get_supported_language_names(cls):
        return {
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

    @classmethod
    def get_language_code(cls, language_name):
        for code, name in cls.get_supported_language_names().items():
            if name.lower() == language_name.lower():
                return code
        raise ValueError(f"Unsupported language: {language_name}")

    @classmethod
    def get_tesseract_language_code(cls, language_code):
        return cls.SUPPORTED_LANGUAGES.get(language_code, 'eng')

if __name__ == "__main__":
    print(f"Application Name: {Config.APP_NAME}")
    print(f"Version: {Config.VERSION}")
    print(f"Supported Languages: {Config.get_supported_language_names()}")
    print(f"Cache Directory: {Config.CACHE_DIR}")
    print(f"Log File: {Config.LOG_FILE}")
    print(f"Maximum PDF Size: {Config.MAX_PDF_SIZE / 1024 / 1024} MB")
    print(f"Default Audio Format: {Config.DEFAULT_AUDIO_FORMAT}")
    print(f"Maximum Audio Length: {Config.MAX_AUDIO_LENGTH / 60} minutes")