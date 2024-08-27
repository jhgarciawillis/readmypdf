import fitz
import tempfile
from PIL import Image
import pytesseract
import streamlit as st
from config import Config
import os
import logging

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
        try:
            return pytesseract.image_to_string(Image.open(image_path), lang=language_code)
        except Exception as e:
            logging.error(f"Error extracting text from image: {str(e)}")
            return ""

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
            logging.error(f"Error extracting text from PDF: {str(e)}")
            raise
        finally:
            if 'pdf_document' in locals():
                pdf_document.close()
        return text_content

    @staticmethod
    @st.cache_data
    def get_table_of_contents(pdf_file):
        try:
            pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
            toc = pdf_document.get_toc()
            pdf_document.close()
            return toc
        except Exception as e:
            logging.error(f"Error extracting table of contents: {str(e)}")
            return []

    @staticmethod
    def get_pdf_metadata(pdf_file):
        try:
            pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
            metadata = pdf_document.metadata
            pdf_document.close()
            return metadata
        except Exception as e:
            logging.error(f"Error extracting PDF metadata: {str(e)}")
            return {}

    @staticmethod
    def count_pages(pdf_file):
        try:
            pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
            page_count = len(pdf_document)
            pdf_document.close()
            return page_count
        except Exception as e:
            logging.error(f"Error counting PDF pages: {str(e)}")
            return 0

    @staticmethod
    @st.cache_data
    def extract_images(pdf_file, output_folder):
        try:
            pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
            for i in range(len(pdf_document)):
                for img in pdf_document.get_page_images(i):
                    xref = img[0]
                    pix = fitz.Pixmap(pdf_document, xref)
                    if pix.n < 5:  # this is GRAY or RGB
                        pix.save(f"{output_folder}/image_{xref}.png")
                    else:  # CMYK: convert to RGB first
                        pix1 = fitz.Pixmap(fitz.csRGB, pix)
                        pix1.save(f"{output_folder}/image_{xref}.png")
                        pix1 = None
                    pix = None
            pdf_document.close()
        except Exception as e:
            logging.error(f"Error extracting images from PDF: {str(e)}")
            raise

    @staticmethod
    def sanitize_filename(filename):
        return "".join([c for c in filename if c.isalpha() or c.isdigit() or c in (' ', '-', '_')]).rstrip()

    @staticmethod
    def save_pdf_pages(pdf_file, output_folder, start_page=1, end_page=None):
        try:
            pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
            num_pages = len(pdf_document)
            end_page = num_pages if end_page is None or end_page > num_pages else end_page

            for page_num in range(start_page - 1, end_page):
                page = pdf_document[page_num]
                pix = page.get_pixmap()
                output_file = os.path.join(output_folder, f"page_{page_num + 1}.png")
                pix.save(output_file)

            pdf_document.close()
        except Exception as e:
            logging.error(f"Error saving PDF pages as images: {str(e)}")
            raise

if __name__ == "__main__":
    # This block can be used for testing the PDFProcessor class
    logging.basicConfig(level=logging.INFO)
    st.set_page_config(page_title="PDF Processor Test", layout="wide")
    st.title("PDF Processor Test")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        
        if st.button("Process PDF"):
            with st.spinner("Processing..."):
                try:
                    text = PDFProcessor.extract_text(uploaded_file)
                    st.text_area("Extracted Text", text, height=300)
                    
                    toc = PDFProcessor.get_table_of_contents(uploaded_file)
                    st.write("Table of Contents:", toc)
                    
                    metadata = PDFProcessor.get_pdf_metadata(uploaded_file)
                    st.write("PDF Metadata:", metadata)
                    
                    page_count = PDFProcessor.count_pages(uploaded_file)
                    st.write(f"Total Pages: {page_count}")
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        PDFProcessor.extract_images(uploaded_file, temp_dir)
                        st.write(f"Images extracted to: {temp_dir}")
                        
                        PDFProcessor.save_pdf_pages(uploaded_file, temp_dir, start_page=1, end_page=5)
                        st.write(f"First 5 pages saved as images in: {temp_dir}")
                    
                    st.success("PDF processed successfully!")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")