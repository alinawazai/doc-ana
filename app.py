import nest_asyncio
import asyncio
import os
import json
import glob
import logging
from uuid import uuid4
from dotenv import load_dotenv
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
from ultralytics import YOLO
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from nltk.tokenize import word_tokenize
import torch
import nltk
import aiofiles
import asyncio

# Initialize async environment
nest_asyncio.apply()
loop = asyncio.get_event_loop()

# Download required NLTK resource if needed
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# Load environment variables
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
COHERE_API_KEY = st.secrets["COHERE_API_KEY"]

# Directory structure (adjust as needed)
DATA_DIR = "data"
LOW_RES_DIR = os.path.join(DATA_DIR, "40_dpi")   # For detection
HIGH_RES_DIR = os.path.join(DATA_DIR, "500_dpi")   # For cropping
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
from google import genai
client = genai.Client(api_key=GEMINI_API_KEY)

# Set up basic logging (optional)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def log_message(msg):
    st.sidebar.write(msg)

# Asynchronous I/O for file operations
async def pdf_to_images_async(pdf_path, output_dir, fixed_length=1080):
    log_message(f"Converting PDF to images at fixed length {fixed_length}px...")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        log_message(f"Created directory: {output_dir}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        log_message(f"Error opening PDF: {e}")
        raise

    file_paths = []
    # Convert pages to images asynchronously
    tasks = []
    for i in range(len(doc)):
        tasks.append(asyncio.create_task(process_page_async(doc, i, base_name, output_dir)))
    
    file_paths = await asyncio.gather(*tasks)
    doc.close()
    log_message("PDF conversion completed.")
    return file_paths

async def process_page_async(doc, i, base_name, output_dir):
    page = doc[i]
    scale = 1080 / page.rect.width
    matrix = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=matrix)
    image_filename = f"{base_name}_page_{i + 1}.jpg"
    image_path = os.path.join(output_dir, image_filename)
    pix.save(image_path)
    log_message(f"Saved image: {image_path}")
    return image_path

class BlockDetectionModel:
    def __init__(self, weight, device=None):
        self.device = "cuda" if (device is None and torch.cuda.is_available()) else "cpu"
        self.model = YOLO(weight).to(self.device)
        log_message(f"YOLO model loaded on {self.device}.")

    async def predict_batch_async(self, images_dir):
        if not os.path.exists(images_dir) or not os.listdir(images_dir):
            raise ValueError(f"Directory {images_dir} is empty or does not exist.")
        images = glob.glob(os.path.join(images_dir, "*.jpg"))
        log_message(f"Found {len(images)} low-res images for detection.")
        
        output = {}
        batch_size = 10  # Process 10 images at a time
        
        tasks = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            log_message(f"Processing images {i + 1} to {min(i + batch_size, len(images))} of {len(images)}.")
            tasks.append(asyncio.create_task(self.process_batch_async(batch)))
        
        results = await asyncio.gather(*tasks)
        
        # Process results into output
        for result in results:
            for image_name, detection_data in result.items():
                output[image_name] = detection_data
        log_message("Block detection completed.")
        return output

    async def process_batch_async(self, batch):
        results = self.model(batch)
        output = {}
        for result in results:
            image_name = os.path.basename(result.path)
            labels = result.boxes.cls.tolist()
            boxes = result.boxes.xywh.tolist()
            output[image_name] = [{"label": label, "bbox": box} for label, box in zip(labels, boxes)]
        return output

# Main async pipeline function
async def process_pipeline(pdf_path, ocr_prompt):
    log_message("Starting processing pipeline...")
    low_res_paths, high_res_paths = await asyncio.gather(
        pdf_to_images_async(pdf_path, LOW_RES_DIR, 662),
        pdf_to_images_async(pdf_path, HIGH_RES_DIR, 4000)
    )

    log_message("Running YOLO detection on low-res images...")
    yolo_model = BlockDetectionModel("best_small_yolo11_block_etraction.pt")
    detection_results = await yolo_model.predict_batch_async(LOW_RES_DIR)

    log_message("Cropping detected regions using high-res images...")
    cropped_data = await crop_and_save_async(detection_results, OUTPUT_DIR)

    log_message("Extracting metadata using Gemini OCR asynchronously...")
    gemini_documents = await process_all_pages_async(cropped_data, ocr_prompt)

    log_message("Building vector store for semantic search...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    example_embedding = embeddings.embed_query("sample text")
    d = len(example_embedding)
    index = faiss.IndexFlatL2(d)
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    uuids = [str(uuid4()) for _ in range(len(gemini_documents))]
    vector_store.add_documents(documents=gemini_documents, ids=uuids)

    return gemini_documents, vector_store

# Your streamlit buttons, layout, and search code remains the same but wrapped inside async handling
if uploaded_pdf:
    os.makedirs(DATA_DIR, exist_ok=True)
    pdf_path = os.path.join(DATA_DIR, uploaded_pdf.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    st.sidebar.success("PDF uploaded successfully.")

if uploaded_pdf and not st.session_state.processed:
    if st.sidebar.button("Run Processing Pipeline"):
        log_message("Processing pipeline started.")
        gemini_documents, vector_store = await process_pipeline(pdf_path, ocr_prompt)
        st.session_state.processed = True
        st.session_state.gemini_documents = gemini_documents
        st.session_state.vector_store = vector_store
        st.sidebar.success("Processing pipeline completed.")
