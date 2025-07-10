import streamlit as st
st.set_page_config(page_title="GeRT - Chat zur Kreisgeschichte", layout="wide")
import os
import shutil
import zipfile
import streamlit.components.v1 as components
from GeRT import load_pdfs_from_list, preprocess_documents, retrieve_relevant_sections, generate_answer_with_rag_sources
from pdf2image import convert_from_path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import xml.etree.ElementTree as ET
import openai
import re
import base64
from PIL import Image

openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Style & Header ---
st.markdown("""
    <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #1d4b93;
            text-align: center;
            margin-bottom: 20px;
        }
        .container {
            padding: 20px;
            background-color: #f4f6f9;
            border-radius: 8px;
        }
    </style>
    <div class="container">
        <h1 class="title">GeRT - Chat zur Kreisgeschichte</h1>
    </div>
""", unsafe_allow_html=True)


#Modelle laden
with st.spinner():
    retriever_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    generator_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    generator_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

#Verzeichnis und PDFs
pdf_dir = "static/korpus"
pdf_paths = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

try:
    documents = load_pdfs_from_list(pdf_paths)
    preprocessed_data = preprocess_documents(documents, retriever_model, generator_tokenizer)
except FileNotFoundError as e:
    st.error(f"{str(e)}")
    documents = {}
    preprocessed_data = []

#PDFHighlighting für Schlüsselbegriffe
def better_highlight(pdf_path, highlight_text):
    output_dir = "static/korpus_highlighted"
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(pdf_path)
    output_path = os.path.join(output_dir, base_name.replace(".pdf", "_highlighted.pdf"))

    doc = fitz.open(pdf_path)
    keywords = highlight_text.split()
    keywords = [k.strip(".,;:-!?()") for k in keywords if len(k) > 4][:5] 

    found_any = False
    for page in doc:
        for word in keywords:
            matches = page.search_for(word)
            if matches:
                found_any = True
            for match in matches:
                annot = page.add_highlight_annot(match)
                annot.set_colors(stroke=(1, 1, 0))
                annot.update()

    if found_any:
        doc.save(output_path)
        print(f"PDF gespeichert: {output_path}")
        result = output_path
    else:
        print(f"Keine Schlüsselbegriffe gefunden in: {pdf_path}")
        result = None

    doc.close()
    return result

#Main
query = st.text_input(" **Stelle eine Frage an GeRT:**")

highlighted_pdf_path = None

if query:
    with st.spinner("Suche nach passenden Antworten..."):
        top_contexts = retrieve_relevant_sections(query, preprocessed_data, retriever_model)

    if top_contexts:
        answer = generate_answer_with_rag_sources(query, top_contexts)

        st.markdown("""
            <div style=\"border: 2px solid #1d4b93; padding: 20px; border-radius: 8px; background-color: #f4f6f9;\">
                <h3 style=\"text-align: center;\">Antwort:</h3>
                <p style=\"font-size: 16px;\">{}</p>
            </div>
        """.format(answer), unsafe_allow_html=True)

        with st.expander("Relevante Textstellen anzeigen"):
            for idx, (doc_name, ctx) in enumerate(top_contexts):
                st.markdown(f"**({idx+1}) {doc_name}**:\n\n{ctx[:500]}...", unsafe_allow_html=True)

        best_doc = top_contexts[0][0]
        best_text = top_contexts[0][1]

        try:
            input_pdf = next(p for p in pdf_paths if os.path.basename(p) == best_doc)
            highlighted_pdf_path = better_highlight(input_pdf, best_text)
        except StopIteration:
            st.error(f"PDF-Datei '{best_doc}' nicht im Verzeichnis gefunden.")
            st.stop()

if highlighted_pdf_path:

    st.markdown("### Abschnitt in der Originalquelle:")
    try:
        all_images = convert_from_path(highlighted_pdf_path, dpi=150)
        num_pages = len(all_images)
        page_to_show = st.slider("Wähle eine Seite zur Anzeige", 1, num_pages, 1)
        st.image(all_images[page_to_show - 1], use_container_width=True)
    except Exception as e:
        st.error(f"Fehler bei der PDF-Anzeige: {e}")
else:
    if query:
        st.warning("Der Text konnte in der PDF nicht gefunden und hervorgehoben werden.")
