import os
import fitz  # pymupdf
from pdf2image import convert_from_path 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
from urllib.parse import quote
import torch
import re
from dotenv import load_dotenv
load_dotenv()
import openai
import shutil
from PIL import Image, ImageDraw
import streamlit as st
import streamlit.components.v1 as components
import json

# === API KEY LADEN ===
openai.api_key = os.getenv("OPENAI_API_KEY")

# === MODELLE LADEN ===
st.info("Modelle werden geladen...")
retriever_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
generator_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
generator_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
st.success("Modelle geladen.")

# === FUNKTIONEN ===

def load_pdfs_from_list(pdf_paths):
    documents = {}
    for file_path in pdf_paths:
        if file_path.endswith('.pdf') and os.path.isfile(file_path):
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            filename = os.path.basename(file_path)
            documents[filename] = text
    if not documents:
        raise FileNotFoundError("Keine passenden PDF-Dateien gefunden.")
    return documents

def split_into_chunks(text, tokenizer, chunk_size=400, overlap=200):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_tokens = []

    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        if len(current_tokens) + len(sentence_tokens) > chunk_size:
            chunk_text = tokenizer.decode(current_tokens, skip_special_tokens=True)
            chunks.append(chunk_text.strip())
            overlap_tokens = current_tokens[-overlap:] if overlap > 0 else []
            current_tokens = list(overlap_tokens)
        current_tokens.extend(sentence_tokens)

    if current_tokens:
        chunk_text = tokenizer.decode(current_tokens, skip_special_tokens=True)
        chunks.append(chunk_text.strip())

    return chunks

def preprocess_documents(documents, model, tokenizer):
    preprocessed = []
    for doc_name, content in documents.items():
        chunks = split_into_chunks(content, tokenizer)
        chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
        for chunk, embedding in zip(chunks, chunk_embeddings):
            preprocessed.append({
                "doc_name": doc_name,
                "chunk": chunk,
                "embedding": embedding
            })
    return preprocessed

def retrieve_relevant_sections(query, preprocessed_data, model, top_k=3):
    query_embedding = model.encode(query, convert_to_tensor=True)
    all_embeddings = torch.stack([entry["embedding"] for entry in preprocessed_data])
    similarities = util.pytorch_cos_sim(query_embedding, all_embeddings).squeeze(0)
    top_indices = torch.topk(similarities, k=top_k).indices
    top_contexts = [(preprocessed_data[i]["doc_name"], preprocessed_data[i]["chunk"]) for i in top_indices]
    return top_contexts

def generate_answer_with_rag_sources(query, top_contexts):
    try:
        context_texts = "\n\n".join(
            [f"Dokument: {doc_name}\n{context}" for doc_name, context in top_contexts]
        )
        prompt = f"""
Du bist ein sachlicher Assistent für historische Dokumente.
Nutze ausschließlich die folgenden Auszüge aus Archivdokumenten zur Beantwortung der Frage.
Gib die Antwort auf Deutsch.

Kontext:
{context_texts}

Frage:
{query}

Antwort:
"""
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Du bist ein Archiv-Assistent."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Fehler bei der OpenAI-Anfrage: {str(e)}"

def highlight_text_in_pdf(input_pdf, texts_to_highlight):
    output_dir = "static/korpus_highlighted"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(input_pdf)
    output_pdf = os.path.join(output_dir, f"highlighted_{base_name}")

    doc = fitz.open(input_pdf)
    matches_total = 0

    for page in doc:
        for text in texts_to_highlight:
            matches = page.search_for(text, hit_max=1000)
            matches_total += len(matches)
            for match in matches:
                annot = page.add_highlight_annot(match)
                annot.set_colors(stroke=(1, 1, 0))
                annot.update()

    doc.save(output_pdf)
    doc.close()
    print(f"{matches_total} Hervorhebungen gespeichert in: {output_pdf}")
    return output_pdf

# === STREAMLIT MAIN ===

def main():
    st.title("Dokumente mit interaktiver PDF-Anzeige")

    pdf_dir = "static/korpus"
    pdf_paths = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    documents = load_pdfs_from_list(pdf_paths)
    preprocessed_data = preprocess_documents(documents, retriever_model, generator_tokenizer)
    
    # from metadaten_utils import lade_metadaten
    # try:
    #     metadaten_dict = lade_metadaten("metadaten.xml")
    # except RuntimeError as e:
    #     st.warning(str(e))
    #     metadaten_dict = {}

    query = st.text_input("Stellen Sie Ihre Frage:")

    if query:
        top_contexts = retrieve_relevant_sections(query, preprocessed_data, retriever_model)

        if top_contexts:
            answer = generate_answer_with_rag_sources(query, top_contexts)
            st.subheader("Antwort:")
            st.write(answer)

            #relevante Abschnitte in PDF
            texts_per_doc = {}
            for doc_name, text in top_contexts:
                texts_per_doc.setdefault(doc_name, []).append(text)

            for doc_name, texts in texts_per_doc.items():
                input_pdf = next(p for p in pdf_paths if os.path.basename(p) == doc_name)
                highlighted_pdf_path = highlight_text_in_pdf(input_pdf, texts)
                rel_path = highlighted_pdf_path.replace("\\", "/")
                quoted_path = quote(rel_path)
                st.markdown(f"[PDF anzeigen]({quoted_path})", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
