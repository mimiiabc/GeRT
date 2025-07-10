# GeRT – Chat zur Kreisgeschichte

GeRT ist ein Retrieval-Augmented-Generation (RAG) basiertes Dialogsystem für Archive, das OCR-erkanne PDF-Dokumente analysiert und archivwissenschaftliche Fragen auf Basis dieser Inhalte beantwortet. Die Anwendung kombiniert Information Retrieval mit einem Large Language Model.

---

## Funktionsweise

1. **PDFs analysieren**: Alle PDF-Dateien im Ordner `static/korpus/` werden automatisch geladen.
2. **Texte in Chunks aufteilen** und **semantisch einbetten** (via `sentence-transformers`)
3. **OpenAI API (GPT-4)** generiert auf Anfrage sachliche Antworten, nur basierend auf archivierten Kontextabschnitten.
4. **Relevante Stellen werden im PDF hervorgehoben** und als Link angezeigt.

---

## Projektstruktur
```text
GeRT/
├── GeRT.py                     # Hauptcode
├── app.py                      # Streamlit-Oberfläche
├── requirements.txt            # Python-Abhängigkeiten
├── .env.example                # Vorlage für Umgebungsvariablen (API Key)
├── metadaten.xml              # Beispielhafte Metadaten (optional)
├── static/korpus/             # Lokaler PDF-Korpus (nicht im Repo)
├── models/                    # Lokale Modelle (nicht im Repo)
└── README.md
```



---

## Installation

1. Repository klonen:
```bash
git clone https://github.com/mimiiabc/GeRT.git
cd GeRT
```
## Virtuelle Umgebung erstellen
```bash
python -m venv .venv
source .venv/bin/activate
```
(unter Windows: .venv\Scripts\activate)

## Abhängigkeiten installieren
pip install -r requirements.txt

---

## **Achtung:**
Dieses Repository enthält aus Speicher- und Lizenzgründen nicht:

- Die PDF-Dokumente (static/korpus/)
- Die Modell-Dateien (models/)

Bitte füge sie manuell hinzu:

static/korpus/PDF-Dateien
models/flan-t5-small/
models/paraphrase-MiniLM-L6-v2/

## **Starten der App**
```bash
streamlit run app.py
```

## Kontakt
Melanie Weber  
weber_melanie@outlook.de  
Projekt im Rahmen des Masterstudiengangs Digital Humanities  
Universität Stuttgart
