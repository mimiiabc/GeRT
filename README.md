# GeRT – Chat zur Kreisgeschichte

GeRT ist ein Retrieval-Augmented-Generation-System (RAG) für Archive, das historische PDF-Dokumente analysiert und archivwissenschaftliche Fragen auf Basis dieser Inhalte beantwortet. Die Anwendung kombiniert moderne Sprachmodelle mit einem semantischen Suchansatz, um archivische Recherche zu unterstützen.

---

## Funktionsweise

1. **PDFs analysieren**: Alle PDF-Dateien im Ordner `static/korpus/` werden automatisch geladen.
2. **Texte in Chunks aufteilen** und **semantisch einbetten** (via `sentence-transformers`)
3. **OpenAI API (GPT-4)** generiert auf Anfrage sachliche Antworten, nur basierend auf archivierten Kontextabschnitten.
4. **Relevante Stellen werden im PDF hervorgehoben** und als Link angezeigt.

---

## Projektstruktur
GeRT/
├── GeRT.py # Hauptcode
├── app.py # Streamlit-Oberfläche
├── requirements.txt # Python-Abhängigkeiten
├── .env.example # Vorlage für Umgebungsvariablen (API Key)
├── metadaten.xml # Beispielhafte Metadaten (optional)
├── static/korpus/ # Lokaler PDF-Korpus (nicht im Repo)
├── models/ # Lokale Modelle (nicht im Repo)
└── README.md
