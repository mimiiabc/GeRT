import xml.etree.ElementTree as ET

def lade_metadaten(pfad="metadaten.xml"):
    """Lädt Metadaten aus einer XML-Datei und gibt sie als Dictionary zurück."""
    metadaten = {}
    try:
        tree = ET.parse(pfad)
        root = tree.getroot()
        for eintrag in root.findall("eintrag"):
            dateiname = eintrag.find("datei").text
            beschreibung = eintrag.find("beschreibung").text
            jahr = eintrag.find("jahr").text
            sperrfrist = eintrag.find("sperrfrist").text.lower() == "ja"
            metadaten[dateiname] = {
                "beschreibung": beschreibung,
                "jahr": jahr,
                "sperrfrist": sperrfrist
            }
    except Exception as e:
        raise RuntimeError(f"Fehler beim Laden der Metadaten: {e}")
    return metadaten
