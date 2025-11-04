from data import load_training_data, load_and_append_data, fetch_wikipedia_summary, fetch_wikipedia_variants, fetch_wikipedia_page_text, fetch_wiktionary_definition, extract_subject_from_question, append_data
import os
import json
import nltk
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import requests
import torch
import torch.nn as nn
from PyQt5.QtWidgets import QApplication
from urllib.parse import quote
import unicodedata
import sys
import subprocess
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk
import matplotlib.pyplot as plt
import re
import mysql.connector
import logging

class LATINInstance:
    def __init__(self, gibberlink):
        self.gibberlink = gibberlink
    def preprocess_text(self, text):
        try:
            self.tokens = word_tokenize(text)
            self.stemmed = [self.stemmer.stem(token) for token in self.tokens]
            self.lemmatized = [self.lemmatizer.lemmatize(token) for token in self.tokens]
            self.pos_tags = pos_tag(self.tokens)
            self.named_entities = ne_chunk(self.pos_tags)
            return {
                "tokens": self.tokens,
                "stemmed": self.stemmed,
                "lemmatized": self.lemmatized,
                "pos_tags": self.pos_tags,
                "named_entities": self.named_entities
            }
        except Exception as e:
            return {"error": str(e)}

    def processTrainingData(self):
        training_data = load_training_data()
        training_data = load_and_append_data(training_data, './assets/comics.json', 'comicSeries')
        training_data = load_and_append_data(training_data, './assets/dishes.json', 'dishes')
        training_data = load_and_append_data(training_data, './assets/books.json', 'books')
        training_data = load_and_append_data(training_data, './assets/movies.json', 'movies')
        training_data = load_and_append_data(training_data, './assets/fruits.json', 'fruits')
        training_data = load_and_append_data(training_data, './assets/animals.json', 'animals')
        training_data = load_and_append_data(training_data, './assets/windows.json', 'windowsVersions')
        training_data = load_and_append_data(training_data, './assets/deutsch6klassebayern.json', 'deutsch6klassebayern')
        training_data = load_and_append_data(training_data, './assets/superMarioGames.json', 'superMarioGames')
        training_data = load_and_append_data(training_data, './assets/informatik6klassebayern.json', 'informatik6klassebayern')
        training_data = load_and_append_data(training_data, './assets/mathematik6klassebayern.json', 'mathematik6klassebayern')
        training_data = self.install_addon_datasets(training_data, './addons')
        self.training_data = training_data
        return training_data
    def loadTrainingDataManually(self, training_data, path):
        try:
            path = str(path)
        except Exception as e:
            print(e)
            return False
        training_data = load_and_append_data(training_data, path)
        return training_data
    def TestIntegrity(self):
        TestInstance = LATINInstance(True)
        print(TestInstance.gibberlink)
        training_test_data= TestInstance.processTrainingData()
        TestInstance.prepareInstance(training_test_data)
        print(TestInstance.vectorizer)
        TestInstance.initTransformer()
        del TestInstance
    def install_addon_datasets(self, training_data, addons_dir='./addons'):
        """
        Scan addons_dir for .json or .mintaiaddon files that contain dataset JSON and append them.
        Uses load_and_append_data which will parse the file and append heuristically.
        """
        if not os.path.isdir(addons_dir):
            return training_data
        for fname in sorted(os.listdir(addons_dir)):
            if not (fname.lower().endswith('.json') or fname.lower().endswith('.mintaiaddon')):
                continue
            path = os.path.join(addons_dir, fname)
            try:
                # try to append the file as JSON dataset; load_and_append_data will open and infer structure
                training_data = load_and_append_data(training_data, path, key=None)
                print(f"Addon-Dataset angehängt: {path}")
            except Exception as e:
                print(f"Fehler beim Anhängen von Addon {path}: {e}")
        return training_data
    def prepareInstance(self, training_data):
        if not training_data:
            raise ValueError("Das Trainingsdatenset ist leer. Bitte überprüfen Sie die Quelle der Daten.")

        # Daten vorverarbeiten
        self.vectorizer = TfidfVectorizer()
        self.questions = [data['question'] for data in training_data]
        self.X = self.vectorizer.fit_transform(self.questions)

        self.answers = [data['answer'] for data in training_data]
        self.model = SVC(kernel='linear')
        self.model.fit(self.X, self.answers)

        # Zusatzfunktionen für NLP
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.modelofAI = 2
        return
    def initTransformer(self):
        # Initialisiere das Transformer-Modell
        input_dim = len(self.vectorizer.vocabulary_)
        model_dim = 512
        num_heads = 8
        num_layers = 6
        output_dim = len(set(self.answers))
        self.transformer_model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim)
    def add_to_training_data(self, question, answer, training_data, file_path='./assets/training_data.json'):
        new_entry = {"question": question, "answer": answer}
        training_data.append(new_entry)
    
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(training_data, file, ensure_ascii=False, indent=4)
    
        # Modelle neu trainieren
        self.questions = [data['question'] for data in training_data]
        X = self.vectorizer.fit_transform(self.questions)
        self.answers = [data['answer'] for data in training_data]
        self.model.fit(X, self.answers)
    def learn_from_interaction(self, user_input, expected_response):
        self.add_to_training_data(user_input, expected_response)
    def generate_math_plot(self, expression, filename="plot.png"):
        try:
            # Beispiel: Parabel zeichnen
            if expression == "parabel":
                x = np.linspace(-10, 10, 400)
                y = x**2
                plt.figure()
                plt.plot(x, y, label="y = x^2")
                plt.title("Parabel")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.axhline(0, color='black',linewidth=0.5)
                plt.axvline(0, color='black',linewidth=0.5)
                plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
                plt.legend()
                plt.savefig(filename)
                plt.close()
                return filename
            else:
                return None
        except Exception as e:
            print(f"Fehler beim Generieren der Grafik: {e}")
            return None
    def visualize_calculation_steps(self, expression, filename="calculation_steps.png"):
        try:
            # Zerlege den Ausdruck in Schritte
            steps = []
            current_expression = expression

            # Berechne Schritt für Schritt unter Berücksichtigung der Reihenfolge der Operationen
            while True:
                # Finde die innerste Klammer oder den nächsten Operator
                match = re.search(r"\(([^()]+)\)", current_expression)  # Suche nach Klammern
                if match:
                    sub_expression = match.group(1)
                    result = eval(sub_expression)
                    steps.append((f"{sub_expression} = {result}", result))
                    current_expression = current_expression.replace(f"({sub_expression})", str(result))
                else:
                    # Keine Klammern mehr, berechne den Rest
                    result = eval(current_expression)
                    steps.append((f"{current_expression} = {result}", result))
                    break
        except Exception as e:
            print("Failed to visualize:" + e)
    def evaluate_math_expression(self, expression):
        try:
            # Berechne das Ergebnis
            result = eval(expression)

            # Visualisiere die Rechenschritte
            filename = self.visualize_calculation_steps(expression)
            if filename:
                self.show_plot_in_gui(filename)

            return f"Das Ergebnis ist: {result}"
        except Exception as e:
            return f"Fehler beim Auswerten des Ausdrucks: {str(e)}"
    def search_web(self, query):
        query = "Unimplemented Feature"
        return query
    def visualize_calculation_steps(self, expression, filename="calculation_steps.png"):
        try:
            # Zerlege den Ausdruck in Schritte
            steps = []
            current_expression = expression

            # Berechne Schritt für Schritt unter Berücksichtigung der Reihenfolge der Operationen
            while True:
                # Finde die innerste Klammer oder den nächsten Operator
                match = re.search(r"\(([^()]+)\)", current_expression)  # Suche nach Klammern
                if match:
                    sub_expression = match.group(1)
                    result = eval(sub_expression)
                    steps.append((f"{sub_expression} = {result}", result))
                    current_expression = current_expression.replace(f"({sub_expression})", str(result))
                else:
                    # Keine Klammern mehr, berechne den Rest
                    result = eval(current_expression)
                    steps.append((f"{current_expression} = {result}", result))
                    break

            # Erstelle die Grafik
            plt.figure(figsize=(10, 6))
            y_pos = range(len(steps))
            expressions = [step[0] for step in steps]
            results = [step[1] for step in steps]

            plt.barh(y_pos, results, color='skyblue')
            plt.yticks(y_pos, expressions)
            plt.xlabel("Ergebnisse")
            plt.title("Rechenschritte")
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
            return filename
        except Exception as e:
            print(f"Fehler bei der Visualisierung der Rechenschritte: {e}")
            return None


    # --- NEW: extract measurement snippets from text and try wiki variants ---
    def extract_measurement_from_text(self, text):
        if not text:
            return None
        # normalize whitespace
        self.txt = re.sub(r'\s+', ' ', text)
        # common unit patterns (DE + EN) and numeric ranges
        self.patterns = [
            r'(\d{1,4}(?:[.,]\d+)?\s?(?:m|Meter|Metern|meter|metres|meters|cm|Zentimeter|centimeter|kg|Kilogramm|g|Gramm|lbs|pounds))',
            r'(\d{1,4}(?:[.,]\d+)?\s?(?:cm|Zentimeter|centimeter))',
            r'(\d{1,4}(?:[.,]\d+)?\s?(?:kg|Kilogramm|g|Gramm|lbs|pounds))',
            r'(\d+(?:[.,]\d+)?\s?[-–]\s?\d+(?:[.,]\d+)?\s?(?:m|cm|ft|in|mm))',
            r'(bis\s+zu\s+\d+(?:[.,]\d+)?\s?(?:m|cm|kg))',
            r'(etwa\s+\d+(?:[.,]\d+)?\s?(?:m|cm|kg))',
            r'(~\s?\d+(?:[.,]\d+)?\s?(?:m|cm|kg))',
        ]
        for p in self.patterns:
            self.m = re.search(p, self.txt, flags=re.IGNORECASE)
            if self.m:
                return self.m.group(1).strip()
        # phrases like "average/typical length is 4.5 m" (EN/DE)
        self.m = re.search(r'(?:average|typical|mean|durchschnittlich|im Durchschnitt)\s+(?:length|Länge)\s*(?:is|ist|:)?\s*([\d.,]+\s?(?:m|cm|metres|meters|ft|feet|in|inch|Zentimeter|Zoll))', self.txt, flags=re.IGNORECASE)
        if self.m:
            return self.m.group(1).strip()
        # "length is X m" or "ist X m"
        self.m = re.search(r'(?:length|Länge)\s*(?:is|ist|:)?\s*([\d.,]+\s?(?:m|cm|ft|in|metres|meters|Zentimeter|Zoll))', self.txt, flags=re.IGNORECASE)
        if self.m:
            return self.m.group(1).strip()
        # fallback: "ist X lang" style
        self.m = re.search(r'ist\s+(?:etwa|ungefähr|ca\.?|circa)?\s*([\d.,\s\-–]+)\s?(m|cm|kg|Meter|Zentimeter|Kilogramm|Zoll|inch|feet)', self.txt, flags=re.IGNORECASE)
        if self.m:
            self.num = self.m.group(1).strip().replace(' ', '')
            self.unit = self.m.group(2).strip()
            return f"{self.num}{self.unit}"
        return None

    # --- NEW: normalize subject and map common synonyms ---
    def normalize_subject(self, subj):
        if not subj:
            return None
        # remove diacritics, collapse whitespace, lowercase
        self.s = unicodedata.normalize('NFKD', subj)
        self.s = ''.join(ch for ch in self.s if not unicodedata.combining(ch))
        self.s = re.sub(r'\s+', ' ', self.s).strip()
        return self.s

    # --- REPLACED/EXTENDED: try targeted queries and DuckDuckGo + english variants ---
    def find_measurement_for_subject(self, subj):
        """
        Try multiple queries (wiki variants + targeted 'Länge' queries + DuckDuckGo snippets)
        Return tuple (measurement_string, source_text_short) or (None, None).
        """
        if not subj:
            return None, None

        self.subj_norm = self.normalize_subject(subj)
        # build candidate queries (DE + EN) and synonyms
        self.candidates = []
        # base
        self.candidates.append(subj)
        self.candidates.append(self.subj_norm)
        # German targeted forms
        self.candidates += [
            f"{subj} Länge",
            f"Länge {subj}",
            f"Durchschnittliche Länge {subj}",
            f"Durchschnittliche Länge {self.subj_norm}",
            f"{subj} Größe",
        ]
        # synonyms for common terms
        if re.search(r'\bauto\b', self.subj_norm, flags=re.IGNORECASE) or re.search(r'\bwagen\b', self.subj_norm, flags=re.IGNORECASE):
            candidates += ["Auto", "Personenkraftwagen", "PKW", "Durchschnittliche Länge Auto", "Durchschnittliche Länge PKW"]
        if re.search(r'\brüssel\b', self.subj_norm, flags=re.IGNORECASE):
            candidates += ["Rüssel", "Elefantenrüssel", "Rüssel Elefant Länge"]

        # english fallbacks
        self.candidates += [
            f"{self.subj_norm} length",
            "car length",
            "average car length",
            "typical car length",
            "elephant trunk length",
            "length of elephant trunk"
        ]

        self.tried = set()
        # try Wikipedia page extracts first (better chance to contain numbers)
        for q in candidates:
            if not q or q in self.tried:
                continue
            self.tried.add(q)
            try:
                self.page_text = fetch_wikipedia_page_text(q)
            except Exception:
                self.page_text = None
            if self.page_text:
                self.measurement = self.extract_measurement_from_text(self.page_text)
                if self.measurement:
                    return self.measurement, f"Wikipedia page: {q}"
            # fallback to summary if page_text didn't exist
            try:
                self.wiki = fetch_wikipedia_summary(q)
            except Exception:
                self.wiki = None
            if self.wiki:
                self.measurement = self.extract_measurement_from_text(self.wiki)
                if self.measurement:
                    return self.measurement, f"Wikipedia summary: {q}"

        # also try broader Wikipedia variants (existing helper)
        try:
            self.wiki_variant = fetch_wikipedia_variants(subj)
        except Exception:
            self.wiki_variant = None
        if self.wiki_variant:
            self.measurement = self.extract_measurement_from_text(self.wiki_variant)
            if self.measurement:
                return self.measurement, "Wikipedia (variant)"

        # lastly try DuckDuckGo search snippets for a few prioritized queries
        for q in [f"{subj} Länge", f"{self.subj_norm} length", "durchschnittliche Länge Auto", "average car length", "elephant trunk length"]:
            try:
                self.sd = self.search_web(q)
            except Exception:
                self.sd = None
            if self.sd and isinstance(self.sd, str):
                measurement = self.extract_measurement_from_text(self.sd)
                if measurement:
                    self.src = "DuckDuckGo: " + q
                    return self.measurement, self.src

        return None, None

    def chatbot_response(self, question):
        print(question)
        try:
            # quick math detection: evaluate simple arithmetic expressions even without "Berechne"
            # matches inputs containing digits and math operators (e.g. "2*4", "12 / (3+1)")
            if re.search(r'\d', question) and re.search(r'[\+\-\*\/\^()]', question):
                try:
                    # use existing evaluate_math_expression when available
                    self.result = self.evaluate_math_expression(question)
                    return self.result, {"tokens": [], "note": "evaluated-as-math"}
                except Exception:
                    # fallback: continue to NLP if evaluation fails
                    pass

            # Gibberlink aktivieren, wenn EXPERIMENTAL-Modus an ist

            # NEW: Definition/Übersetzungsanfragen (Wortbedeutung)
            if re.search(r'\bwas\s+bedeutet\b|\bwas\s+heißt\b|\bbedeutung\s+von\b', question, flags=re.IGNORECASE):
                self.term = extract_subject_from_question(question)
                if self.term:
                    # prefer wiktionary, fallback to wikipedia summary/page
                    self.definition = fetch_wiktionary_definition(self.term)
                    if not self.definition:
                        # try wikipedia page text then summary
                        self.definition = fetch_wikipedia_page_text(self.term) or fetch_wikipedia_summary(self.term)
                    if self.definition:
                        self.first_para = self.definition.split("\n\n")[0].strip()
                        return f"'{self.term}' bedeutet:\n{self.first_para}", {"tokens": [], "note": "dictionary"}
                return "Dazu konnte ich leider keine Definition finden.", {"tokens": [], "note": "no-definition"}

            # REPLACED: improved measurement question handling
            if re.search(r'\bwie\s+(lang|groß|hoch|schwer|alt)\b', question, flags=re.IGNORECASE):
                self.subj = extract_subject_from_question(question)
                # first try direct wiki variants (keeps previous behavior)
                self.wiki = fetch_wikipedia_variants(self.subj) if self.subj else None
                if self.wiki:
                    self.measurement = self.extract_measurement_from_text(self.wiki)
                    if self.measurement:
                        self.subj_display = self.subj or "das Objekt"
                        return f"Der {self.subj_display} ist ungefähr {self.measurement}. (Quelle: Wikipedia)", {"tokens": [], "note": "wikipedia-measurement"}
                # if initial wiki summary didn't contain measurement, do targeted search attempts
                self.measurement, self.source = self.find_measurement_for_subject(self.subj)
                if self.measurement:
                    self.subj_display = self.subj or "das Objekt"
                    return f"Der {self.subj_display} ist ungefähr {self.measurement}. (Quelle: {self.source})", {"tokens": [], "note": "web-measurement"}
                # fallback: if we had a wiki without measurement, return it (better than nothing)
                if self.wiki:
                    self.subj_display = self.subj or "das Objekt"
                    return f"Ich habe zu '{self.subj_display}' folgende Info auf Wikipedia gefunden:\n{self.wiki}", {"tokens": [], "note": "wikipedia-summary"}
                # nothing found -> fall back to general NLP below (or inform about missing data)
                return "Dazu habe ich leider keine eindeutige Längenangabe gefunden.", {"tokens": [], "note": "no-measurement"}

            # Mathematischer Ausdruck
            #if question.startswith("Berechne"):
            #    self.expression = question.split("Berechne")[-1].strip()
            #    return self.evaluate_math_expression(self.expression), None

            # Websuche
            if question.lower().startswith("suche nach"):
                self.query = question.split("suche nach")[-1].strip()
                return self.search_web(self.query), None


            # NLP-Modell

            # --- Exact match fallback before model prediction ---
            # Lowercase and strip for robust matching
            q_norm = question.strip().lower()
            for idx, q in enumerate(self.questions):
                if q.strip().lower() == q_norm:
                    return self.answers[idx], self.preprocess_text(question)

            self.question_tfidf = self.vectorizer.transform([question])
            self.response = self.model.predict(self.question_tfidf)[0]
            self.nlp_info = self.preprocess_text(question)

            # Try to enrich the model response with a deterministic Wikipedia summary
            try:
                self.subj = extract_subject_from_question(question)
                # do not attempt wiki if subject extraction failed (or was math)
                self.wiki = fetch_wikipedia_summary(self.subj) if self.subj else None
                if self.wiki:
                    # append but avoid duplicating if already included
                    if isinstance(self.response, str) and "[Wikipedia]" not in self.response:
                        self.response = (self.response or "") + "\n\n[Wikipedia]: " + self.wiki
            except Exception:
                # degrade gracefully: return model response without augmentation
                pass

            return self.response, self.nlp_info

        except Exception as e:
            return f"Fehler bei der Verarbeitung der Frage: {str(e)}", None

    # Funktion zum Hinzufügen von neuen Fragen und Antworten
    def add_to_training_data(self, question, answer, training_data, file_path='training_data.json'):
        self.new_entry = {"question": question, "answer": answer}
        training_data.append(self.new_entry)
    
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(training_data, file, ensure_ascii=False, indent=4)
    
        # Modelle neu trainieren
        self.questions = [data['question'] for data in training_data]
        X = self.vectorizer.fit_transform(self.questions)
        self.answers = [data['answer'] for data in training_data]
        self.model.fit(X, self.answers)

    # Automatisches Lernen aus Interaktionen
    def learn_from_interaction(self, user_input, expected_response):
        self.add_to_training_data(user_input, expected_response)
    from PyQt5.QtWidgets import QLabel, QVBoxLayout, QDialog
    from PyQt5.QtGui import QPixmap



    # Funktion zur Überprüfung und Verbesserung der Antwort
    def validate_response(self, user_input, response):
        print(f"Chatbot: {response}")
        self.feedback = input("War die Antwort korrekt? (ja/nein): ").strip().lower()
        if self.feedback == "nein":
            self.correct_answer = input("Wie hätte ich antworten sollen? ")
            self.learn_from_interaction(user_input, self.correct_answer)
            return self.correct_answer
        return response
    def load_mysql_datasets(self, username, password, database, AutoTeach=True, training_data=None, key="gamesmcde"):
        db = mysql.connector.connect(
            host="mintai.ddns.net",
            port=3306,
            user=username,
            password=password,
            database=database
        )
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT topic, description, example_title, example_summary, question FROM gamesmc_content")
        rows = cursor.fetchall()
        cursor.close()
        db.close()

        # Struktur für key "gamesmcde"
        structured = {key: []}
        topic_map = {}

        for row in rows:
            topic = row["topic"]
            if topic not in topic_map:
                topic_map[topic] = {
                    "topic": topic,
                    "description": row["description"],
                    "examples": []
                }

            topic_map[topic]["examples"].append({
                "title": row["example_title"],
                "summary": row["example_summary"],
                "questions": [row["question"]] if row["question"] else []
            })

        structured[key] = list(topic_map.values())

        if AutoTeach and training_data is not None:
            training_data = append_data(training_data, structured, key=key)
            logging.info(f"Trainingseinträge erzeugt: {len(training_data)}")
            # --- retrain model with new data ---
            self.prepareInstance(training_data)
            return training_data

        return structured
    def load_crawler_json_dataset(self, json_path, training_data=None, key="gamesmcde", AutoTeach=True):
        """
        Loads a dataset exported by crawler.py as JSON and appends it to training_data.
        """
        from data import load_crawler_json, append_data
        if training_data is None:
            training_data = []
        structured = load_crawler_json(json_path, key=key)
        training_data = append_data(training_data, structured, key=key)
        if AutoTeach:
            self.prepareInstance(training_data)
        return training_data

# Transformer-Architektur
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, model_dim)
        self.transformer = nn.Transformer(model_dim, num_heads, num_layers)
        self.fc = nn.Linear(model_dim, output_dim)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        output = self.transformer(src, tgt)
        output = self.fc(output)
        return output
