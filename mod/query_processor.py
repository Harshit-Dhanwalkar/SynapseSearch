import os
import re

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nltk_data_dir = os.path.join(project_root, "data", "nltk_data")
os.environ["NLTK_DATA"] = nltk_data_dir
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

import nltk
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from symspellpy import SymSpell

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


class QueryProcessor:
    def __init__(self, stopwords_file="stopwords.txt"):
        # Spell checker
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2)
        dict_path = os.path.join(
            os.path.dirname(__file__), "data", "frequency_dict.txt"
        )
        if os.path.exists(dict_path):
            try:
                self.sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
            except Exception:
                # If the dict is malformed or unreadable, don't crash - continue without it
                pass

        self.stopwords = self._load_stopwords(stopwords_file)
        self.stemmer = PorterStemmer()

    def process(self, query, expand_synonyms=True):
        if not query:
            return ""

        try:
            query = self.correct_spelling(query)
        except Exception:
            # if SymSpell fails for any reason, continue with original
            pass
        tokens = self.preprocess_query(query)
        if not tokens:
            return ""
        tokens = [self.stemmer.stem(t) for t in tokens]

        if expand_synonyms and len(tokens) > 1:
            expanded_tokens = set(tokens)
            for t in tokens:
                expanded_tokens.extend([t, t])
                # add synonyms (get_synonyms returns a list)
                try:
                    syns = self.get_synonyms(t)
                except Exception:
                    syns = []
                # only append synonyms that are not equal to the token already present
                for s in syns:
                    if s != t:
                        expanded_tokens.append(s)
            tokens = expanded_tokens

        return " ".join(tokens)

    def _load_stopwords(self, filename):
        stopwords = set()
        try:
            filepath = os.path.join(os.path.dirname(__file__), filename)
            with open(filepath, "r", encoding="utf-8") as f:
                stopwords = {word.strip().lower() for word in f}
        except FileNotFoundError:
            print(f"Warning: Stopwords file '{filename}' not found.")
        return stopwords

    def correct_spelling(self, query):
        try:
            suggestions = self.sym_spell.lookup_compound(query, max_edit_distance=2)
            return suggestions[0].term if suggestions else query
        except Exception:
            return query

    def get_synonyms(self, word):
        result = []
        if not word or len(word) <= 2 or not word.isalpha():
            return result

        seen = set()
        for syn in wordnet.synsets(word, pos=wordnet.NOUN):
            for lemma in syn.lemmas():
                if lemma.count() < 2:
                    continue  # skip rare/obscure lemma senses
                s = lemma.name().replace("_", " ").lower()
                if not s.isalpha() or len(s) <= 2:
                    continue
                stemmed = self.stemmer.stem(s)
                if stemmed == word:
                    continue
                if stemmed in seen:
                    continue
                seen.add(stemmed)
                result.append(stemmed)
                if len(result) >= max_syn:
                    return result
        return result

    def expand_with_synonyms(self, query):
        #     terms = set(query.split())
        #     for term in query.split():
        #         # Only expand terms longer than 2 characters
        #         if len(term) > 2:
        #             for syn in wordnet.synsets(term):
        #                 # Only include common synonyms (exclude scientific/obscure terms)
        #                 for lemma in syn.lemmas():
        #                     if (
        #                         lemma.count() > 1
        #                     ):  # Only include terms that appear frequently
        #                         synonym = lemma.name().replace("_", " ")
        #                         # Filter out overly long or complex terms
        #                         if len(synonym.split()) <= 2 and len(synonym) < 20:
        #                             terms.add(synonym)
        #     return " ".join(terms)
        query = re.sub(r"[^\w\s]", "", query.lower())
        tokens = [t for t in query.split() if t and t not in self.stopwords]
        expanded = []
        for t in tokens:
            expanded.append(t)
            expanded.extend(self.get_synonyms(t))
        return expanded

    def preprocess_query(self, query, expand_synonyms=False):
        # Clean and tokenize the query
        query = re.sub(r"[^\w\s]", "", query.lower())
        # tokens = query.split()

        # Remove stopwords
        tokens = [t for t in query.split() if t and t not in self.stopwords]
        if expand_synonyms:
            expanded = []
            for t in tokens:
                expanded.append(t)
                expanded.extend(self.get_synonyms(t))
            return expanded
        return tokens
