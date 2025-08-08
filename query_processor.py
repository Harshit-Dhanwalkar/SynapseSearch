import os
import re

import nltk
from nltk.corpus import wordnet
from symspellpy import SymSpell

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


class QueryProcessor:
    def __init__(self, stopwords_file="stopwords.txt"):
        # Initialize spell checker
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2)
        dict_path = os.path.join(
            os.path.dirname(__file__), "data", "frequency_dict.txt"
        )
        if os.path.exists(dict_path):
            self.sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)

        # Load stopwords
        self.stopwords = self._load_stopwords(stopwords_file)

    def process(self, query):
        #     query = self.correct_spelling(query)
        #     return self.expand_with_synonyms(query)
        return query  # Skipping spell check and synonym expansion for debugging

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
        #     suggestions = self.sym_spell.lookup_compound(query, max_edit_distance=2)
        #     return suggestions[0].term if suggestions else query
        pass

    def get_synonyms(self, word):
        # synonyms = set()
        # for syn in wordnet.synsets(word):
        #     for lemma in syn.lemmas():
        #         synonyms.add(lemma.name())
        # return synonyms
        pass

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
        pass

    def preprocess_query(self, query, expand_synonyms=False):
        # Clean and tokenize the query
        query = re.sub(r"[^\w\s]", "", query.lower())
        tokens = query.split()

        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stopwords]

        if expand_synonyms:
            expanded_tokens = set(tokens)
            for token in tokens:
                expanded_tokens.update(self.get_synonyms(token))
            return list(expanded_tokens)

        return tokens
