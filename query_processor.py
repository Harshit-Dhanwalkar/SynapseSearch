import os

import nltk
from nltk.corpus import wordnet

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from symspellpy import SymSpell


class QueryProcessor:
    def __init__(self):
        # Initialize spell checker
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2)
        dict_path = os.path.join(
            os.path.dirname(__file__), "data", "frequency_dict.txt"
        )
        if os.path.exists(dict_path):
            self.sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)

        # Initialize WordNet
        nltk.download("wordnet", quiet=True)

    # def process(self, query):
    #     query = self.correct_spelling(query)
    #     return self.expand_with_synonyms(query)
    def process(self, query):
        return query  # Skipping spell check and synonym expansion for debugging

    # def correct_spelling(self, query):
    #     suggestions = self.sym_spell.lookup_compound(query, max_edit_distance=2)
    #     return suggestions[0].term if suggestions else query

    # def expand_with_synonyms(self, query):
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
