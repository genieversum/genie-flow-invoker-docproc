import re
from collections import OrderedDict
from functools import partial
from typing import Optional

from nltk import PunktSentenceTokenizer, TreebankWordTokenizer, TreebankWordDetokenizer


def remove_numbers(text):
    text = re.sub("[0-9]{5,}", "#####", text)
    text = re.sub("[0-9]{4}", "####", text)
    text = re.sub("[0-9]{3}", "###", text)
    text = re.sub("[0-9]{2}", "##", text)
    return text


def replace_special_terms(special_term_replacements: dict[str, str], text: str):
    for term, placeholder in special_term_replacements.items():
        text = text.replace(term, placeholder)
    return text


def tokenize_detokenize(text):
    sent_tokenizer = PunktSentenceTokenizer()
    sentences = sent_tokenizer.tokenize(text)

    word_tokenizer = TreebankWordTokenizer()
    sentences_words = [word_tokenizer.tokenize(sentence) for sentence in sentences]

    detokenizer = TreebankWordDetokenizer()
    return "\n".join(detokenizer.detokenize(words) for words in sentences_words)


class TextCleaner:

    def __init__(
        self,
        clean_multiple_newlines: bool = True,
        clean_multiple_spaces: bool = True,
        clean_tabs: bool = True,
        clean_numbers: bool = True,
        special_term_replacements: Optional[dict] = None,
        tokenize_detokenize: bool = True,
    ):
        self.clean_multiple_newlines = clean_multiple_newlines
        self.clean_multiple_spaces = clean_multiple_spaces
        self.clean_tabs = clean_tabs
        self.clean_numbers = clean_numbers
        self.special_term_replacements = (
            special_term_replacements if special_term_replacements is not None else {}
        )
        self.tokenize_detokenize = tokenize_detokenize

    def clean(self, text: str) -> str:
        cleaners = OrderedDict(
            clean_multiple_newlines=lambda t: re.sub(r"\n{2,}", "\n", t),
            clean_multiple_spaces=lambda t: re.sub(r"\s{2,}", " ", t),
            clean_tabs=lambda t: re.sub(r"\t+", " ", t),
            clean_numbers=remove_numbers,
            special_term_replacements=partial(
                replace_special_terms, self.special_term_replacements
            ),
            tokenize_detokenize=tokenize_detokenize,
        )
        for switch, func in cleaners.items():
            if getattr(self, switch):
                text = func(text)

        return text


if __name__ == "__main__":
    import nltk

    nltk.download("punkt_tab")
