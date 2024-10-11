import re
from typing import Optional, Literal

from bidict import bidict

from genie import GenieInvoker
from invoker.docproc.codec import PydanticInputDecoder, PydanticOutputEncoder
from invoker.docproc.model import ParsedDocument


speller_language_type = Literal["uk", "us"]

class EnglishSpeller:
    def __init__(self):
        self._UK_US_WORDLIST: Optional[bidict[str, str]] = None

    def _ensure_wordlist(self):
        if self._UK_US_WORDLIST is not None:
            return

        self._UK_US_WORDLIST = bidict({})
        with open("resources/words.txt", "r", encoding="utf-8") as f:
            for line in f:
                uk, us = line.split("\t")
                self._UK_US_WORDLIST[uk] = us

    def to_uk(self, us_word: str) -> str:
        self._ensure_wordlist()
        try:
            return self._UK_US_WORDLIST.inverse[us_word]
        except KeyError:
            return us_word

    def to_us(self, uk_word: str) -> str:
        try:
            return self._UK_US_WORDLIST[uk_word]
        except KeyError:
            return uk_word

    def to(self, word: str, target: speller_language_type) -> str:
        if target == "uk":
            return self.to_uk(word)
        if target == "us":
            return self.to_us(word)
        return word


_ENGLISH_SPELLER = EnglishSpeller()


def remove_multiple_newlines(text):
    return re.sub(r"\n+", "\n", text)


def remove_leading_and_trailing_whitespace(text):
    return text.strip()


def remove_multiple_spaces(text):
    return re.sub(r"\s+", " ", text)


def remove_numbers(text):
    text = re.sub('[0-9]{5,}', '#####', text)
    text = re.sub('[0-9]{4}', '####', text)
    text = re.sub('[0-9]{3}', '###', text)
    text = re.sub('[0-9]{2}', '##', text)
    return text


def clean_spelling(text: str, target: Literal["uk", "us"]):
    words = [
        _ENGLISH_SPELLER.to(word, target)
        for word in text.split(" ")
    ]
    return " ".join(words)


class DocumentCleanInvoker(
    GenieInvoker,
    PydanticInputDecoder[ParsedDocument],
    PydanticOutputEncoder[ParsedDocument],
):

    def __init__(
            self,
            clean_multiple_newlines: bool = True,
            clean_leading_and_trailing_whitespace: bool = True,
            clean_multiple_spaces: bool = True,
            clean_numbers: bool = True,
            target_spelling: speller_language_type = "us",
    ):
        self.clean_multiple_newlines = clean_multiple_newlines
        self.clean_leading_and_trailing_whitespace = clean_leading_and_trailing_whitespace
        self.clean_multiple_spaces = clean_multiple_spaces
        self.clean_numbers = clean_numbers
        self.target_spelling = target_spelling

    @classmethod
    def from_config(cls, config: dict):
        clean_multiple_newlines = config.get("clean_multiple_newlines", True)
        clean_leading_and_trailing_whitespace = config.get(
            "clean_leading_and_trailing_whitespace",
            True,
        )
        clean_multiple_spaces = config.get("clean_multiple_spaces", True)
        clean_numbers = config.get("clean_numbers", True)
        target_spelling: speller_language_type = (
            "us" if config.get("target_spelling", "us") == "us" else "uk"
        )

        return cls(
            clean_multiple_newlines=clean_multiple_newlines,
            clean_leading_and_trailing_whitespace=clean_leading_and_trailing_whitespace,
            clean_multiple_spaces=clean_multiple_spaces,
            clean_numbers=clean_numbers,
            target_spelling=target_spelling,
        )

    def invoke(self, content: str) -> str:
        if self.clean_multiple_newlines:
            content = remove_multiple_newlines(content)
        if self.clean_leading_and_trailing_whitespace:
            content = remove_leading_and_trailing_whitespace(content)
        if self.clean_multiple_spaces:
            content = remove_multiple_spaces(content)
        if self.clean_numbers:
            content = remove_numbers(content)
        content = clean_spelling(content, self.target_spelling)
        return content
