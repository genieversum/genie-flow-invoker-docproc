from typing import NamedTuple, Literal

import nltk
from black import Optional
from nltk import RegexpTokenizer, WordPunctTokenizer, TreebankWordTokenizer

from invoker.docproc.model import ParsedDocument, DocumentChunk


WordSPanTagLex = NamedTuple(
    'WordSPanTagLex',
    [
        ('word', str),
        ("span", tuple[int, int]),
        ('tag', str),
        ('lexical', bool),
    ]
)


def calculate_lexical_density(word_span_tag: list[WordSPanTagLex]) -> float:
    """
    Calculate the lexical density of a given list of word-span-tags. The lexical
    density is calculated as the count of all word-span-tag of lexicographical words
    as a fraction of the total number of words. A lexicographical word is defined
    as not being a stopword and having a POS tag of meaning (see self.lexicographical_tags).

    :param word_span_tag: a list of WordSpanTag combinations
    :return: the lexical density (or zero if the given list is empty)
    """
    if len(word_span_tag) == 0:
        return 0.0

    lex_count = sum(1 for wst in word_span_tag if wst.lexical)
    return lex_count / len(word_span_tag)


class LexicalDensitySplitter:

    def __init__(
            self,
            min_words: int,
            max_words: int,
            overlap: int,
            target_density: float,
            strategy: Literal["shortest", "best", "longest"] = "best",
    ):
        """
        A LexicalDensitySplitter splits a document into chunks according to the given
        target density. Density is defined as the fraction of lexical words that should
        be part of the chunk as a minimum.

        Chunks will have a minimum of `min_words` and a maximum of `max_words`.

        The strategy determines if
        - the shortest chunk is found with a lexical density above `target_density`
        - the best chunk is found with a lexical density above `target_density`
        - the longest chunk is found with a lexical density above `target_density`

        :param min_words: minimal number of words that should be part of the chunk
        :param max_words: maximal number of words that should be part of the chunk
        :param overlap: the stride to take for each chunk
        :param target_density: the minimal density fraction to reach
        :param strategy: the strategy to use (shortest, best or longest)
        """
        self.min_words = min_words
        self.max_words = max_words
        self.overlap = overlap
        self.target_density = target_density
        self.strategy = strategy
        self.stopwords_set = set(word.lower() for word in nltk.corpus.stopwords.words("english"))
        self.lexicographical_tags = {
            "NN",
            "NNS",
            "NNP",
            "NNPS",
            "JJ",
            "JJR",
            "JJS",
            "VB",
            "VBD",
            "VBG",
            "VBN",
            "VBP",
            "VBZ",
            "RB",
            "RBR",
            "RBS",
        }
        self.tokenizer = TreebankWordTokenizer()

    def split(self, document: ParsedDocument) -> list[DocumentChunk]:
        words = self.tokenizer.tokenize(document.document_text)
        word_spans = self.tokenizer.span_tokenize(document.document_text)
        pos_tags = [t[1] for t in nltk.tag.pos_tag(words)]
        lexical_word = [
            w.lower() not in self.stopwords_set and pos_tags[i] in self.lexicographical_tags
            for i, w in enumerate(words)
        ]

        word_span_tag: list[WordSPanTagLex] = [
            WordSPanTagLex(*t)
            for t in zip(words, word_spans, pos_tags, lexical_word)
        ]

        start = 0
        chunks = []
        while start < len(word_span_tag):
            end = start + self.min_words

            chunk_to_add: Optional[list[WordSPanTagLex]] = None
            best_density: float = 0.0
            while end - start < self.max_words and end < len(word_span_tag):
                current_chunk = word_span_tag[start:end]
                density = calculate_lexical_density(current_chunk)

                if density >= self.target_density:
                    if self.strategy == "shortest":
                        chunk_to_add = current_chunk
                        break
                    if self.strategy == "best" and density > best_density:
                        chunk_to_add = current_chunk
                        best_density = density
                    else:
                        chunk_to_add = current_chunk

                end += 1

            if chunk_to_add is not None:
                chunks.append(chunk_to_add)

            start += self.overlap

        return [
            DocumentChunk(
                filename=document.filename,
                document_chunk=document.document_text[wst[0].span[0]:wst[-1].span[1]],
                original_span=(wst[0].span[0], wst[-1].span[1])
            )
            for wst in chunks
        ]
