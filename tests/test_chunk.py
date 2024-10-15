import nltk
from nltk import TreebankWordDetokenizer, TreebankWordTokenizer

from invoker.docproc.chunk import LexicalDensitySplitter
from invoker.docproc.model import ParsedDocument


def test_lexical_chunk_empty():
    chunker = LexicalDensitySplitter(5, 15, 2, 0.8)

    empty_document = ParsedDocument(
        filename="some empty file.txt",
        document_text="",
        document_metadata=dict(),
    )

    chunks = chunker.split(empty_document)

    assert len(chunks) == 0


def test_lexical_chunk_no_meaning():
    chunker = LexicalDensitySplitter(5, 15, 2, 0.8)

    document = ParsedDocument(
        filename="some file.txt",
        document_text="""
this is just a bunch of nonsense.
        """,
        document_metadata=dict(),
    )

    chunks = chunker.split(document)

    assert len(chunks) == 0


def test_lexical_chunk_with_meaning():
    chunker = LexicalDensitySplitter(15, 40, 5, 0.6)

    document = ParsedDocument(
        filename="some file.txt",
        document_text="""
Sweden, formally the Kingdom of Sweden, is a Nordic country located on the Scandinavian
Peninsula in Northern Europe. It borders Norway to the west and north, and Finland to the 
east. At 450,295 square kilometres (173,860 sq mi), Sweden is the largest Nordic country 
and the fifth-largest country in Europe. The capital and largest city is Stockholm.

Sweden has a population of 10.6 million, and a low population density of 25.5 inhabitants 
per square kilometre (66/sq mi); 88% of Swedes reside in urban areas. They are mostly in the 
central and southern half of the country. Sweden's urban areas together cover 1.5% of its land area. 
Sweden has a diverse climate owing to the length of the country, which ranges from 55°N to 69°N. 
        """,
        document_metadata=dict(),
    )

    chunks = chunker.split(document)

    tokenizer = TreebankWordTokenizer()
    lexical_tags = {
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

    for c in chunks:
        words = tokenizer.tokenize(c.document_chunk)
        assert 15 <= len(words) <= 40

        pos: list[tuple[str, str]] = nltk.tag.pos_tag(words)
        lexical_words = sum(1 for p in pos if p[1] in lexical_tags)
        assert lexical_words / len(words) >= 0.6