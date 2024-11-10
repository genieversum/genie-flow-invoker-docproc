import re

from genie_flow_invoker.invoker.docproc.clean import DocumentCleanInvoker, TextCleaner
from genie_flow_invoker.invoker.docproc.model import ChunkedDocument


def test_no_cleaning(hamlet_content):
    cleaner = TextCleaner(
        clean_multiple_newlines=False,
        clean_multiple_spaces=False,
        clean_tabs=False,
        clean_numbers=False,
        special_term_replacements={},
        tokenize_detokenize=False,
    )

    cleaned_hamlet = cleaner.clean(hamlet_content)
    assert cleaned_hamlet == hamlet_content


def test_clean_multiple_newlines(hamlet_content):
    cleaner = TextCleaner(
        clean_multiple_newlines=True,
        clean_multiple_spaces=False,
        clean_tabs=False,
        clean_numbers=False,
        special_term_replacements={},
        tokenize_detokenize=False,
    )

    _, original_nr = re.subn("\n{2,}", "\n", hamlet_content)
    cleaned_hamlet = cleaner.clean(hamlet_content)
    _, new_nr = re.subn("\n{2,}", "\n", cleaned_hamlet)

    assert original_nr > new_nr
    assert new_nr == 0


def test_clean_multiple_spaces(hamlet_content):
    cleaner = TextCleaner(
        clean_multiple_newlines=False,
        clean_multiple_spaces=True,
        clean_tabs=False,
        clean_numbers=False,
        special_term_replacements={},
        tokenize_detokenize=False,
    )

    _, original_nr = re.subn("\s{2,}", " ", hamlet_content)
    cleaned_hamlet = cleaner.clean(hamlet_content)
    _, new_nr = re.subn("\s{2,}", " ", cleaned_hamlet)

    assert original_nr > new_nr
    assert new_nr == 0


def test_clean_tabs(hamlet_content):
    cleaner = TextCleaner(
        clean_multiple_newlines=False,
        clean_multiple_spaces=False,
        clean_tabs=True,
        clean_numbers=False,
        special_term_replacements={},
        tokenize_detokenize=False,
    )

    _, original_nr = re.subn("\t+", " ", hamlet_content)
    cleaned_hamlet = cleaner.clean(hamlet_content)
    _, new_nr = re.subn("\t+", " ", cleaned_hamlet)

    assert "\t" not in cleaned_hamlet
    assert original_nr > new_nr
    assert new_nr == 0


def test_clean_numbers(hamlet_content):
    cleaner = TextCleaner(
        clean_multiple_newlines=False,
        clean_multiple_spaces=False,
        clean_tabs=False,
        clean_numbers=True,
        special_term_replacements={},
        tokenize_detokenize=False,
    )

    cleaned_hamlet = cleaner.clean(hamlet_content)

    assert "2015" in hamlet_content
    assert "2015" not in cleaned_hamlet
    assert "####" in cleaned_hamlet
    assert "Hamlet" in cleaned_hamlet


def test_clean_special_term_replacements(hamlet_content):
    cleaner = TextCleaner(
        clean_multiple_newlines=False,
        clean_multiple_spaces=False,
        clean_tabs=False,
        clean_numbers=False,
        special_term_replacements={
            "HAMLET": "%%Prince of Denmark%%",
        },
        tokenize_detokenize=False,
    )

    cleaned_hamlet = cleaner.clean(hamlet_content)

    assert "HAMLET" in hamlet_content
    assert "HAMLET" not in cleaned_hamlet
    assert "%%Prince of Denmark%%" in cleaned_hamlet
    assert "Ophelia" in cleaned_hamlet


def test_clean_tokenize_detokenize(hamlet_content):
    cleaner = TextCleaner(
        clean_multiple_newlines=False,
        clean_multiple_spaces=False,
        clean_tabs=False,
        clean_numbers=False,
        special_term_replacements={},
        tokenize_detokenize=True,
    )

    cleaned_hamlet = cleaner.clean(hamlet_content)

    assert """To be or not to be--that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And, by opposing, end them. To die, to sleep--""" in hamlet_content
    assert (
        "To be or not to be--that is the question: Whether'tis nobler in the mind to "
        "suffer The slings and arrows of outrageous fortune, Or to take arms against a "
        "sea of troubles And, by opposing, end them." in cleaned_hamlet
    )


def test_invocation(hamlet_document):
    clean_invoker = DocumentCleanInvoker(
        clean_multiple_newlines=True,
        clean_multiple_spaces=False,
        clean_tabs=False,
        clean_numbers=False,
        special_term_replacements={},
        tokenize_detokenize=False,
    )

    cleaned_document_json = clean_invoker.invoke(hamlet_document.model_dump_json())
    cleaned_document = ChunkedDocument.model_validate_json(cleaned_document_json)

    assert cleaned_document.filename == hamlet_document.filename
    assert len(cleaned_document.chunks) == len(hamlet_document.chunks)

    for i, cleaned_chunk in enumerate(cleaned_document.chunks):
        _, original_nr = re.subn(
            "\n{2,}",
            "\n"
            , hamlet_document.chunks[i].content
        )
        _, new_nr = re.subn(
            "\n{2,}",
            "\n",
            cleaned_chunk.content
        )

        assert original_nr >= new_nr
        assert new_nr == 0
