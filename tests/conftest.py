import os

from pytest import fixture

from invoker.docproc.clean import DocumentCleanInvoker


@fixture(scope='module')
def wilhelmus_path():
    return "resources/Wilhelmus-van-Nassouwe.pdf"

@fixture(scope='module')
def hamlet_path():
    return "resources/hamlet_TXT_FolgerShakespeare.txt"

@fixture(scope='module')
def hamlet_content(hamlet_path):
    with open(hamlet_path, "r") as f:
        return f.read()

@fixture(scope='module')
def hamlet_content_cleaned(hamlet_content):
    cleaner = DocumentCleanInvoker(
        clean_multiple_newlines=True,
        clean_multiple_spaces=True,
        clean_tabs=True,
        clean_numbers=True,
        special_term_replacements={},
        tokenize_detokenize=True,
    )
    return cleaner.invoke(hamlet_content)

@fixture(scope='module')
def tika_url():
    return os.environ.get("TIKA_URL", "http://localhost:9998/")
