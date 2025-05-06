from genie_flow_invoker.doc_proc import DocumentChunk
from pytest import fixture

from genie_flow_invoker.invoker.docproc.chunk.transcript import TranscriptSplitter


@fixture
def teams_vtt():
    with open('resources/TEAMS-SurveyResponse-Genie.vtt', 'r') as f:
        return f.read()


@fixture
def zoom_vtt():
    with open("resources/ZOOM-video1009782420.vtt", "r") as f:
        return f.read()


def test_read_teams_webvtt(teams_vtt):
    splitter = TranscriptSplitter()

    parent = DocumentChunk(
        content=teams_vtt,
        parent_id=None,
        original_span=(0, len(teams_vtt)),
        hierarchy_level=0,
    )
    children = splitter.split(parent)

    assert len(children) == 145
    for child in children:
        assert child.hierarchy_level == 1
    assert children[0].content.startswith("Maybe my English is a bit Dutch but.")
    assert children[-1].content.startswith("But.")

def test_read_zoom_webvtt(zoom_vtt):
    splitter = TranscriptSplitter()

    parent = DocumentChunk(
        content=zoom_vtt,
        parent_id=None,
        original_span=(0, len(zoom_vtt)),
        hierarchy_level=0,
    )
    children = splitter.split(parent)

    assert len(children) == 20
    for child in children:
        assert child.custom_properties["party_name"] == "UNKNOWN"
    assert children[0].content.startswith("Workers link.\nNow you can play the streets.")
    assert children[-1].content.startswith("I like the icons.\nRed is a very good color")


def test_read_non_webvtt(sweden_text):
    splitter = TranscriptSplitter()

    parent = DocumentChunk(
        content=sweden_text,
        parent_id=None,
        original_span=(0, len(sweden_text)),
        hierarchy_level=0,
    )
    children = splitter.split(parent)

    assert len(children) == 0
