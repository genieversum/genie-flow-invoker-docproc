import json
import os

import requests
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
def sweden_text():
    return """
Sweden, formally the Kingdom of Sweden, is a Nordic country located on the Scandinavian
Peninsula in Northern Europe. It borders Norway to the west and north, and Finland to the 
east. At 450,295 square kilometres (173,860 sq mi), Sweden is the largest Nordic country 
and the fifth-largest country in Europe. The capital and largest city is Stockholm.

Sweden has a population of 10.6 million, and a low population density of 25.5 inhabitants 
per square kilometre (66/sq mi); 88% of Swedes reside in urban areas. They are mostly in the 
central and southern half of the country. Sweden's urban areas together cover 1.5% of its land area. 
Sweden has a diverse climate owing to the length of the country, which ranges from 55°N to 69°N. 
        """

@fixture(scope='module')
def netherlands_text():
    return """
The Netherlands, informally Holland, is a country in Northwestern Europe, with overseas 
territories in the Caribbean. It is the largest of the four constituent countries of the 
Kingdom of the Netherlands. The Netherlands consists of twelve provinces; it borders Germany 
to the east and Belgium to the south, with a North Sea coastline to the north and west. 
It shares maritime borders with the United Kingdom, Germany, and Belgium. The official 
language is Dutch, with West Frisian as a secondary official language in the province of 
Friesland. Dutch, English, and Papiamento are official in the Caribbean territories.

Netherlands literally means "lower countries" in reference to its low elevation and flat 
topography, with 26% below sea level. Most of the areas below sea level, known as polders, 
are the result of land reclamation that began in the 14th century. In the Republican period, 
which began in 1588, the Netherlands entered a unique era of political, economic, and cultural 
greatness, ranked among the most powerful and influential in Europe and the world; this period 
is known as the Dutch Golden Age. During this time, its trading companies, the Dutch East 
India Company and the Dutch West India Company, established colonies and trading posts all 
over the world. 
    """

@fixture(scope='module')
def tika_url():
    return os.environ.get("TIKA_URL", "http://localhost:9998/")


class MockRequestResponse:
    def __init__(
            self,
            status_code=200,
            text="",
            json_data=None,
    ):
        self.status_code = status_code
        self._text = text
        self._json_data = json_data

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.HTTPError(self.status_code)

    @property
    def text(self):
        if self._text:
            return self._text
        if self._json_data:
            return json.dumps(self._json_data)
        raise ValueError("No value provided that can be returned as text")

    def json(self):
        return self._json_data
