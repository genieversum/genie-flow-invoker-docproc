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


@fixture(scope="session")
def pa_text():
    return """
We’ve been enabling leaders to unlock ingenuity to address their toughest challenges for 
nearly 80 years.

England. 1943. A time of national crisis.

The men who had been manufacturing for the war effort had been called to the front lines, 
leaving the factories empty. Our experts approached the government with an offer: to help 
inspire and encourage women into the workforce, training them to succeed, in the most advanced 
technological facilities of the day.

This was the age of Taylorism and Fordism, the era of scientific management that treated 
people as machines. Workers’ needs, motivations and feelings had been disregarded.

Our approach was completely different. Innovative thinking. Breakthrough technology. With a 
human-centred approach that appreciated diversity, valued individuals, and inspired teams to 
deliver previously unachievable results. And so, PA (short for Personnel Administration) 
was born.

Since then, the innovation has continued. The first self-service parking system, the world’s 
first private digital telephone exchange, the original brushless servo motor, and the handheld 
tonometer. The first recordable compact discs, the disposable pregnancy test, the first 
breath-actuated metered dose inhaler. 3G and 4G critical national infrastructure. A 
hyperloop to revolutionise travel. Seaweed replacements for plastics. The world’s first 
tea-sheet. Cobots in care. A smart sock that keeps babies safe.

Whatever changes in PA, some things will always stay the same. Our care for our clients and 
the customers and citizens they serve. And our dedication for the positive impact we can make 
on the world.
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


@fixture(scope="session")
def docker_compose_file():
    return "resources/docker-compose.yaml"

@fixture(scope="session")
def docker_setup(docker_setup):
    if "TIKA_URL" in os.environ or "T2V_URL" in os.environ:
        return False
    return docker_setup


@fixture(scope="session")
def docker_cleanup(docker_cleanup):
    if "TIKA_URL" in os.environ or "T2V_URL" in os.environ:
        return False
    return docker_cleanup


@fixture(scope="session")
def tika_url(docker_services):
    if "TIKA_URL" in os.environ:
        return os.environ.get("TIKA_URL")

    tika_port = docker_services.port_for("tika", 9998)
    url = f"http://localhost:{tika_port}"

    def tika_is_responsive():
        try:
            response = requests.get(url)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    docker_services.wait_until_responsive(
        timeout=30.0,
        pause=0.1,
        check=tika_is_responsive
    )
    return url


@fixture(scope="session")
def t2v_url(docker_services):
    if "T2V_URL" in os.environ:
        return os.environ.get("T2V_URL")

    t2v_port = docker_services.port_for("t2v-transformers", 8080)
    url = f"http://localhost:{t2v_port}"

    def t2v_is_responsive():
        try:
            response = requests.get(f"{url}/meta")
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    docker_services.wait_until_responsive(
        timeout=30.0,
        pause=0.1,
        check=t2v_is_responsive
    )
    return url


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
