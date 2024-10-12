import os

from pytest import fixture

@fixture(scope='module')
def wilhelmus_path():
    return "resources/Wilhelmus-van-Nassouwe.pdf"

@fixture(scope='module')
def tika_url():
    return os.environ.get("TIKA_URL", "http://localhost:9998/")
