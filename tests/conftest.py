from pytest import fixture

@fixture(scope='module')
def wilhelmus_path():
    return "tests/resources/Wilhelmus-van-Nassouwe.pdf"

@fixture(scope='module')
def