from pytest import fixture

@fixture(scope='module')
def wilhelmus_path():
    return "resources/Wilhelmus-van-Nassouwe.pdf"

@fixture(scope='module')
def tika_url():
    return "http://apache-tika:9998/"
