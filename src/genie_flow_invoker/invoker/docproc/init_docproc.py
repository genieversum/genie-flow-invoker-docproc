import os

import nltk


def init():
    nltk_download_dir = os.getenv('NLTK_DATA', None)
    nltk.download('punkt_tab', download_dir=nltk_download_dir)
    nltk.download('stopwords', download_dir=nltk_download_dir)
