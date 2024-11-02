from typing import Optional

from genie_flow_invoker.genie import GenieInvoker

from invoker.docproc.clean.cleaner import TextCleaner
from invoker.docproc.codec import PydanticInputDecoder, PydanticOutputEncoder
from invoker.docproc.model import ChunkedDocument


SPECIAL_TERMS = {
    "i.e.": "%%IE%%",
    "e.g.": "%%EG%%",
    "etc.": "%%ETC%%",
    ".com": "%%DOTCOM%%",
    "www.": "%%WWW%%",
}


class DocumentCleanInvoker(
    GenieInvoker,
    PydanticInputDecoder[ChunkedDocument],
    PydanticOutputEncoder[ChunkedDocument],
):
    """
    This invoker takes a document an applies some clean-up methods to correct for some common
    mistakes in documents or artifacts that have been introduced by parsing a document. The
    clean-up is applied to each and every chunk contained in the input chunked document.

    This invoker can do the following clean-ups:

    - remove multiple newlines: a sequence of two or more newlines is replaced by a single
    - remove multiple spaces: a sequence of two or more whitespace characters is replaced
      by a single space
    - clean up tabs: remove a sequence of one or more tabs and replace by a single space
    - clean numbers: replace numbers that have n digits by n # characters, for n being
      larger than 1 and below 6. For n == 1, leave alone. For n >= 6, replace by five #
      characters.
    - special term replacement: replace a special term with a placeholder.
    - tokenize and de-tokenize: tokenize a document into sentences, then tokens, and then
      reconstruct the document using newlines between every reconstructed sentence.

    When using the tokenize / de-tokenize cleanup, it is expected that the user of this
    Invoker has downloaded the `punkt_tab` in the default place. One can run this module
    to use the NLTK download utility to make that download.
    """

    def __init__(
        self,
        clean_multiple_newlines: bool = True,
        clean_multiple_spaces: bool = True,
        clean_tabs: bool = True,
        clean_numbers: bool = True,
        special_term_replacements: Optional[dict] = None,
        tokenize_detokenize: bool = True,
    ):
        self._cleaner = TextCleaner(
            clean_multiple_newlines,
            clean_multiple_spaces,
            clean_tabs,
            clean_numbers,
            special_term_replacements,
            tokenize_detokenize,
        )

    @classmethod
    def from_config(cls, config: dict):
        """
        Creates the DocumentCleanInvoker object from config. Config can contain any of the
        following keys:

        - clean_multiple_newlines (bool): If True, the function will remove consecutive newlines
        - clean_multiple_spaces (bool): If True, the function will remove consecutive spaces
        - clean_tabs (bool): If True, the function will remove tabs and replace them with a space
        - clean_numbers (bool): If True, the function will remove numbers larger than 9 and replace
          them with ##, ###, #### or ##### for any number with 5 or more digits
        - clean_special_terms (bool | dict): If True, the function will remove the standard
          special terms, if provided with a dict, this is expected to be term/placeholder
        - tokenize_detokenize (bool): If True, the function will tokenize the document
          into sentences, then each sentence is tokenized into a list of tokens.
          Then the document is reconstructed by de-tokenizing the sentences and all
          sentences are joined with a newline in between.

        Standard special terms replacements are:
        - "i.e.": "%%IE%%"
        - "e.g.": "%%EG%%"
        - "etc.": "%%ETC%%"
        - ".com": "%%DOTCOM%%"
        - "www.": "%%WWW%%"

        :param config: the config as specified in the `meta.yaml` file
        :return: a cleaned document
        """
        clean_multiple_newlines = config.get("clean_multiple_newlines", True)
        clean_multiple_spaces = config.get("clean_multiple_spaces", True)
        clean_tabs = config.get("clean_tabs", True)
        clean_numbers = config.get("clean_numbers", True)

        clean_special_terms: bool | dict = config.get("clean_special_terms", True)
        if isinstance(clean_special_terms, dict):
            special_term_replacements = clean_special_terms
        elif isinstance(clean_special_terms, bool) and clean_special_terms:
            special_term_replacements = SPECIAL_TERMS
        else:
            special_term_replacements = dict()

        tokenize_detokenize = config.get("tokenize_detokenize", True)

        return cls(
            clean_multiple_newlines=clean_multiple_newlines,
            clean_multiple_spaces=clean_multiple_spaces,
            clean_tabs=clean_tabs,
            clean_numbers=clean_numbers,
            special_term_replacements=special_term_replacements,
            tokenize_detokenize=tokenize_detokenize,
        )

    def invoke(self, content: str) -> str:
        document = self._decode_input(content)

        for chunk in document.chunks:
            chunk.content = self._cleaner.clean(chunk.content)

        return self._encode_output(document)
