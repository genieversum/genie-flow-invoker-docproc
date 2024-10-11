from tika import parser

from genie_flow_invoker.genie import GenieInvoker
from genie_flow_invoker.invoker.docproc.codec import PydanticInputDecoder, PydanticOutputEncoder
from genie_flow_invoker.utils import get_config_value

from genie_flow_invoker.invoker.docproc.model import DocumentInput, ParsedDocument


class DocumentParseInvoker(
    GenieInvoker,
    PydanticInputDecoder[DocumentInput],
    PydanticOutputEncoder[ParsedDocument],
):
    """
    Parse a binary document into text and their metadata.
    """

    def __init__(
            self,
            tika_service_url: str,
    ):
        self._tika_service_url = tika_service_url


    @classmethod
    def from_config(cls, config: dict):
        """
        The `meta.yaml` for the parser should contain the following properties:
        - tika_service_url: the url of the tika service

        :param config: the dictionary of the configuration
        :return: a new instantiated invoker
        """
        tika_service_url = get_config_value(
            config,
            "TIKA_SERVICE_URL",
            "tika_service_url",
            "Tike Service URL",
            None,
        )
        if tika_service_url is None:
            raise ValueError("No tika service url provided")
        return cls(tika_service_url=tika_service_url)

    def invoke(self, content: str) -> str:
        input_document = self._decode_input(content)
        parsed_result = parser.from_buffer(
            input_document.byte_io,
            serverEndpoint=self._tika_service_url,
        )
        parsed_document = ParsedDocument(
            filename=input_document.filename,
            document_text=parsed_result["content"],
            document_metadata=parsed_result["metadata"],
        )
        return self._encode_output(parsed_document)
