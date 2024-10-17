import backoff
from loguru import logger
from tika import parser

from genie_flow_invoker.genie import GenieInvoker
from genie_flow_invoker.invoker.docproc.codec import PydanticInputDecoder, PydanticOutputEncoder
from genie_flow_invoker.utils import get_config_value

from genie_flow_invoker.invoker.docproc.model import RawDocumentFile, ParsedDocument


class DocumentParseInvoker(
    GenieInvoker,
    PydanticInputDecoder[RawDocumentFile],
    PydanticOutputEncoder[ParsedDocument],
):
    """
    Parse a binary document into text and their metadata.
    """

    def __init__(
            self,
            tika_service_url: str,
            backoff_max_time=61,
            backoff_max_tries=10,
    ):
        self._tika_service_url = tika_service_url
        self._backoff_max_time = backoff_max_time
        self._backoff_max_tries = backoff_max_tries


    @classmethod
    def from_config(cls, config: dict):
        """
        The `meta.yaml` for the parser should contain the following properties:
        - tika_service_url: the url of the tika service
        - backoff_max_time: the maximum time in seconds between retries
        - backoff_max_tries: the maximum number of retries

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

        backoff_max_time = get_config_value(
            config,
            "TIKA_BACKOFF_MAX_TIME",
            "backoff_max_time",
            "Max backoff time (seconds)",
            61,
        )
        backoff_max_tries = get_config_value(
            config,
            "TIKA_MAX_BACKOFF_TRIES",
            "backoff_max_tries",
            "Max backoff tries",
            15,
        )

        return cls(
            tika_service_url=tika_service_url,
            backoff_max_time=backoff_max_time,
            backoff_max_tries=backoff_max_tries,
        )

    def invoke(self, content: str) -> str:
        input_document = self._decode_input(content)

        def backoff_logger(details):
            logger.info(
                "Backing off {wait:0.1f} seconds after {tries} tries ",
                "for a {cls} invocation",
                **details,
                cls=self.__class__.__name__,
            )

        @backoff.on_exception(
            wait_gen=backoff.fibo,
            max_value=self._backoff_max_time,
            max_tries=self._backoff_max_tries,
            exception=TimeoutError,
            on_backoff=backoff_logger,
        )
        def parse_with_backoff():
            result = parser.from_buffer(
                input_document.byte_io,
                serverEndpoint=self._tika_service_url,
            )
            if result["status_code"] in [408, 429, 500]:
                logger.warning("Receiving status code {}, from Tika", result["status_code"])
                raise TimeoutError()

        parsed_result = parse_with_backoff()
        parsed_document = ParsedDocument(
            filename=input_document.filename,
            document_text=parsed_result["content"],
            document_metadata=parsed_result["metadata"],
        )
        return self._encode_output(parsed_document)
