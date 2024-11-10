from typing import Callable, Sequence

import backoff
from loguru import logger


class BackoffCaller:

    def __init__(
        self,
        retry_exceptions: type[Exception] | Sequence[type[Exception]],
        logging_class: type[object],
        backoff_max_time=61,
        backoff_max_tries=10,
    ):
        self._retry_exceptions = retry_exceptions
        self._logging_class = logging_class
        self._backoff_max_time = backoff_max_time
        self._backoff_max_tries = backoff_max_tries

    def backoff_logger(self, details):
        logger.info(
            "Backing off {wait:0.1f} seconds after {tries} tries ",
            "for a {cls} invocation",
            **details,
            cls=self._logging_class,
        )

    def call(self, func: Callable, *args, **kwargs):

        @backoff.on_exception(
            wait_gen=backoff.fibo,
            max_value=self._backoff_max_time,
            max_tries=self._backoff_max_tries,
            exception=self._retry_exceptions,
            on_backoff=self.backoff_logger,
        )
        def make_call():
            return func(*args, **kwargs)

        return make_call()
