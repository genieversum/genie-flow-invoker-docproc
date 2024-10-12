from typing import TypeVar, Generic, get_args, Type

from pydantic import BaseModel

from genie_flow_invoker.codec import AbstractInputDecoder, AbstractOutputEncoder

T = TypeVar("T", bound=BaseModel)


def extract_input_model_class(cls: Type[object]) -> Type[BaseModel]:
    for c in cls.__orig_bases__:
        try:
            if issubclass(c.__origin__, AbstractInputDecoder):
                return get_args(c)[0]
        except AttributeError:
            pass
    raise ValueError(f"Cannot extract input model class from {cls}")


# def extract_output_model_class(cls: Type[object]) -> Type[BaseModel]:
#     for c in cls.__orig_bases__:
#         if isinstance(c, AbstractOutputEncoder):
#             return get_args(c)[0]
#     raise ValueError(f"Cannot extract output model class from {cls}")


class PydanticInputDecoder(AbstractInputDecoder, Generic[T]):
    """
    A Mixin that decodes JSON input into a Pydantic model object. The type of model
    is defined as a Generic type when mixing this class in. It will figure out what
    subclass of pydantic.BaseModel has been given and create a new instance of that
    class.

    For example:
    ```
    class SomeInvoker(GenieInvoker, PydanticInputDecoder[SomeInputModel]):
        ...
    ```

    This invoker class will have the method `self._decode_input` that will take a
    JSON string in and return a `SomeInputModel` instance populated with the data
    from that JSON string.
    """

    def _decode_input(self, content: str) -> T:
        """
        Decode the string input into a Pydantic model object. The string input is
        expected to be a JSON representation. The Pydantic model class is defined
        by the type hint used to declare this Mixin.

        :param content: the JSON string carrying the data to decode
        :return: a Pydantic model instance of type T
        """
        cls = extract_input_model_class(self.__class__)
        return cls.model_validate_json(content)


class PydanticOutputEncoder(AbstractOutputEncoder, Generic[T]):
    """
    This Encoder Mixin turns a Pydantic model instance into a JSON representation of
    the object.
    """

    def _encode_output(self, output: T) -> str:
        """
        Turn a Pydantic model instance of type T into a JSON representation of that
        object.

        :param output: the Pydantic model of type T to output as JSON
        :return: a JSON representation of the object
        """
        return output.model_dump_json()
