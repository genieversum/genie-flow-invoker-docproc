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

    def _decode_input(self, content: str) -> T:
        cls = extract_input_model_class(self.__class__)
        return cls.model_validate_json(content)


class PydanticOutputEncoder(AbstractOutputEncoder, Generic[T]):

    def _encode_output(self, output: T) -> str:
        return output.model_dump_json()
