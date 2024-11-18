from abc import ABC, abstractmethod

from genie_flow_invoker.invoker.docproc.model import DocumentChunk


class AbstractSplitter(ABC):

    @abstractmethod
    def split(self, document: DocumentChunk) -> list[DocumentChunk]:
        raise NotImplementedError("Subclass should implement this method")
