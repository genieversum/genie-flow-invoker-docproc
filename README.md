# Genie Flow Invoker Document Process
The invokers in this package involve the processing of documents. Documents can be anything
like a Microsoft Word, PowerPoint, PDF or a simple text document. The following process steps
are implemented:

* *PARSE* - turning a binary document file into text
* *CLEAN* - removing spurious elements that should not be part of the text
* *CHUNK* - breaking up a text into smaller parts, following different strategies
* *EMBED* - adding a vector embedding to a piece of text
* *SEARCH* - given a list of vectors, find the nearest neighbors of a search-text

## Parsing a Document - the `DocumentParseInvoker`
For parsing we use [Apache Tika](https://tika.apache.org/), a general purpose parsing engine
that can turn many known binary document formats into plain text.

The input that is expected for this invoker is the `filename` and `document_data`, a base64
encoded representation of the binary file.

The output contains the same `filename`, some further meta data and the complete text that
has been parsed from the document.

## Cleaning a Document - the `DocumentCleanInvoker`
tbd

## Chunking a Document - the `DocumentChunkInvoker`
tbd

## Embedding a Document - the `DocumentEmbedInvoker`
tbd