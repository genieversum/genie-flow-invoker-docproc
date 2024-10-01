# Genie Flow Invoker Document Process

This group of invokers delivers the ability to process documents, using the following process steps:

_parse_
: the process of turning any document into clean text. Reading a Microsoft Word or Powerpoint, a PDF or even an HTML page and outputting clean text.

_clean_
: the process of removing surpurfluous text, such as extra space characters, non-printable characters etc.

_chunk_
: the process of chunking a clean text block into smaller chunks. A number of chunking strategies is available.

_embed_
: the process of finding a position in hyperspace of a given text, changing a chunk into a vector.

This invoker aims to be the 'Swiss Army Knife' for processing documents. The user of this invoker should be able to call either of the listed processes separately, or place some of these processes into a graph of tasks.