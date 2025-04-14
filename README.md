# Genie Flow Invoker Document Process
The invokers in this package involve the processing of documents. Documents can be anything
like a Microsoft Word, PowerPoint, PDF or a simple text document. The following process steps
are implemented:

* *PARSE* - turning a binary document file into text
* *CLEAN* - removing spurious elements that should not be part of the text
* *CHUNK* - breaking up a text into smaller parts, following different strategies
* *EMBED* - adding a vector embedding to a piece of text
* *SEARCH* - given a list of vectors, find the nearest neighbors of a search-text

## Installing
Installing is done through a normal `pip install` using the appropriate package registry.
After installing, one needs to download a number of NLTK corpora. This can be done by 
executing the command `init_docproc`. This script will download the required corpora and
place them in the standard directory (see https://www.nltk.org/data.html) which is a directory
called `nltk_data` in the user's home directory, or if the environment variable `NLTK_DATA`
is set, into the directory specified.

## The Chunked Document class
The core of this set of invokers revolves around the `ChunkedDocument` object. This is a class
that contains the `filename` of the original document, some possible metadata (a dict with
key-value pairs) and a list of document chunks.

Every `DocumentChunk` contains text, the original span (starting and ending position of that
text within the original document), a hierarchy level and a parent id.

### hierarchy within chunks
Every chunk within a `ChunkedDocument` is set somewhere in a tree. The root of that tree being
the original text, and flowing down into that tree, smaller chunks. Every smaller chunk is
taken from their 'parent' chunk.

The `ChunkedDocument` contains a list of `DocumentChunks` where for every document chunk, 
it is recorded at what level in the tree they sit and who their parent is. The root of the
tree (the complete document) sits at level 0, one level lower into the tree (the first 
level of chunks) all have hierarchy level 1, and chunks out of these (even smaller chunks)
have level 2, etc.

### operation level
A concept for some of the invokers is the "operation level". This is the level inside the
tree to which the invoker is applied. For instance, splitting up all hierarchy level 1
chunks into smaller chunks should be done by specifying `operation-Level=1` in the splitter
invoker. As a result, the `ChunkedDocument` will be extended by new chunks at hierarchy
level 2, and their parents being set to their respective level 1 chunks that they were 
sourced from.

Not specifying an operation level will execute the invoker to all existing chunks.

## Parsing a Document - the `DocumentParseInvoker`
For parsing we use [Apache Tika](https://tika.apache.org/), a general purpose parsing engine
that can turn many known binary document formats into plain text.

The input that is expected for this invoker is the `filename` and `document_data`, a base64
encoded representation of the binary file.

The output is a `ChunkedDocument` that contains the same `filename`, some further meta
data and one chunk pertaining to the complete text that has been parsed from the document.

When parsing an empty document, the resulting `ChunkedDocument` will only contain the filename,
an empty meta data dictionary and an empty list of chunks.

## Cleaning a Document - the `DocumentCleanInvoker`
The `DocumentCleanInvoker` takes in a `ChunkedDocumetn` and "cleans' the content of it's
chunks. The following cleaning operations have been defined:

`clean_multiple_newlines`
: This will reduce any sequence of two or more newlines back to just one newline character.

`clean_multiple_spaces`
: This will reduce any sequence of two or more whitespace characters back to just one space.

`clean_tabs`
: This will reduce any sequence of one or more tab characters back to just one space.

`clean_numbers`
: This will replace numbers larger than 9 to a sequence of `#` characters of a length
equal to the number of digits of the original number. Beyond five digits, the length 
of the sequence will remain five (so `#####`). If a number is split by `,` or `.` characters,
the digits are treated as they are and these `,` and `.` are left alone. For example:
the number `3.14159265` would be replaced by `3.#####`.

`special_term_replacements`
: This will replace any predefined term with something else. If a value of `True` is given
the default replacements are used. Alternatively, a dictionary of from:to pairs can be
specified. The default replacements are:

```python
SPECIAL_TERMS = {
    "i.e.": "%%IE%%",
    "e.g.": "%%EG%%",
    "etc.": "%%ETC%%",
    ".com": "%%DOTCOM%%",
    "www.": "%%WWW%%",
}
```

`tokenize_detokenize`
: This complex cleaner uses NLTK tokenizers to split a text into tokens, and then recompiles
these tokens back into a sentence.

### tokenize - detokenize cleaning
We use a number of tokenizers from the [NLTK](https://www.nltk.org/) package.

The aim for this cleaning strategy is to ensure sentences are properly identified as full sentences.
Taking into account, for instance, that a period character after an abbreviation does not indicate the
end of a sentence. Or, if there is no period at the end of a sentence but the next sentence starts with
a word that regularly starts a new sentence, this is still seen as two separate sentences. And a similar
approach is followed for the words within each of these sentences.

After using this cleaning strategy, the resulting text will contain full sentences that are
appropriately spelled and constructed.

Breaking a text up in sentences is done using the [`PunktSentenceTokenizer`](https://www.nltk.org/api/nltk.tokenize.PunktSentenceTokenizer.html).

> A sentence tokenizer which uses an unsupervised algorithm to build a model for abbreviation words,
> collocations, and words that start sentences; and then uses that model to find sentence boundaries. 
> This approach has been shown to work well for many European languages.

The resulting sentences are then tokenized using the [`TreebankWordTokenizer`](https://www.nltk.org/api/nltk.tokenize.treebank.html#nltk.tokenize.treebank.TreebankWordTokenizer).

> The Treebank tokenizer uses regular expressions to tokenize text as in Penn Treebank.
> This tokenizer performs the following steps:
> * split standard contractions, e.g. don't -> do n't and they'll -> they 'll
> * treat most punctuation characters as separate tokens
> * split off commas and single quotes, when followed by whitespace
> * separate periods that appear at the end of line

The resulting list of lists (sentences of words) are then de-tokenized using first the [`TreebankWordDetokenizer`](https://www.nltk.org/api/nltk.tokenize.treebank.html#nltk.tokenize.treebank.TreebankWordDetokenizer)
to reconstruct sentences. These sentences are then joined together with a newline character.

## Chunking a Document
For chunking texts, this package currently implements two strategies: the `FixedWordCountSplitterInvoker`
that splits texts into fixed size chunks, and the `LexicalDensitySplitInvoker` which uses 
the concept of [lexical density](https://en.wikipedia.org/wiki/Lexical_density) to create
chunks that contain a similar information density.

The input into a Chunking Invoker is a `ChunkedDocument`. Depending on whether an `operation_level`
has been set, the "chunking" is applied to all or only chunks or only the chunks at the given
level of the hierarchy.

The output of these invokers is the same `ChunkedDocument`, where new chunks are added to the
list of chunks. For these newly added chunks, the hierarchy level is one higher than the chunk
that they are based on, and their parent is set to the id of that same parent chunk.

### Fixed Word splitting
The `FixedWordsSplitter` splits a text into windows of a fixed size, then moves the window a 
predefined number of words and creates a new chunk.

This splitter has the ability to ignore stop words. These words are not counted towards the
number of words that are included in the chunk. Stop words are taking from the NLTK corpus
for English stopwords (see https://www.nltk.org/nltk_data/).

When the window cannot be filled to the number of required words, because there are no words
left in the (remainder of the) sentence, the default behavior is to produce smaller chunks
in the trailing end of the sentence. With a flag one can prevent this from happening and only
produce chunks that have the configured number of words.

Words are created using the [`TreebankWordTokenizer`](https://www.nltk.org/api/nltk.tokenize.treebank.html#nltk.tokenize.treebank.TreebankWordTokenizer)
and new chunks are created by it's counterpart [`TreebankWordDetokenizer`](https://www.nltk.org/api/nltk.tokenize.treebank.html#nltk.tokenize.treebank.TreebankWordDetokenizer).

There following settings can be made:

`max_words`
: The maximum number of words that should fit into one chunk.

`overlap`
: The number of words to skip each time for a new chunk to be created.

`ignore_stop_words`
:  (default False) A boolean indicating if (English) stopwords should be ignored when counting.

`drop_trailing`
:  (default False) Whether to drop trailing windows that do not have enough (`max_words') words.

`operation_level`
: (default `None`) The level in the hierarchy that this invoker needs to be applied to.

### Lexical Density Splitting
The concept of [lexical density](https://en.wikipedia.org/wiki/Lexical_density) determines how
much "real information" is contained in a chunk.

> Lexical density estimates the linguistic complexity in a written or spoken composition from
> the functional words (grammatical units) and content words (lexical units, lexemes).

The lexical density is calculated in this package is by taking the fraction of lexical words
over the total number of words. Lexical words are Nouns, Adjectives, Verbs and Adverbs. The
package uses NLTK Part of Speech tagging to assign POS tags to all words in a chunk.

The user of this invoker should set a minimum fraction of lexical density to be reached, as
well as a min and max number of words in a chunk. The strategy then determines if the shortest
or longest chunk is found that fits within these bounds (more than `min_words`, less than
`max_words` and with a lexical density larger than `target_density`. An alternative strategy
would be to find the chunk that has the highest density.

The `LexicalDensitySplitter` has the following configurations:

`min_words`
: The minimal number of words that should go into a new chunk.

`max_words`
: The maximum number of words that go into a chunk.

`overlap`
: The number of words to skip forward before the next chunk window is started.

`target_density`
: A number between 0.0 and 1.0 that sets the threshold for the lexical density. Chunks that
do not have lexical density more than this will not be created.

`strategy`
: (default `best`) a string that can be either `shortest`, `best` or `longest`

### Transcript Splitting
A transcript in the form of a [WebVTT file](https://en.wikipedia.org/wiki/WebVTT) will be
chunked in a chunk per speaker. So, a parent chunk containing the following:

```plain
WEBVTT

ec4ac57d-840f-4f63-a0c1-808e3e218362/13-0
00:00:03.317 --> 00:00:04.797
<v Jochem Michiels>Maybe my English is a bit Dutch but.</v>

ec4ac57d-840f-4f63-a0c1-808e3e218362/32-0
00:00:06.807 --> 00:00:10.249
<v Hanna Berg>OK. But before we start,
I thought it might be good to have a</v>

```
will be split into two child-chunks, each pertaining to the captions in the example.

Consecutive captions of the same speaker are merged into one `DocumentChunk`. 

Every child chunk will have an `original_span` being the start and end second of the chunk.
Also, the following custom attributes (available to Genie Flow invokers from version 0.6.4
of the `genie-flow-invoker` package) will be added:

`party_name`
: the name of the speaker, if given in the transcript, or "UNKNOWN" if no name given.

`seconds_start`
: the second from the start of the transcript in which the statement started being made

`seconds_end`
: the second form the transcript in which the statement finished being made

`duration`
: the number of seconds the statement took being made

`identifier`
: an optional identifier if provided in the transcript 


## Embedding a Document - the `DocumentEmbedInvoker`
Embedding, or [Word Embedding](https://en.wikipedia.org/wiki/Word_embedding) is the process
of creating a vector that represents a word or piece of text. Based on these vectors, text
that is semantically close to each other, are likely to be related.

The `EmbedInvoker` uses an external service to do the actual vectorization. Typically these
are the [containers](https://github.com/weaviate/t2v-transformers-models) that Weaviate has
put around the Huggingface text-to-vector (t2v) models. When using this embedding invoker
there needs to be a service available that has the interface that Weaviate implemented.

Input into this invoker can be either a `ChunkedDocument` or a string. If the input is a
string, the result is the vector embedding of that string -- a list of floating values. If
the input is a ChunkedDocument, the result is that same ChunkedDocument - but with a vector
embedding for each and every chunk in that document.

As playfully laid out in ["The Art of Pooling Embeddings"](https://blog.ml6.eu/the-art-of-pooling-embeddings-c56575114cf8)
there are a number of different pooling strategies. The default that is implemented by Weaviate
stands at "masked_mean" but others are available and configurable in this invoker.

The following configuration parameters are available:

`text2vec_url`
: The url for the test to vector service.

`pooling_strategy`
: (default `None`) the optional pooling strategy.

`backoff_max_time`
: The maximum time this invoker should wait before re-trying a failed request.

`backoff_max_tries`
: The maximum number of retries to the service.

## Similarity Search
The `SimilaritySearchInvoker` implements the search for similar chunks in a `ChunkedDocument`.

Input into this invoker is a `SimilaritySearch` object, the output is a `SimilarityResults`
object.

### SimilaritySearch
This object contains the relevant information for conducting a similarity search. Of course
the chunks to search in, and the query to search for. But also parameters that inform the 
invoker on how to limit the results, how to deal with chunk parents and what distance metric
it should use.

The list of chunks is expected to be the same as the list of `DocumentChunk`s that is maintained
in a `ChunkedDocument`. So we can just get `chunked_document.chunks` to go into that field.

The query should be passed as a vector. This can be easily found using the `EmbeddingInvoker`
where the input is just the search string. The output of that invoker (a list of floating
values) is the search query vector.

We can also instruct this invoker to only work at a certain operation level. This means that
the search will only be conducted within chunks that have the given hierarchy level.

By default, this invoker will return _all_ chunks that are searched, in descending distance
from the search query vector. These results can be limited by specifying a horizon and/or a
top. The first specifying the maximum distance that is still acceptable. The latter specifying
exactly how many results are expected. The result is limited by both these factors if they are
specified.

Two parent strategies have been implemented: `include` and `replace`. The former will just
include any parent of the chunks that are retrieved. The latter will drop the children and
replace them with their parents.

#### Configuration
Configuration of this invoker follows a slightly different pattern than with the other invokers
in this package. The `meta.yaml` file does not have to contain any parameters. There are no
services to configure or other meta information to pass. All the configuration is done via
the input that is sent to the invoker.

However, when parameters are specified in `meta.yaml`, they form a default value and can be
left out of the input JSON. So, if the `meta.yaml` specifies `method: cosine`, this does not
have to be specified in the input.

The following parameters are available:

`filename`
: The name of the originating file.

`chunks`
: the list of `DocumentChunk` objects to search within. These chunks need to be embedded
(meaning that their `vector` field has been set).

`query_emedding`
: The embedding of the query to search for.

`operation_level`
: The optional hierarchy level of the chunks that should be searched within.

`horizon`
: The optional minimum distance a chunk needs to have to be included in the result.

`top`
: The optional maximum number of result chunks that will be returned.

`parent_strategy`
: The optional strategy to follow for including any parents: include or replace.

`method`
: The distance metric to be used. Can be `cosine` (the default), `euclidean` or `manhattan`.

`include_vector`
: (default False) A boolean indicating if the embedding vector of the resulting chunks
should also be included in the output.

### SimilarityResults
The resulting object contains a list of the chunks that have been found, in order of distance.
These chunks are represented by the `DocumentChunk` object, accompanied by their distance
towards the search vector.

This information is represented in an object with the fields `chunk` and `distance`. The
first containing the full `DocumentChunk` data and the second containing a floating point
value representing the distance.

If the configuration parameter `include_vector` is `False` (the default), then the resulting
`DocumentChunk`s will not contain their `embedding` vector.