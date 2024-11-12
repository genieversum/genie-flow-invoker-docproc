# Genie Flow Invoker Document Process
The invokers in this package involve the processing of documents. Documents can be anything
like a Microsoft Word, PowerPoint, PDF or a simple text document. The following process steps
are implemented:

* *PARSE* - turning a binary document file into text
* *CLEAN* - removing spurious elements that should not be part of the text
* *CHUNK* - breaking up a text into smaller parts, following different strategies
* *EMBED* - adding a vector embedding to a piece of text
* *SEARCH* - given a list of vectors, find the nearest neighbors of a search-text

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
: (default None) The level in the hierarchy that this invoker needs to be applied to.

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

## Embedding a Document - the `DocumentEmbedInvoker`
tbd

## Installing
Installing is done through a normal `pip install` using the appropriate package registry.
After installing, one needs to download a number of NLTK corpora. This can be done by 
executing the command `init_nltk_data`. This script will download the required corpora and
place them in the standard directory (see https://www.nltk.org/data.html) which is a directory
called `nltk_data` in the user's home directory, or if the environment variable `NLTK_DATA`
is set, into the directory specified.
