% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/1_1_textEmbed.R
\name{textEmbedLayerAggregation}
\alias{textEmbedLayerAggregation}
\title{Aggregate layers}
\usage{
textEmbedLayerAggregation(
  word_embeddings_layers,
  layers = "all",
  aggregation_from_layers_to_tokens = "concatenate",
  aggregation_from_tokens_to_texts = "mean",
  return_tokens = FALSE,
  tokens_select = NULL,
  tokens_deselect = NULL
)
}
\arguments{
\item{word_embeddings_layers}{Layers returned by the textEmbedRawLayers function.}

\item{layers}{(character or numeric) The numbers of the layers to be aggregated
(e.g., c(11:12) to aggregate the eleventh and twelfth).
Note that layer 0 is the input embedding to the transformer, and should normally not be used.
Selecting 'all' thus removes layer 0 (default = "all")}

\item{aggregation_from_layers_to_tokens}{(character) Method to carry out the aggregation among
the layers for each word/token, including "min", "max" and "mean" which takes the minimum,
maximum or mean across each column; or "concatenate", which links together each layer of the
word embedding to one long row (default = "concatenate").}

\item{aggregation_from_tokens_to_texts}{(character) Method to carry out the aggregation among the word embeddings
for the words/tokens, including "min", "max" and "mean" which takes the minimum, maximum or mean across each column;
or "concatenate", which links together each layer of the word embedding to one long row (default = "mean").}

\item{return_tokens}{(boolean) If TRUE, provide the tokens used in the specified transformer model (default = FALSE).}

\item{tokens_select}{(character) Option to only select embeddings linked to specific tokens
in the textEmbedLayerAggregation() phase such as "[CLS]" and "[SEP]" (default NULL).}

\item{tokens_deselect}{(character) Option to deselect embeddings linked to specific tokens in
the textEmbedLayerAggregation() phase such as "[CLS]" and "[SEP]" (default NULL).}
}
\value{
A tibble with word embeddings. Note that layer 0 is the input embedding to
the transformer, which is normally not used.
}
\description{
textEmbedLayerAggregation selects and aggregates layers of hidden states to form a word embedding.
}
\examples{
# Aggregate the hidden states from textEmbedRawLayers
# to create a word embedding representing the entire text.
# This is achieved by concatenating layer 11 and 12.
\dontrun{
word_embedding <- textEmbedLayerAggregation(
  imf_embeddings_11_12$context_tokens,
  layers = 11:12,
  aggregation_from_layers_to_tokens = "concatenate",
  aggregation_from_tokens_to_texts = "mean"
)

# Examine word_embedding
word_embedding
}
}
\seealso{
See \code{\link{textEmbedRawLayers}} and \code{\link{textEmbed}}.
}
