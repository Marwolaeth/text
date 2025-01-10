library(tok)

model = 'manifesto-project/manifestoberta-xlm-roberta-56policy-topics-sentence-2024-1-1'
model = 'sergeyzh/rubert-tiny-turbo'

tok::tokenizer$from_pretrained(model)
?tokenizer
??from_pretrained
tok::tokenizer$from_pretrained
hfhub::hub_download

?curl::curl_download

textGeneration(
  'A long text that exceed the `max_length` parameter value.',
  model = 'gpt2',
  max_new_tokens = 2
)

textGeneration(
  'A long text that exceed the `max_length` parameter value.',
  model = 'gpt2',
  max_new_tokens = 2L,
  set_seed = 111L
)

textGeneration(
  'A long text that exceed the `max_length` parameter value.',
  model = 'gpt2',
  max_new_tokens = 2L,
  max_length = 1L,
  return_full_text = FALSE,
  set_seed = 111
)

textGeneration(
  'A long text that exceed the `max_length` parameter value.',
  model = 'gpt2',
  max_new_tokens = 2L,
  min_new_tokens = 0,
  set_seed = 111
)

textGeneration(
  'A long text that exceed the `max_length` parameter value.',
  model = 'gpt2',
  max_new_tokens = 2L,
  min_length = 5L,
  set_seed = 111
)

textGeneration(
  'A long text that exceed the `max_length` parameter value.',
  model = 'gpt2',
  max_new_tokens = 0,
  min_new_tokens = 2,
  set_seed = 111
)
