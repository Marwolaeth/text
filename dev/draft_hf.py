import torch
import huggingface_hub
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import pipeline
from tokenizers import XLMRobertaTokenizerFast

model = 'manifesto-project/manifestoberta-xlm-roberta-56policy-topics-sentence-2024-1-1'
model = 'sergeyzh/rubert-tiny-turbo'

config = AutoConfig.from_pretrained(model, output_hidden_states=True)
config
config.tokenizer_class
config.do_lower_case

AutoTokenizer.__name__
tokenizer = AutoTokenizer.from_pretrained(model)
transformer_model = AutoModel.from_pretrained(model, config=config, trust_remote_code=True)

pipe = pipeline(
  'text-classification',
  model = model
)

AutoTokenizer.from_pretrained(f'{config.model_type}-base-uncased')
tokenizer = AutoTokenizer.from_pretrained(f'{config.model_type}-large')
transformers.XLMRobertaTokenizer.from_pretrained(model)

tokenizer.save_pretrained(
  'manifestoberta-xlm-roberta-56policy-topics-sentence-2024-1-1'
)

tokenizer.save_vocabulary(
  'manifestoberta-xlm-roberta-56policy-topics-sentence-2024-1-1'
)

## Text Generation ----
txt = "Umbrella представляет революционный продукт, который изменит представление о микропроцессорах."
context_innovative = "Это инновационная компания."
context_conservative = "Это консервативная компания."

txts = [f"{txt} {context}" for context in [context_conservative, context_innovative]]

res = hgTransformerGetTextGeneration(
  txts,
  'gpt2',
  max_new_tokens = 1,
  output_scores = True
)
res

## Perplexity ----
import math

sequences = [
  "Umbrella представляет революционный продукт, который изменит представление о микропроцессорах.",
  "В последнее время Umbrella совсем перестала развивать свои решения."
]
zsresult = hgTransformerGetZeroShot(
  sequences,
  candidate_labels = ['консервативная компания', 'инновационная компания'],
  model = 'Marwolaeth/rosberta-nli-terra-v0',
  hypothesis_template = "Umbrella – {}."
)

text = "Y представляет революционный продукт, который изменит представление о микропроцессорах."
text = "В последнее время Y совсем перестала развивать новые решения."
context_innovative = f"{text} Y — очень инновационная компания."
context_conservative = f"{text} Y — очень консервативная компания."
context_normal = f"{text} Y — обычная солидная компания."
text_strings = [context_conservative, context_normal, context_innovative]

phrase = text_strings[1]

def calc_loss(
  phrase: str,
  model,
  tokenizer,
  device
):

    phrase = tokenizer.encode(phrase)
    # If the phrase is only 1 token long, an error can occur
    if len(phrase) == 1:
         phrase.append(tokenizer.eos_token_id)
    phrase = torch.tensor(phrase, dtype=torch.long, device=device)
    phrase = phrase.unsqueeze(0)
    with torch.no_grad():
        out = model(phrase, labels=phrase)
    
    return out.loss.item()



def hgTransformerGetZeroShotPerplexity(text_strings,
                            candidate_labels,
                            hypothesis_template = "This example is {}.",
                            model = 'gpt2',
                            device = 'cpu',
                            tokenizer_parallelism = False,
                            logging_level = 'warning',
                            hg_gated = False,
                            hg_token = "",
                            trust_remote_code = False,
                            set_seed = None):
  set_seed = _as_integer(set_seed)
  if isinstance(set_seed, int):
        torch.manual_seed(set_seed)
  set_logging_level(logging_level)
  set_tokenizer_parallelism(tokenizer_parallelism)
  device, device_num = get_device(device)
  
  # check and adjust input types
  if not isinstance(text_strings, list):
      text_strings = [text_strings]
  
  if hg_gated:
      set_hg_gated_access(access_token=hg_token)
  
  tokenizer = AutoTokenizer.from_pretrained(
    model,
    trust_remote_code=trust_remote_code
  )
  transformer_model = AutoModelForCausalLM.from_pretrained(
    model,
    trust_remote_code=trust_remote_code,
    output_hidden_states=False
  )
  tokenizer.add_special_tokens({'pad_token': '[PAD]'})
  tokenizer.pad_token
  
  transformer_model.to(device)
  
  losses = [
    calc_loss(text, transformer_model, tokenizer, device)
    for text in text_strings
  ]
  
  # scores = [0 - loss*50 for loss in losses]
  # torch.softmax(torch.tensor(scores), dim=0)
  
  result = [
    {'sequence': text_strings[i], 'perplexity': losses[i]}
    for i in range(len(losses))
  ]
  
  
