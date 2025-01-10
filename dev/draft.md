### Pull Request: More Text Generation Parameters

**Description**: 

This pull request introduces several enhancements to the text generation functionality in the `text` package. The following changes have been made:

- **New Parameters**: Added `max_length`, `max_new_tokens`, `min_length`, and `min_new_tokens` to the `hgTransformerGetTextGeneration` function. These parameters provide users with greater control and are the ones I was missing for my specific use cases. Some minor message edits were also made.

- **R Interface Update**: Updated the `textGeneration` function in the R interface to accept the new parameters and ensure they are passed to the underlying Python function.

- **Expanded Tests**: Added comprehensive tests to validate the new functionality and ensure robustness.

Please note that while I have made these enhancements, the R-CMD-Check still runs into some errors and warnings that are not related to the proposed changes. I also encountered an issue when `nltk` failed to find `tokenizers/punkt_tab`, so I added a modified copy of `punk` downloading script, as a precaution. Additionally, the package authors are welcome to rearrange the new arguments as they see fit.

Thank you for considering this pull request!
