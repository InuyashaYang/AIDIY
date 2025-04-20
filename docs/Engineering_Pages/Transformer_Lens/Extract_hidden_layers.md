## Extraction of GPT2's 6th layer's post-resid hidden state output

### The reason to choose wikitext-103-raw-v1 for GPT2
First we choose wikitext-103 as our dataset, specifically the version of Salesforces, there exist two versions of wikitexi-103: wikitext-103-v1 and wikitext-103-raw-v1, the difference between the two is that wikitext-103-raw-v1 contains all the character-level token which may not be contained in wiki's vocabulary and has not been replaced by <unk>, wikitext-103-v1, on the other side, is processed to word-level, replacing every rare word to <unk>, it's only for older models that don't adopt a tokenizer.
Obviously, GPT2 suits wikitext-103-raw-v1 more.

### Initiate Reasoning
We would like to record the prediction vector of GPT2 on each sequence, we may acquire an output sequence whose index ranges from 1 to N, of which the index k means the prediction based on the input tokens 0, ... ,k-1. Not every sequence requires the initial token to be the very beginning of the whole dataset, thereby we have to come up with a method to denote the beginning of every sequence.
