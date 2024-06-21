# TuneX

**TuneX** is a powerful command-line tool designed to provide a comprehensive solution for working with Large Language Models (LLMs). It offers a unified interface for running, fine-tuning, and instruction tuning LLMs, making it an essential utility for researchers and developers in the field of natural language processing and artificial intelligence.

Key features of TuneX include:
- Support for multiple LLM architectures (GPT2, Llama, Mistral, Gemma)
- Flexible tokenizer options
- Chat interface support
- Various prompt style options
- Advanced text generation techniques (Top-p, Top-k, Beam Search)
- Extensive fine-tuning capabilities, including full fine-tuning, adapters, and LoRA
- Instruction tuning based on human preferences (RLHF, PPO, DPO, RLOO)
- Comprehensive documentation with examples

TuneX simplifies complex LLM-related tasks through an intuitive command-line interface, allowing users to easily run models, fine-tune on custom datasets, and implement advanced instruction tuning techniques. Whether you're a beginner experimenting with LLMs or an experienced researcher pushing the boundaries of AI, TuneX provides a streamlined, command-line driven approach to support your work.

## Installation

```bash
pip install tunex
```

## Quick start

```bash
# tunex [action] [checkpoit directory / model]
tunex	download  gpt2
tunex	chat      checkpoints/gpt2
tunex	list
```

### Listing Supportive models

```bash
tunex list
>>
 _____                __  __
|_   _|   _ _ __   ___\ \/ /
  | || | | | '_ \ / _ \\  / 
  | || |_| | | | |  __//  \ 
  |_| \__,_|_| |_|\___/_/\_\
                            

Supported models: 

1 gpt2
2 gpt2-medium
3 gpt2-large
4 gpt2-xl
```

### Downloading and chatting with models

```bash
tunex download "gpt2"
# download the gpt2 model and store it within "checkpoints/gpt2" by default
```

```bash
tunex chat "checkpoints/gpt2"

>>
 _____                __  __
|_   _|   _ _ __   ___\ \/ /
  | || | | | '_ \ / _ \\  / 
  | || |_| | | | |  __//  \ 
  |_| \__,_|_| |_|\___/_/\_\
                            

Initiating chat mode with gpt2

Setting sead to 42

>> Prompt: 
```

### General help

```bash
# tunex [action] -h
tunex download -h
>>
Download weights or tokenizer data from the Hugging Face Hub.

positional arguments:
  repo_id               The repository ID in the format ``org/name`` or ``user/name`` as shown in Hugging Face. (required, type: str)

optional arguments:
  -h, --help            Show this help message and exit.
  --config CONFIG       Path to a configuration file.
  --print_config[=flags]
                        Print the configuration after applying all other arguments and exit. The optional flags customizes the output and are one or more keywords
                        separated by comma. The supported flags are: comments, skip_default, skip_null.
  --access_token ACCESS_TOKEN
                        Hugging Face API token to access models with restrictions. (type: Union[str, null], default: null)
  --tokenizer_only {true,false}
                        Whether to download only the tokenizer files. (type: bool, default: False)
  --convert_checkpoint {true,false}
                        Whether to convert the checkpoint files from hugging face format after downloading. (type: bool, default: True)
  --checkpoint_dir CHECKPOINT_DIR
                        Where to save the downloaded files. (type: <class 'Path'>, default: checkpoints)
  --model_name MODEL_NAME
                        The existing config name to use for this repo_id. This is useful to download alternative weights of existing architectures. (type:
                        Union[str, null], default: null)
```



## Features Roadmap

- [ ] Multiple LLM support
  - [x]  GPT2
  - [ ] Llama
  - [ ] Mistral
  - [ ] Gemma
- [ ] Support for different Tokenizers
- [x] Chat Interface support
- [ ] Different Prompt Style support
- [ ] Text generation
  - [x] Top-p
  - [x] Top-k
  - [ ] Beam Search
- [ ] Finetuning Support with different datasets
  - [ ] Full finetuning
  - [ ] Adaptars
  - [ ] LoRA
- [ ] Instruction Tuning on Human Preferences
  - [ ] RLHF
  - [ ] PPO
  - [ ] DPO
  - [ ] RLOO
- [ ] Comprehensive Documentation with example

## Acknowledgements

- [@litgpt](https://github.com/Lightning-AI/litgpt)
- [@torchtune](https://github.com/pytorch/torchtune)