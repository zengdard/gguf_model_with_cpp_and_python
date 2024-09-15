
# LlamaCpp Python API

This project provides a Python binding for the LlamaCpp library, allowing you to use Llama language models in Python applications. (Soruce form llama.c)

## Features

- Load and use Llama models
- Generate text based on prompts
- Batch processing of multiple prompts
- Multi-threaded text generation
- Configurable context size and thread count

## Prerequisites

- C++ compiler with C++11 support
- Python 3.x
- pybind11
- LlamaCpp library

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/llama-cpp-python-api.git
   ```

2. Build the Python module:
   ```
   cd llama-cpp-python-api
   python setup.py build_ext -i
   ```

## Usage

### Initializing the model

```python
from llama_cpp import LlamaModel

model = LlamaModel(
    model_path="path/to/your/model.bin",
    n_ctx=2048,
    n_threads=-1,
    use_mlock=False,
    use_mmap=True
)
```

### Generating text

```python
prompt = "Once upon a time"
generated_text = model.generate(prompt, n_predict=128)
print(generated_text)
```

### Batch processing

```python
prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
results = model.batch_process(prompts, n_predict=128)
for result in results:
    print(result)
```

## API Reference

### LlamaModel

#### Constructor

```python
LlamaModel(model_path: str, n_ctx: int = 2048, n_threads: int = -1, use_mlock: bool = False, use_mmap: bool = True)
```

- `model_path`: Path to the Llama model file
- `n_ctx`: Context size (default: 2048)
- `n_threads`: Number of threads to use (-1 for auto-detect, default: -1)
- `use_mlock`: Use mlock to keep model in memory (default: False)
- `use_mmap`: Use mmap for faster loading (default: True)

#### Methods

```python
generate(prompt: str, n_predict: int = 128) -> str
```
Generates text based on the given prompt.

```python
batch_process(prompts: List[str], n_predict: int = 128) -> List[str]
```
Processes a batch of prompts in parallel.

## Contributing

Contributions to this project are welcome. Please feel free to submit issues and pull requests.

## License

[Insert license information here]
