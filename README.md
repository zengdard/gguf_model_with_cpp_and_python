# LlamaTranslator

LlamaTranslator is a Python library that uses the Llama model to perform text translations. It offers individual and batch translation capabilities, as well as the ability to process entire books.

## Features

- Individual text translation
- Batch translation of multiple texts
- Processing and translation of entire books
- Detailed logging for operation tracking

## Prerequisites

- Python 3.x
- llama_cpp library

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/LlamaTranslator.git
   ```

2. Install dependencies:
   ```
   pip install llama_cpp
   ```

## Usage

### Initializing the translator

```python
from llama_translator import LlamaTranslator

translator = LlamaTranslator("path/to/your/model.gguf")
```

### Translating a text

```python
translated_text = translator.translate("Text to translate", target_language="French")
print(translated_text)
```

### Batch translation

```python
texts = ["Text 1", "Text 2", "Text 3"]
translated_texts = translator.batch_translate(texts, target_language="French")
```

### Processing a book

```python
from llama_translator import process_book

translated_book = process_book("path/to/your/book.txt", translator)
```

## Configuration

- You can adjust model parameters (such as `n_ctx` and `n_threads`) when initializing `LlamaTranslator`.
- The chunk size for book processing can be adjusted in the `process_book` function.

## Logging

The script uses Python's `logging` module to provide detailed information about the translation process. Logs are configured to display in the console.

## Note

This code uses the "Phi-3.5-mini-instruct-IQ3_M.gguf" model. Make sure you have the correct model or adjust the path according to your configuration.

## Contributing

Contributions to this project are welcome. Feel free to open an issue or submit a pull request.

