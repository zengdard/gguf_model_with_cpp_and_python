import llama_cpp
import logging
from typing import List
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LlamaTranslator:
    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int = -1):
        try:
            logger.info(f"Initializing LlamaTranslator with model: {model_path}")
            self.model = llama_cpp.LlamaModel(model_path, n_ctx=n_ctx, n_threads=n_threads)
            logger.info("LlamaTranslator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LlamaTranslator: {str(e)}")
            raise

    def translate(self, text: str, target_language: str = "French") -> str:
        try:
            prompt = f"Translate the following text to {target_language}: {text}"
            logger.debug(f"Translating text: {text[:50]}...")
            result = self.model.generate(prompt, n_predict=256)
            logger.debug(f"Translation completed: {result[:50]}...")
            return result
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            raise

    def batch_translate(self, texts: List[str], target_language: str = "French") -> List[str]:
        try:
            prompts = [f"Translate the following text to {target_language}: {text}" for text in texts]
            logger.info(f"Starting batch translation of {len(texts)} texts")
            start_time = time.time()
            results = self.model.batch_process(prompts, n_predict=256)
            end_time = time.time()
            logger.info(f"Batch translation completed in {end_time - start_time:.2f} seconds")
            return results
        except Exception as e:
            logger.error(f"Batch translation failed: {str(e)}")
            raise

def process_book(book_path: str, translator: LlamaTranslator, chunk_size: int = 1000) -> str:
    try:
        logger.info(f"Processing book: {book_path}")
        with open(book_path, 'r') as file:
            content = file.read()

        # Split the book into manageable chunks
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        logger.info(f"Book split into {len(chunks)} chunks")

        translated_chunks = translator.batch_translate(chunks)
        translated_book = ''.join(translated_chunks)

        logger.info(f"Book translation completed. Total length: {len(translated_book)} characters")
        return translated_book
    except Exception as e:
        logger.error(f"Failed to process book: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        translator = LlamaTranslator("Phi-3.5-mini-instruct-IQ3_M.gguf")
        
        # Optionally apply LoRA for translation task
        # translator.model.set_lora("path/to/translation_lora.bin")

        translated_book = translator.translate("Implement more robust error handling in the C++ code, especially around memory management and graph operations.")

        print(translated_book)
        
        logger.info("Translation process completed successfully")
    except Exception as e:
        logger.critical(f"An error occurred during the translation process: {str(e)}")
