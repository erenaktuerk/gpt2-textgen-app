# app/services/text_generator.py

from transformers import pipeline, set_seed

class GPT2TextGenerator:
    def _init_(self):
        """
        Initializes the GPT-2 text generation pipeline.
        """
        self.generator = pipeline("text-generation", model="gpt2")
        set_seed(42)

    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 1.0) -> str:
        """
        Generates text based on a prompt using GPT-2.

        Args:
            prompt (str): Input prompt for the model.
            max_length (int): Total length of the generated sequence.
            temperature (float): Sampling temperature.

        Returns:
            str: Generated text.
        """
        output = self.generator(
            prompt,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=1
        )
        return output[0]['generated_text']