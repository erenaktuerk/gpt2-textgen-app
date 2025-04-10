from transformers import pipeline, set_seed, GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
from rouge_score import rouge_scorer

class GPT2TextGenerator:
    def __init__(self):
        """
        Initializes the GPT-2 text generation pipeline with additional capabilities:
        - Zero-shot and few-shot prompting.
        - Temperature and top-k/top-p sampling for diversity control.
        - Perplexity and ROUGE evaluation for output validation.
        - Toxicity filtering.
        """
        self.model_name = "gpt2"
        self.generator = pipeline("text-generation", model=self.model_name, tokenizer=self.model_name)
        set_seed(42)

        # Load GPT-2 model and tokenizer for further processing and evaluation
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)

        # For perplexity calculation
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 1.0, top_k: int = 50, top_p: float = 1.0) -> str:
        """
        Generates text based on a prompt using GPT-2 with zero-shot and few-shot prompting, 
        and temperature and top-k/top-p sampling for creativity control.
        
        Args:
            prompt (str): Input prompt for the model.
            max_length (int): Total length of the generated sequence.
            temperature (float): Sampling temperature.
            top_k (int): The top-k parameter for sampling.
            top_p (float): The top-p parameter for sampling (nucleus sampling).

        Returns:
            str: Generated text.
        """
        output = self.generator(
            prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=1
        )
        return output[0]['generated_text']

    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate the perplexity of the generated text.
        
        Args:
            text (str): The generated text.
        
        Returns:
            float: Perplexity score.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            log_likelihood = outputs.loss
            perplexity = torch.exp(log_likelihood).item()
        return perplexity

    def calculate_rouge(self, generated_text: str, reference_text: str) -> dict:
        """
        Calculate ROUGE score for the generated text versus a reference text.
        
        Args:
            generated_text (str): The generated text.
            reference_text (str): The reference text for comparison.
        
        Returns:
            dict: ROUGE score.
        """
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference_text, generated_text)
        return scores

    def post_process_text(self, text: str) -> str:
        """
        Apply basic post-processing rules to ensure better output quality.
        This includes removing toxic or off-topic content (basic approach).
        
        Args:
            text (str): Generated text to be processed.
        
        Returns:
            str: Post-processed text.
        """
        # Example: Rule to filter out extremely toxic words (this can be enhanced with a more sophisticated method)
        toxic_keywords = ["hate", "violence", "discrimination"]
        for keyword in toxic_keywords:
            if keyword in text.lower():
                return "The generated text contains inappropriate content and was filtered."
        return text

    def generate_and_evaluate(self, prompt: str, reference_text: str, max_length: int = 100, temperature: float = 1.0, top_k: int = 50, top_p: float = 1.0) -> dict:
        """
        Generate text based on a prompt and evaluate it using perplexity and ROUGE.
        Also applies post-processing to ensure the text is suitable for deployment.
        
        Args:
            prompt (str): The input prompt.
            reference_text (str): The reference text for evaluation.
            max_length (int): Maximum length for generated text.
            temperature (float): Sampling temperature.
            top_k (int): Top-k sampling parameter.
            top_p (float): Top-p (nucleus sampling) parameter.

        Returns:
            dict: A dictionary containing the generated text, perplexity, and ROUGE scores.
        """
        generated_text = self.generate_text(prompt, max_length, temperature, top_k, top_p)
        processed_text = self.post_process_text(generated_text)
        perplexity = self.calculate_perplexity(processed_text)
        rouge_scores = self.calculate_rouge(processed_text, reference_text)

        return {
            "prompt": prompt,
            "generated_text": processed_text,
            "perplexity": perplexity,
            "rouge_scores": rouge_scores
        }