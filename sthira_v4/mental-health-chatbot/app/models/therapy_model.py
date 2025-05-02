from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict

class TherapyLanguageModel:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"):
        """
        Initialize the therapy language model.
        
        :param model_name: Hugging Face model identifier
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def generate_response(self, prompt: str, context: List[Dict] = None, max_length: int = 200) -> str:
        """
        Generate a therapeutic response.
        
        :param prompt: Engineered therapeutic prompt
        :param context: Conversation context
        :param max_length: Maximum response length
        :return: Generated response
        """
        try:
            # Prepare input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Generate response
            outputs = self.model.generate(
                **inputs, 
                max_length=max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            # Decode and return response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        
        except Exception as e:
            print(f"Error in response generation: {e}")
            return "I'm experiencing some difficulties right now. Could you rephrase your message?"