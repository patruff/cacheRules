import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, Optional
import json
import os

class CAGSystem:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        """Initialize the CAG system with a specified model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.cache: Dict[str, Any] = {}
        self.document_content: Optional[str] = None
        
    def preload_document(self, file_path: str) -> None:
        """
        Preload document content and compute KV cache.
        Args:
            file_path: Path to the document file
        """
        # Read document content
        with open(file_path, 'r', encoding='utf-8') as f:
            self.document_content = f.read()
            
        # Tokenize and prepare input
        inputs = self.tokenizer(self.document_content, return_tensors="pt", truncation=True, 
                              max_length=2048)  # Adjust max_length as needed
        
        # Generate KV cache by running a forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, use_cache=True)
            self.cache = outputs.past_key_values
            
        print(f"Document preloaded and cached: {file_path}")

    def generate_response(self, query: str, max_length: int = 100) -> str:
        """
        Generate a response using the cached context.
        Args:
            query: User query
            max_length: Maximum length of the generated response
        Returns:
            Generated response
        """
        if self.cache is None or self.document_content is None:
            raise ValueError("No document has been preloaded. Call preload_document first.")
            
        # Prepare input by combining cached context and query
        combined_input = f"Context: {self.document_content}\nQuestion: {query}\nAnswer:"
        inputs = self.tokenizer(combined_input, return_tensors="pt", truncation=True,
                              max_length=2048)

        # Generate response using cached KV
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                past_key_values=self.cache,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Answer:")[-1].strip()

    def save_cache(self, cache_path: str) -> None:
        """Save the KV cache to disk."""
        cache_data = {
            'document_content': self.document_content,
            'cache': [
                tuple(tensor.cpu().numpy().tolist() for tensor in cache_tuple)
                for cache_tuple in self.cache
            ]
        }
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)

    def load_cache(self, cache_path: str) -> None:
        """Load a previously saved KV cache."""
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
            
        self.document_content = cache_data['document_content']
        self.cache = tuple(
            tuple(torch.tensor(arr) for arr in cache_tuple)
            for cache_tuple in cache_data['cache']
        )

def main():
    # Initialize CAG system
    cag = CAGSystem()
    
    # Path to the document
    document_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dickens.txt')
    cache_path = os.path.join(os.path.dirname(__file__), 'dickens_cache.json')
    
    # Preload and cache document
    cag.preload_document(document_path)
    
    # Save cache for future use
    cag.save_cache(cache_path)
    
    # Example queries
    queries = [
        "What are the main themes in the text?",
        "Who are the main characters?"
    ]
    
    # Generate responses
    for query in queries:
        response = cag.generate_response(query)
        print(f"\nQuery: {query}")
        print(f"Response: {response}")

if __name__ == "__main__":
    main()