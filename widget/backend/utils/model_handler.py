"""
Model Handler for MedGemma
Handles model loading and inference
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class ModelHandler:
    """Handles MedGemma model loading and inference"""
    
    def __init__(self, model_id="google/medgemma-4b-it"):
        """
        Initialize the model handler
        
        Args:
            model_id: HuggingFace model identifier
        """
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer"""
        try:
            logger.info(f"Loading tokenizer from {self.model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            logger.info(f"Loading model from {self.model_id}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=torch.float32,  # Critical: use float32
                low_cpu_mem_usage=True
            )
            self.model.eval()
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def is_loaded(self):
        """Check if model is loaded"""
        return self.model is not None and self.tokenizer is not None

    def generate_response(self, user_message, conversation_history=None, max_tokens=300):
        """
        Generate response for user message
        
        Args:
            user_message: User's input message
            conversation_history: List of previous messages [{"role": "user/assistant", "content": "..."}]
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded")
        
        try:
            # Prepare messages with conversation history
            messages = []
            
            # Add system-like instruction as first user message
            if not conversation_history or len(conversation_history) == 0:
                messages.append({
                    "role": "user",
                    "content": "You are a helpful medical assistant. Keep responses concise (2-3 paragraphs maximum), clear, and conversational. Avoid long bullet lists unless specifically asked."
                })
                messages.append({
                    "role": "assistant",
                    "content": "I understand. I'll provide concise, clear medical information."
                })
            
            # Add conversation history if provided
            if conversation_history:
                messages.extend(conversation_history)
            
            # Add current user message
            messages.append({"role": "user", "content": user_message})

            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt"
            ).to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_tokens,
                    min_new_tokens=20,
                    do_sample=True,
                    temperature=0.6, # lower for more focused response
                    top_p=0.8,       # lower for less variety
                    repetition_penalty=1.2, # discourage repetition
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    use_cache=True,
                    early_stopping=True # stop when eos is generated?
                )
            
            # Decode only the generated tokens
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(
                generated_tokens, 
                skip_special_tokens=True
            ).stip() # decode only the new tokens?
            
            # Clean up response
            response = response.strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
