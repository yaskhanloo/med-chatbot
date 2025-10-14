"""
Model Handler for MedGemma - FIXED VERSION
Prevents hallucination and runaway generation
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import logging

logger = logging.getLogger(__name__)


class EosListStoppingCriteria(StoppingCriteria):
    """Custom stopping criteria that stops on multiple possible end tokens"""
    def __init__(self, eos_sequence, tokenizer):
        self.eos_sequence = eos_sequence
        self.tokenizer = tokenizer
        
    def __call__(self, input_ids, scores, **kwargs):
        # Check if we've generated any of the stop sequences
        last_ids = input_ids[0, -len(self.eos_sequence):].tolist()
        return last_ids == self.eos_sequence


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
        self.system_prompt_added = False  # Track if system prompt was added
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer"""
        try:
            logger.info(f"Loading tokenizer from {self.model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Loading model from {self.model_id}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=torch.float32,  # Critical: use float32
                low_cpu_mem_usage=True
            )
            self.model.eval()
            
            logger.info("Model loaded successfully!")
            logger.info(f"Special tokens - EOS: {self.tokenizer.eos_token} (id: {self.tokenizer.eos_token_id})")
            
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
            # Prepare messages
            messages = []
            
            # Only add system prompt once at the beginning of a new conversation
            if not conversation_history or len(conversation_history) == 0:
                if not self.system_prompt_added:
                    # Add as part of the first user message to avoid confusing the model
                    enhanced_message = (
                        "Instructions: Provide concise medical information (2-3 paragraphs max). "
                        "Be clear and conversational. Now, answer this: " + user_message
                    )
                    messages.append({"role": "user", "content": enhanced_message})
                    self.system_prompt_added = True
                else:
                    messages.append({"role": "user", "content": user_message})
            else:
                # Add conversation history
                messages.extend(conversation_history)
                messages.append({"role": "user", "content": user_message})

            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize with proper truncation
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=False  # Don't pad unnecessarily
            ).to(self.model.device)
            
            # Create stopping criteria for common end patterns
            stopping_criteria = StoppingCriteriaList()
            
            # Add criteria for double newline (often indicates end of response)
            double_newline_ids = self.tokenizer.encode("\n\n", add_special_tokens=False)
            if len(double_newline_ids) > 0:
                stopping_criteria.append(EosListStoppingCriteria(double_newline_ids, self.tokenizer))
            
            # Add criteria for "User:" pattern (indicates model trying to generate next turn)
            user_pattern_ids = self.tokenizer.encode("\nUser:", add_special_tokens=False)
            if len(user_pattern_ids) > 0:
                stopping_criteria.append(EosListStoppingCriteria(user_pattern_ids, self.tokenizer))
            
            # Generate with better control
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_tokens,
                    # REMOVED min_new_tokens - this was forcing generation!
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=40,  # Add top_k for better quality
                    repetition_penalty=1.15,
                    length_penalty=0.8,  # Slightly discourage long outputs
                    eos_token_id=[self.tokenizer.eos_token_id],  # Ensure it's a list
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                    stopping_criteria=stopping_criteria,
                    # Stop when we hit EOS or max length
                    early_stopping=True
                )
            
            # Decode only the generated tokens
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            
            # Decode with special token skipping
            response = self.tokenizer.decode(
                generated_tokens, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Post-process to clean up common issues
            response = self._clean_response(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def _clean_response(self, response):
        """
        Clean up the generated response to remove common artifacts
        
        Args:
            response: Raw generated text
            
        Returns:
            Cleaned response
        """
        # Remove any trailing incomplete sentences
        response = response.strip()
        
        # If response tries to continue the conversation (hallucination pattern)
        if "User:" in response or "Assistant:" in response:
            response = response.split("User:")[0].split("Assistant:")[0]
        
        # Remove repetitive endings
        lines = response.split('\n')
        if len(lines) > 1:
            # Check if last line is repetitive or incomplete
            last_line = lines[-1].strip()
            if len(last_line) < 20 and not last_line.endswith('.'):
                # Likely an incomplete thought, remove it
                response = '\n'.join(lines[:-1])
        
        # Remove multiple consecutive newlines
        while '\n\n\n' in response:
            response = response.replace('\n\n\n', '\n\n')
        
        # Ensure response ends properly
        response = response.strip()
        if response and not response[-1] in '.!?':
            # Find last complete sentence
            for punct in ['.', '!', '?']:
                last_idx = response.rfind(punct)
                if last_idx > 0:
                    response = response[:last_idx + 1]
                    break
        
        return response
    
    def reset_conversation(self):
        """Reset conversation state for new chat session"""
        self.system_prompt_added = False