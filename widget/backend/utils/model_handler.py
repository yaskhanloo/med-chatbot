"""
Model Handler using vLLM for Microsoft Phi
"""
from vllm import LLM, SamplingParams
import logging

logger = logging.getLogger(__name__)


class ModelHandler:
    """Handles Microsoft Phi model loading and inference using vLLM"""
    
    def __init__(self, model_id="microsoft/phi-3.5-mini-instruct"):
        """
        Initialize the model handler with vLLM
        
        Args:
            model_id: HuggingFace model identifier
        """
        self.model_id = model_id
        self.llm = None
        self._load_model()
    
    def _load_model(self):
        """Load the model using vLLM"""
        try:
            logger.info(f"Loading model with vLLM: {self.model_id}...")
            
            # Initialize vLLM
            self.llm = LLM(
                model=self.model_id,
                trust_remote_code=True,
                dtype="float16",  # Use float16 for older GPUs (compute capability < 8.0)
                max_model_len=4096,
                gpu_memory_utilization=0.9,
            )
            
            logger.info("vLLM model loaded successfully!")

        except Exception as e:
            logger.error(f"Failed to load model with vLLM: {str(e)}")
            raise
    
    def is_loaded(self):
        """Check if model is loaded"""
        return self.llm is not None

    def generate_response(self, user_message, conversation_history=None, max_tokens=512):
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
            # Prepare messages with system prompt
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful medical assistant. Be conversational and concise. For greetings, respond naturally and briefly. For medical questions, provide clear, structured information in 2-3 focused paragraphs."
                }
            ]
            
            # Add conversation history
            if conversation_history:
                messages.extend(conversation_history)
            
            # Add current user message
            messages.append({"role": "user", "content": user_message})
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.9,
                top_k=40,
                max_tokens=max_tokens,
                repetition_penalty=1.1,  # Reduced to prevent cutting off valid content
                stop=["<|end|>", "<|endoftext|>", "\n\nUser:", "\n\nHuman:", "User:", "Human:", "\n\n\n"],                
            )
            
            # vLLM will automatically apply the correct chat template for Phi
            # Just pass the messages directly
            outputs = self.llm.chat(
                messages=[messages],
                sampling_params=sampling_params,
                use_tqdm=False
            )
            
            # Extract response
            response = outputs[0].outputs[0].text
            
            # Clean up
            response = self._clean_response(response)
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def _clean_response(self, response):
        """Clean up the generated response"""
        response = response.strip()
        
        # Remove special tokens
        for token in ["<|end|>", "<|endoftext|>", "<|user|>", "<|assistant|>", "<|system|>"]:
            response = response.replace(token, "")
        
        # Stop at conversation markers
        for marker in ["User:", "Human:", "Assistant:"]:
            if marker in response:
                response = response.split(marker)[0]
        
        # Collapse excessive newlines (but keep paragraph structure)
        while '\n\n\n' in response:
            response = response.replace('\n\n\n', '\n\n')
        
        response = response.strip()
        
        # Only trim if response doesn't end properly AND is clearly incomplete
        if response and len(response) > 100:  # Only clean longer responses
            if not response[-1] in '.!?' and not response.endswith('...'):
                # Find the last complete sentence
                for punct in ['. ', '! ', '? ']:
                    last_idx = response.rfind(punct)
                    if last_idx > len(response) * 0.7:  # Only trim if we're near the end
                        response = response[:last_idx + 1]
                        break
        
        return response
    
    def reset_conversation(self):
        """Reset conversation state"""
        pass  # vLLM is stateless, nothing to reset
