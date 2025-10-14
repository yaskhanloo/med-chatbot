# medgemma_pipeline_fixed.py
from transformers import pipeline, AutoTokenizer, logging
import torch
import warnings

# Suppress the generation flags warning
warnings.filterwarnings('ignore', message='.*generation flags.*')
logging.set_verbosity_error()  # Only show errors

def test_with_pipeline_fixed():
    """
    Fixed pipeline test that properly handles MedGemma's chat template
    and uses stable float32 dtype.
    """
    print("Setting up pipeline with correct settings...")
    
    # Load tokenizer separately to apply chat template
    tokenizer = AutoTokenizer.from_pretrained("google/medgemma-4b-it")
    
    # Create pipeline with float32 (not float16!)
    pipe = pipeline(
        "text-generation",
        model="google/medgemma-4b-it",
        tokenizer=tokenizer,
        torch_dtype=torch.float32,  # FIXED: Use float32
        device_map="auto"
    )
    
    # Original medical report query
    user_prompt = """Given the following medical report, extract all clinical variables with their values:

Patient presents with:
- Temperature: 102°F
- Blood pressure: 120/80 mmHg  
- Heart rate: 88 bpm
- WBC count: 12,000 cells/μL
- Hemoglobin: 14.2 g/dL

Extract the variables in a structured format:"""

    # CRITICAL FIX: Apply chat template manually
    messages = [{"role": "user", "content": user_prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    print("\n" + "="*60)
    print("FORMATTED PROMPT (first 200 chars):")
    print(repr(formatted_prompt[:200]))
    print("="*60)

    print("\nGenerating response...")
    result = pipe(
        formatted_prompt,
        max_new_tokens=250,
        do_sample=False,  # Deterministic
        # Don't set temperature when do_sample=False
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        return_full_text=False,  # Only return generated part
    )
    
    print("\n" + "="*60)
    print("EXTRACTED CLINICAL VARIABLES:")
    print("="*60)
    print(result[0]['generated_text'])
    print("="*60)


def test_with_direct_model():
    """
    Alternative: Skip pipeline API and use model directly
    This gives you more control.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("\n" + "="*60)
    print("ALTERNATIVE: Using model directly (more control)")
    print("="*60)
    
    model_name = "google/medgemma-4b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model.eval()
    
    user_prompt = """Extract clinical variables from:
Temperature 102°F, BP 120/80 mmHg, HR 88 bpm, WBC 12,000 cells/μL"""
    
    messages = [{"role": "user", "content": user_prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True
        )
    
    # Decode only the generated part
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    print("\nDirect model output:")
    print(response)
    print("="*60)


if __name__ == "__main__":
    print("Method 1: Fixed Pipeline")
    test_with_pipeline_fixed()
    
    print("\n\n")
    
    print("Method 2: Direct Model (Recommended)")
    test_with_direct_model()