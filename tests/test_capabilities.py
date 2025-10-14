# test_medgemma_capabilities.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def test_basic_capabilities():
    model_name = "google/medgemma-4b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model.eval()
    
    test_prompts = [
        # Test 1: Basic medical knowledge
        "What is hypertension?",
        
        # Test 2: Medical calculation
        "Convert 102°F to Celsius.",
        
        # Test 3: Clinical interpretation
        "Is a blood pressure of 160/95 mmHg normal?",
        
        # Test 4: Simple extraction
        "What numbers are in this text: Temperature 102F, BP 120/80",
        
        # Test 5: Medical data interpretation
        "Patient has WBC 15,000. Is this normal?",
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {prompt}")
        print("-"*40)
        
        # Use the chat template
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_special_tokens=True,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(
                formatted, 
                return_tensors="pt", 
                truncation=True,
                padding="longest",
                max_length=1024)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the model's response
        if prompt in response:
            model_response = response.split(prompt)[-1].strip()
        else:
            model_response = response.split("model\n")[-1] if "model\n" in response else response
        
        print("RESPONSE:", model_response[:300])

if __name__ == "__main__":
    test_basic_capabilities()
