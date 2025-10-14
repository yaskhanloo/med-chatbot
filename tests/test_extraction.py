# medgemma_clinical_extraction.py
# Final working version - extracts clinical variables from medical text
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "google/medgemma-4b-it"

def extract_clinical_variables(medical_text):
    """
    Extracts clinical variables from medical text using MedGemma model.
    
    Args:
        medical_text: String containing medical report or clinical notes
        
    Returns:
        String with extracted clinical variables and their values
    """
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # CRITICAL: Use float32, not float16!
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Create extraction prompt
    prompt = f"""Extract all clinical variables and their values from the following medical text.

Medical Report:
{medical_text}

List each clinical variable with its value and unit."""

    messages = [{"role": "user", "content": prompt}]

    # Apply chat template
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_special_tokens=True,
        add_generation_prompt=True,
    )

    # Tokenize
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=1024,
    )
    inputs = {k: v.cuda() for k, v in inputs.items()}

    # Generate extraction
    print("Extracting clinical variables...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,  # Enough for complete extraction
            do_sample=False,  # Deterministic for consistent extraction
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

    # Decode just the generated portion
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    extraction_result = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return extraction_result


def main():
    # Example medical text
    medical_report = """Patient presents with fever of 102°F and cough for 3 days. Blood pressure is 120/80 mmHg. 
Heart rate is 88 bpm. Patient reports headache and fatigue. Laboratory results show 
WBC count of 12,000 cells/μL, hemoglobin 14.2 g/dL, glucose 95 mg/dL."""

    print("="*60)
    print("MEDICAL TEXT:")
    print(medical_report)
    print("="*60)

    # Extract variables
    result = extract_clinical_variables(medical_report)

    print("\n" + "="*60)
    print("EXTRACTED CLINICAL VARIABLES:")
    print(result)
    print("="*60)


if __name__ == "__main__":
    main()