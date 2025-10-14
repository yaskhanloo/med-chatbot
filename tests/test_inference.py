# test_inference.py
# Testing MedGemma with German medical prompts
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "google/medgemma-4b-it"

print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float32,  # ADDED: Explicit dtype for consistency
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# List of German test prompts
test_prompts = [
    # 1. General reasoning (differential diagnosis)
    "Patient kommt mit Fieber und starken Kopfschmerzen in die Notaufnahme. Welche möglichen Diagnosen könnten in Betracht gezogen werden?",

    # 2. Structured variable extraction
    "Analysiere den folgenden Text und extrahiere die wichtigsten klinischen Variablen: Diagnose, Symptome, Laborwerte, verabreichte Medikamente.\n\nText: Patient kommt mit Fieber und starken Kopfschmerzen in die Notaufnahme. Labor: CRP 125 mg/L, Leukozyten 14.000/µL. Medikamente: Paracetamol 1 g i.v.",

    # 3. Guideline recall
    "Welche aktuellen Leitlinien gelten in Deutschland für die Behandlung einer bakteriellen Meningitis bei Erwachsenen?",

    # 4. Summarization
    "Fasse den folgenden Arztbrief in drei kurzen Sätzen zusammen.\n\nText: Patientin, 67 Jahre, wird wegen akuter Verwirrtheit und Fieber aufgenommen. MRT zeigt multiple ischämische Läsionen im rechten MCA-Territorium. Blutkulturen positiv für Streptococcus pneumoniae. Behandlung mit Ceftriaxon begonnen.",

    # 5. Step-by-step clinical reasoning
    "Ein 45-jähriger Patient klagt über Brustschmerzen seit 2 Stunden, ausstrahlend in den linken Arm. Beschreibe die möglichen Ursachen und den empfohlenen diagnostischen Ablauf in der Notaufnahme."
]

print(f"\nRunning {len(test_prompts)} test cases...\n")

# Run all prompts
for i, prompt_text in enumerate(test_prompts, start=1):
    print(f"\n{'='*60}")
    print(f"TEST CASE {i}")
    print(f"{'='*60}")
    print(f"User (German):\n{prompt_text}\n")

    messages = [{"role": "user", "content": prompt_text}]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate with explicit parameters
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,  # Increased from 200 for more complete responses
            do_sample=False,  # Deterministic
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True
        )

    # Decode only the generated part
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print(f"Model (German response):\n{response}\n")
    print(f"Tokens generated: {len(generated_tokens)}")

print("\n" + "="*60)
print("All tests completed!")
print("="*60)