import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "google/medgemma-4b-it"

def debug_generation():
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Tokenizer info
    print("Tokenizer settings:")
    print(f"EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    print(f"PAD token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    print(f"BOS token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")

    # Setup prompt
    test_prompt = "What is fever?"
    messages = [{"role": "user", "content": test_prompt}]

    print("\n" + "="*60)
    print("Method 1: Chat template with forced generation")
    # Apply chat template
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_special_tokens=True,
        add_generation_prompt=True,
    )
    print("Formatted prompt:", repr(formatted))

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=1024,
    )
    inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print("Response:", response)

    print("\n" + "="*60)
    print("Method 2: Direct prompt without template")
    simple_prompt = f"<start_of_turn>user\n{test_prompt}<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(
        simple_prompt,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=1024,
        add_special_tokens=True,
    )
    inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print("Response:", response)

    print("\n" + "="*60)
    print("Method 3: With sampling parameters")
    chat_formatted = formatted
    inputs = tokenizer(
        chat_formatted,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=1024,
        add_special_tokens=False,
    )
    inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print("Response:", response)

if __name__ == "__main__":
    debug_generation()

