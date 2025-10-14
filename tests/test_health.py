# check_model_health.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def check_model_health():
    model_name = "google/medgemma-4b-it"
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # drop float16 for now
        torch_dtype=torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    
    print("\nModel info:")
    print(f"Model type: {type(model)}")
    print(f"Model config: {model.config.model_type}")
    
    cfg = model.config.to_dict()
    print(f"Vocab size: {cfg.get('vocab_size', tokenizer.vocab_size)}")
    
    # Gemma3 doesn’t expose hidden_size in the config,
    # but you can grab the embedding dimension directly:
    embed_dim = model.get_input_embeddings().weight.shape[1]
    print(f"Embedding dim: {embed_dim}")
    
    # Check if model outputs make sense
    print("\nTesting model forward pass...")
    
    # Create a simple input
    test_input = "Hello"
    inputs = tokenizer(test_input, return_tensors="pt")
    input_ids = inputs['input_ids'].cuda()
    
    print(f"Input IDs: {input_ids}")
    print(f"Input tokens: {tokenizer.convert_ids_to_tokens(input_ids[0].tolist())}")
    
    # Get model output
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
        
    print(f"Output shape: {logits.shape}")
    print(f"Output dtype: {logits.dtype}")
    
    # Check if logits are reasonable
    last_token_logits = logits[0, -1, :]
    print(f"Min logit: {last_token_logits.min().item()}")
    print(f"Max logit: {last_token_logits.max().item()}")
    print(f"Mean logit: {last_token_logits.mean().item()}")
    
    # Get top 5 predicted tokens
    top5 = torch.topk(last_token_logits, 5)
    print(f"\nTop 5 predicted next tokens:")
    for i, (value, idx) in enumerate(zip(top5.values, top5.indices)):
        token = tokenizer.decode([idx.item()])
        print(f"  {i+1}. Token ID {idx.item()}: '{token}' (logit: {value.item():.2f})")
    
    # Check if the model is in eval mode
    print(f"\nModel training mode: {model.training}")
    
    # Try a minimal generation
    print("\n" + "="*60)
    print("Testing minimal generation without special tokens...")
    
    simple_input = tokenizer.encode("The", return_tensors="pt").cuda()
    print(f"Input: 'The' -> IDs: {simple_input}")
    
    try:
        with torch.no_grad():
            output = model.generate(
                simple_input,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        print(f"Generated IDs: {output}")
        print(f"Decoded: '{tokenizer.decode(output[0])}'")
    except Exception as e:
        print(f"Generation failed: {e}")

if __name__ == "__main__":
    check_model_health()
