# check_medgemma_format.py
from transformers import AutoTokenizer

model_name = "google/medgemma-4b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Check if there's a chat template
if hasattr(tokenizer, 'chat_template'):
    print("Chat template found:")
    print(tokenizer.chat_template)

# Check special tokens
print("\nSpecial tokens:")
print(f"BOS: {tokenizer.bos_token}")
print(f"EOS: {tokenizer.eos_token}")
print(f"PAD: {tokenizer.pad_token}")
print(f"UNK: {tokenizer.unk_token}")

# Check if there are any special instruction tokens
special_tokens = tokenizer.special_tokens_map
print("\nAll special tokens:")
for key, value in special_tokens.items():
    print(f"{key}: {value}")
