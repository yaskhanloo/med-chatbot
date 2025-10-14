# cuda_debug.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Enable CUDA debugging
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"PyTorch version: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

# Try a simple tensor operation
try:
    test_tensor = torch.randn(100, 100).cuda()
    result = torch.softmax(test_tensor, dim=-1)
    print("Basic CUDA operations work ✓")
except Exception as e:
    print(f"CUDA error in basic operations: {e}")

# Test with smaller model first
print("\nTesting with smaller tensors...")
model_name = "google/medgemma-4b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Create minimal test
test_ids = torch.tensor([[1, 2, 3, 4, 5]]).cuda()
print(f"Test tensor on CUDA: {test_ids.device}")
