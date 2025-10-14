"""
Fixed multimodal smoke test for MedGemma (google/medgemma-4b-it)

Key fix: Added <start_of_image> token to prompt as required by Gemma3Processor
"""

import os
import pytest
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "google/medgemma-4b-it"


def _pick_dtype():
    """Pick appropriate dtype based on available hardware"""
    if torch.cuda.is_available():
        # Use float32 for stability (based on previous debugging)
        return torch.float32
    return torch.float32


@pytest.fixture(scope="session")
def hf_access_token():
    """Get HF token from environment variable"""
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        pytest.skip(
            "HF token not found. Set env var HF_TOKEN (or HUGGINGFACE_TOKEN) to run this test."
        )
    return token


@pytest.fixture(scope="session")
def processor_and_model(hf_access_token):
    """Load processor and model once per test session"""
    torch_dtype = _pick_dtype()

    print(f"\nLoading model with dtype={torch_dtype}...")
    
    # Load processor - use 'token' instead of deprecated 'use_auth_token'
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, 
        token=hf_access_token, 
        trust_remote_code=True
    )
    
    # Load model
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        token=hf_access_token,
        torch_dtype=torch_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    
    print(f"Model loaded on device: {next(model.parameters()).device}")
    return processor, model


def _make_dummy_image(size=(224, 224), color=(200, 200, 200)):
    """Create a simple gray square with a darker cross"""
    img = Image.new("RGB", size, color)
    px = img.load()
    mid_x = size[0] // 2
    mid_y = size[1] // 2
    # Draw horizontal line
    for x in range(size[0]):
        px[x, mid_y] = (80, 80, 80)
    # Draw vertical line
    for y in range(size[1]):
        px[mid_x, y] = (80, 80, 80)
    return img


def test_multimodal_smoke(processor_and_model):
    """Test that the model can process images and text together"""
    processor, model = processor_and_model

    # Create dummy image
    image = _make_dummy_image()
    
    # CRITICAL FIX: Include <start_of_image> token in the prompt!
    # The Gemma3Processor expects this token to know where to insert the image
    prompt = (
        "<start_of_image>You see a simple synthetic medical diagram: "
        "a cross on a square background. In one short sentence, state what you see."
    )

    print(f"\nPrompt: {prompt}")
    print(f"Image: {image.size} {image.mode}")
    
    # Process inputs
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    print(f"Input keys: {inputs.keys()}")
    print(f"Input shapes: {[(k, v.shape) for k, v in inputs.items()]}")

    # Generate response
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100,
            do_sample=False,  # Deterministic for testing
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    # Decode output
    if hasattr(processor, "tokenizer"):
        text = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    else:
        text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    print(f"\nModel output: {text}")

    # --- Assertions ---
    assert isinstance(text, str), "Model output should decode to a string."
    assert len(text.strip()) >= 5, f"Unexpectedly short output: {repr(text)}"
    
    # Check that output doesn't contain unprocessed placeholder tokens
    # (It's OK to have the token in input, but not in output)
    assert not text.endswith("<start_of_image>"), "Output shouldn't end with image token"
    
    print("\n✅ Multimodal smoke test passed!")