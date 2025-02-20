import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests
   


def get_device_map() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device_map()

print("I am running on: " + device)
processor = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-O-0924',
    trust_remote_code=True,
    use_fast=False,
    torch_dtype='auto',
    #device_map = device
    device_map = "cpu"
    #device_map = "cuda" if torch.cuda.is_available() else device_map == "cpu"
)

model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-O-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    low_cpu_mem_usage = True,
    #device_map = device
    device_map = "cpu"
    #device_map = "cuda" if torch.cuda.is_available() else device_map == "cpu"
)
inputs = processor.process(
    images = [Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)],
    text="what animal."
)
inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

output = model.generate_from_batch(
    inputs,
    GenerationConfig(max_new_tokens=500, stop_strings="<|endoftext|>"),
    tokenizer=processor.tokenizer
)
generated_tokens = output[0,inputs['input_ids'].size(1):]
generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(generated_text)