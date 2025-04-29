import torch
from transformers import GPT2Model, GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from collections import defaultdict
import time 


def pre_hook(name, start_events): 
    def hook(module, inputs):
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            start_events[name] = start_event
        else:
            start_events[name] = time.time()
    return hook

def post_hook(name, start_events, attention_times):
    def hook(module, inputs, outputs):
        if torch.cuda.is_available():
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            torch.cuda.synchronize()  
            start_event = start_events.pop(name, None)
            if start_event:
                elapsed = start_event.elapsed_time(end_event) / 1000  
                attention_times[name] += elapsed
        else:
            end_time = time.time()
            start_time = start_events.pop(name, end_time)
            elapsed = end_time - start_time
            attention_times[name] += elapsed
    return hook

def gpt2_profile(text):
    print("### GPT-2 Inference Profiling Start ###\n")
    model = GPT2Model.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()  

    # text = "This is a test input token for gpt2 model." # modify the text according to the input scale you want to test.
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

    hooks = []
    attention_times = defaultdict(float)
    start_events = {}

    for name, module in model.named_modules():
        if isinstance(module, GPT2Attention): 
            print(f"Register a hook to: {name}")
            pre_handle = module.register_forward_pre_hook(pre_hook(name, start_events))
            post_handle = module.register_forward_hook(post_hook(name, start_events, attention_times))
            hooks.extend([pre_handle, post_handle])

    with torch.no_grad():
        model(**inputs)

    attention_times.clear()
    start_events.clear()

    num_runs = 100
    total_time = 0.0

    for _ in range(num_runs):
        with torch.no_grad():
            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                model(**inputs)
                end_event.record()
                torch.cuda.synchronize()
                total_time += start_event.elapsed_time(end_event) / 1000  
            else:
                start_time = time.time()
                model(**inputs)
                total_time += time.time() - start_time

    avg_total_time = total_time / num_runs
    sum_attention = sum(attention_times.values()) / num_runs
    percentage = (sum_attention / avg_total_time) * 100

    print(f"\nAverage execution time per inference: {(avg_total_time*1000):.6f}ms")
    print(f"Total execution time for all attention layers: {(sum_attention*1000):.6f}ms")
    print(f"Proportion of time spent on attention computation: {percentage:.2f}%")

    print("\nAverage execution time for each attention layer:")
    for name, time_spent in attention_times.items():
        avg_time = time_spent / num_runs
        print(f"{name}: {(avg_time*1000):.6f}ms")

    for hook in hooks:
        hook.remove()
    
    print("\n### GPT-2 Inference Profiling Completed ###\n")
