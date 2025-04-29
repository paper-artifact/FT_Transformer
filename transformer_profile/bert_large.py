import torch
import time
from transformers import BertModel, BertTokenizer
from transformers.models.bert.modeling_bert import BertSelfAttention
from collections import defaultdict


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

def bert_large_profile(text):
    print(f"\n### BERT-Large Inference Profiling Start ###\n")
    
    model = BertModel.from_pretrained("bert-large-uncased") 
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # text = "This is a test sentence for BERT-Large timing analysis " * 10  # modify the text according to the input scale you want to test.
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)

    hooks = []
    attention_times = defaultdict(float)
    start_events = {}

    for name, module in model.named_modules():
        if isinstance(module, BertSelfAttention):
            print(f"Register a hook to: {name}")
            pre_handle = module.register_forward_pre_hook(pre_hook(name, start_events))
            post_handle = module.register_forward_hook(post_hook(name, start_events, attention_times))
            hooks.extend([pre_handle, post_handle])

    with torch.no_grad():
        try:
            model(**inputs)
        except RuntimeError as e:
            print("Out of memory, please try a smaller batch size or sequence length")
            raise

    attention_times.clear()
    start_events.clear()

    num_runs = 50  
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

    print(f"\nAverage execution time per inference: {(avg_total_time*1000):.4f}ms")
    print(f"Total execution time for all attention layers: {(sum_attention*1000):.4f}ms")
    print(f"Proportion of time spent on attention computation: {percentage:.1f}%")

    print("\nAverage execution time for each attention layer:")
    for layer in range(24):  
        name = f"encoder.layer.{layer}.attention.self"
        avg_time = attention_times.get(name, 0) / num_runs
        print(f"Layer {layer:02d}: {(avg_time*1000):.6f}ms")

    for hook in hooks:
        hook.remove()
    
    print("\n### BERT-Large Inference Profiling Completed ###\n")