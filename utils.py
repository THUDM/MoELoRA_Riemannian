import torch
import numpy as np

def plot_values(model):
    print("Model Weights:")
    names, values = [], []
    for name, param in model.named_parameters():
        names.append(name)
        values.append(param.detach().cpu().numpy() if param is not None else [0])
    for i in range(len(names)):
        min_value = np.min(values[i])
        max_value = np.max(values[i])
        if np.isnan(max_value) or np.isnan(min_value):
            print(f"Rank:{torch.distributed.get_rank()}; {names[i]}; Values has NaN.")

def plot_gradients(model):
    print("Model Gradients:")
    names, grads = [], []
    for name, param in model.named_parameters():
        names.append(name)
        grads.append(param.grad.detach().cpu().numpy() if param.grad is not None else [0])
    for i in range(len(names)):
        min_value = np.min(grads[i])
        max_value = np.max(grads[i])
        if np.isnan(max_value) or np.isnan(min_value):
            print(f"Rank:{torch.distributed.get_rank()}; {names[i]}; Grads has NaN.")

def init_lora_A(lora_A_module):
    torch.nn.init.kaiming_uniform_(lora_A_module.weight, a=5**0.5)

def init_lora_B(lora_B_module):
    torch.nn.init.zeros_(lora_B_module.weight)

def init_gate(gate_module):
    # no need to init gate
    pass

def wrap_print_function(file_path: str):
    f = open(file_path, "w+")
    def print_log(log, end="\n"):
        f.write(str(log) + end)
        f.flush()
        print(log, end = end)
    return print_log