import copy
import peft
import torch
from typing import Any

class MoELoRA(peft.tuners.lora.layer.Linear):
    
    def set_moe(self, number_experts: int, top_k: int):

        assert top_k <= number_experts
        self.number_experts = number_experts
        self.top_k = top_k
        self.descent_strategy = None

        self.lora_A = torch.nn.ModuleList([
            copy.deepcopy(self.lora_A.default) 
            for i in range(self.number_experts)
        ])
        self.lora_B = torch.nn.ModuleList([
            copy.deepcopy(self.lora_B.default) 
            for i in range(self.number_experts)
        ])
        self.gate = torch.nn.Linear(
            self.lora_A[0].weight.size(1), 
            self.number_experts, 
            bias = False,
            dtype = self.lora_A[0].weight.dtype
        )

        old_adapters = len(self.active_adapters)
        for i in range(self.number_experts):
            self.active_adapters.append(i)
        for i in range(old_adapters):
            self.active_adapters.pop(0)
        self.use_dora = {
            i: self.use_dora['default'] 
            for i in range(self.number_experts)
        }
    
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:

        assert self.descent_strategy in {"riemannian", "moe-riemannian"}
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters: 
        # if not using LoRA.
            if self.merged:
                raise NotImplementedError(
                    "MoE-LoRA shouldn't be merged. Some errors may happen."
                )
            result = self.base_layer(x, *args, **kwargs)

        elif adapter_names is not None: 
        # if using mixed LoRA.
            raise NotImplementedError(
                "Should run 'result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)'" + 
                "but mixed_batch_forward() hasn't been implemented for MoE yet."
            )
        
        elif self.merged: 
        # if LoRA is already merged into base model.
            raise NotImplementedError(
                "MoE-LoRA shouldn't be merged now. Some errors may happen."
            )
        
        else: 
        # if using normal MoE-LoRA forwarding.
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            # compute gate:
            g = self.gate(x).to(torch_result_dtype)
            g = torch.nn.functional.softmax(g, dim = -1)
            _, selected_expert_ids = torch.topk(g, self.top_k, dim = -1)
            one_hot_selected_ids = torch.nn.functional.one_hot(
                selected_expert_ids, 
                num_classes = self.number_experts
            ).sum(dim = 2)
            g = g * one_hot_selected_ids
            g = g / g.sum(dim = -1, keepdim = True)
            #new_g = torch.zeros_like(g).to(result.device).to(torch_result_dtype)
            #new_g[one_hot_selected_ids == 1] = g[one_hot_selected_ids == 1]
            #g = new_g / new_g.sum(dim = -1, keepdim = True)

            # run experts:
            for active_adapter in self.active_adapters:
                if active_adapter >= len(self.lora_A):
                    # if Expert ID overflow the total number of experts. should raise an error.
                    raise RuntimeError
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout['default']  #dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling['default']       #scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)
                if not self.use_dora[active_adapter]:
                    expert_output = lora_B(lora_A(dropout(x))) * scaling
                    gate_value = g[:, :, active_adapter].unsqueeze(-1)
                    if self.descent_strategy == "moe-riemannian":
                        sqrt_gate_value_const = (gate_value**0.5).clone().detach()
                        expert_output_const = expert_output.clone().detach()
                        #sqrt_gate_value = (gate_value + 1e-10)**0.5
                        #gate_value_const = gate_value.clone().detach()
                        #weighted_expert_output = (gate_value/(sqrt_gate_value_const + 1e-9))*expert_output + (gate_value_const - sqrt_gate_value_const)*expert_output_const
                        weighted_expert_output = sqrt_gate_value_const * expert_output + (gate_value - sqrt_gate_value_const) * expert_output_const
                    else:
                        weighted_expert_output = gate_value * expert_output
                    result = result + weighted_expert_output
                else:
                    if isinstance(dropout, torch.nn.Identity) or not self.training:
                        base_result = result
                    else:
                        x = dropout(x)
                        base_result = None
                    expert_output = self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                        base_result=base_result,
                    )
                    gate_value = g[:, :, active_adapter].unsqueeze(-1)
                    if self.descent_strategy == "moe-riemannian":
                        sqrt_gate_value_const = (gate_value**0.5).clone().detach()
                        expert_output_const = expert_output.clone().detach()
                        #sqrt_gate_value = (gate_value + 1e-10)**0.5
                        #gate_value_const = gate_value.clone().detach()
                        #weighted_expert_output = (gate_value/(sqrt_gate_value_const + 1e-9))*expert_output + (gate_value_const - sqrt_gate_value_const)*expert_output_const
                        weighted_expert_output = sqrt_gate_value_const * expert_output + (gate_value - sqrt_gate_value_const) * expert_output_const
                    else:
                        weighted_expert_output = gate_value * expert_output    
                    result = result + weighted_expert_output
            result = result.to(torch_result_dtype)
        return result
