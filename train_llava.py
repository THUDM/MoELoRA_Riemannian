import os
import sys
import peft
import torch
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import get_linear_schedule_with_warmup, BitsAndBytesConfig
from transformers import AutoProcessor, LlavaForConditionalGeneration

from moe_lora import MoELoRA
from utils import plot_gradients, plot_values
from custom_optimizer import AdamW, SGD, AdamWr, SGDr
from utils import init_lora_A, init_lora_B, init_gate, wrap_print_function

from dataset import CoLADataset
from dataset import ScienceQADataset
from dataset import BoolQDataset
from dataset import CommonsenseQADataset
from dataset import SST2Dataset
from dataset import MRPCDataset
from dataset import QQPDataset
from dataset import QNLIDataset
from dataset import STSBDataset
from dataset import WNLIDataset
from dataset import MixedQADataset
from dataset import ImageText2TextDataset
from dataset import VMCBenchDataset

MyDataset = None
DATASET_PATH = None
DATASET_NAME = None
CANDIDATE_DATASETS = {
    "ScienceQA":        [ScienceQADataset,      "../datasets/scienceqa/science_qa.hf"],
    "CommonsenseQA":    [CommonsenseQADataset,  "../datasets/CommonsenseQA"],

    "CoLA":             [CoLADataset,           "../datasets/glue_data/CoLA"],
    "SST-2":            [SST2Dataset,           "../datasets/glue_data/SST-2"],
    "MRPC":             [MRPCDataset,           "../datasets/glue_data/MRPC"],
    "QQP":              [QQPDataset,            "../datasets/glue_data/QQP"],
    "QNLI":             [QNLIDataset,           "../datasets/glue_data/QNLI"],
    "STS-B":            [STSBDataset,           "../datasets/glue_data/STS-B"],
    "WNLI":             [WNLIDataset,           "../datasets/glue_data/WNLI"],

    "MixedQA":          [MixedQADataset,        "../datasets"],

    "ImageText2Text":   [ImageText2TextDataset, "./datasets/visual7w"],
    "VMCBench":         [VMCBenchDataset,       "./datasets/VMCBench/data"]
}

print_log = None

def load_model(pretrained_model_path, lora_r, lora_alpha, lora_dropout, lora_modules, number_experts, top_k, torch_type, descent_strategy):
    
    #tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
    #tokenizer.pad_token = tokenizer.eos_token
    #tokenizer.pad_token_id = tokenizer.eos_token_id
    #tokenizer.padding_side='left'
    processor = AutoProcessor.from_pretrained(pretrained_model_path)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit = True,                        # 或者 load_in_8bit=True，根据需要设置
        #llm_int8_threshold = 6.0,
        #llm_int8_has_fp16_weight = False,
        llm_int8_enable_fp32_cpu_offload = True,
        bnb_4bit_compute_dtype = torch.float16,     # 计算时使用的精度
        bnb_4bit_quant_type = "nf4",                # 采用nf4量化法，默认为fp4
        bnb_4bit_use_double_quant = True,           # 是否采用双量化
    )
    model = LlavaForConditionalGeneration.from_pretrained( #AutoModelForCausalLM.from_pretrained(
        pretrained_model_path, 
        torch_dtype = torch_type, 
        low_cpu_mem_usage = True,
        quantization_config = quantization_config
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r = lora_r,
        lora_alpha = lora_alpha,
        target_modules = lora_modules,
        lora_dropout = lora_dropout,
        bias = "none",
        task_type = "CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    for name, module in model.named_modules():
        if isinstance(module, peft.tuners.lora.LoraLayer):
            module.__class__ = MoELoRA
            module.set_moe(number_experts = number_experts, top_k = top_k)
            module.descent_strategy = descent_strategy
            for i in range(number_experts):
                init_lora_A(module.lora_A[i])
                init_lora_B(module.lora_B[i])
                init_gate(module.gate)
            print_log(name)
    global AdamWr, SGDr
    AdamWr.number_experts = number_experts
    SGDr.number_experts = number_experts
    return processor, model

def distribution_initialize():
    local_rank = os.environ['LOCAL_RANK']
    torch.distributed.init_process_group(backend = 'nccl')
    torch.distributed.barrier()
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    device = torch.device("cuda", rank)
    return world_size, local_rank, device

def valid(gradient_update_steps, epoch, val_loader, model):
    with torch.no_grad():
        valid_loss = []
        tqdm_val_loader = tqdm(val_loader) if torch.distributed.get_rank() == 0 else val_loader
        for batch in tqdm_val_loader:
            outputs = model(**batch)
            loss = outputs.loss.item()
            valid_loss.append(loss)
        avg_valid_loss = sum(valid_loss)/len(valid_loss)
        print_log(f"Steps:{gradient_update_steps}; Epoch:{epoch}; Rank:{torch.distributed.get_rank()}; Validation Loss:{avg_valid_loss}\n", end = "")

def test(gradient_update_steps, epoch, test_loader, model, tokenizer, device, max_window_length, max_generated_tokens):
    global MyDataset
    with torch.no_grad():
        test_outputs, test_inputs, test_labels = [], [], []
        tqdm_test_loader = tqdm(test_loader) if torch.distributed.get_rank() == 0 else test_loader
        for batch in tqdm_test_loader:
            outputs = model.module.generate(
                input_ids = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
                pixel_values = batch["pixel_values"].to(device),
                max_new_tokens = max_generated_tokens,
                #eos_token_id = tokenizer.eos_token_id,
                #pad_token_id = tokenizer.pad_token_id,
                do_sample = False,
                temperature = None,
                top_p = None
            )
            if outputs.size(1) < max_window_length + max_generated_tokens:
                format_outputs = torch.ones(outputs.size(0), max_window_length + max_generated_tokens, dtype = outputs.dtype).to(device) * 32001 #tokenizer.pad_token_id
                format_outputs[:, :outputs.size(1)] = outputs
                test_outputs.append(format_outputs)
            else:
                test_outputs.append(outputs)
            test_inputs.append(batch["input_ids"])
            test_labels.append(batch["labels"])
        outputs = torch.cat(test_outputs, dim = 0)
        inputs = torch.cat(test_inputs, dim = 0)
        labels = torch.cat(test_labels, dim = 0)
        metrics = MyDataset.evaluate(inputs = inputs, labels = labels, preds = outputs, tokenizer = tokenizer)
        print_log(f"Steps:{gradient_update_steps}; Epoch:{epoch}; Rank:{torch.distributed.get_rank()}; Accuracy:{metrics}\n", end = "")

def train(train_loader, val_loader, test_loader, model, per_gpu_batch_size, accumulated_steps, optimizer, scheduler, max_gate_grad_norm, total_gradient_update_steps, eval_steps, generate_param):
    gradient_update_steps = 0
    total_steps = 0
    max_epoch = int((total_gradient_update_steps * accumulated_steps) / len(train_loader)) + 1
    tokenizer = generate_param["tokenizer"]
    device = generate_param["device"]
    max_window_length = generate_param["max_window_length"]
    max_generated_tokens = generate_param["max_generated_tokens"]
    num_bit = len(str(per_gpu_batch_size*len(train_loader)))
    fmt = '%'+str(num_bit)+'d'
    model.train()
    optimizer["expert"].zero_grad()
    optimizer["gate"].zero_grad()
    loss_item = 0.0
    start_time = datetime.now()
    for epoch in range(max_epoch):
        for batch in train_loader:
            if gradient_update_steps == total_gradient_update_steps:
                return model
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / accumulated_steps
            loss.backward()
            loss_item = loss_item + loss.item()
            total_steps += 1
            if total_steps % accumulated_steps == 0:
                if max_gate_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_([param for name, param in model.named_parameters() if name.find('gate.') != -1], max_norm = max_gate_grad_norm)
                optimizer["expert"].step()
                optimizer["gate"].step()
                scheduler["expert"].step()
                scheduler["gate"].step()
                optimizer["expert"].zero_grad()
                optimizer["gate"].zero_grad()
                gradient_update_steps += 1
                time_str = str(datetime.now() - start_time).split(".")[0]
                print_log(f"[{time_str}] Rank:{torch.distributed.get_rank()}; Step:{'%4d'%gradient_update_steps}/{total_gradient_update_steps}; Epoch:{'%2d'%epoch}({fmt%(per_gpu_batch_size*(total_steps % len(train_loader)))}/{per_gpu_batch_size*len(train_loader)}); Loss:{loss_item};\tLR:{[g['lr'] for part in optimizer for g in optimizer[part].param_groups]}\n", end = "")
                loss_item = 0.0
                if gradient_update_steps % eval_steps == 0:
                    torch.distributed.barrier()
                    model.eval()
                    valid(gradient_update_steps, epoch, val_loader, model)
                    torch.distributed.barrier()
                    test(gradient_update_steps, epoch, test_loader, model, tokenizer, device, max_window_length, max_generated_tokens)
                    torch.distributed.barrier()
                    model.train()
                    start_time = datetime.now()


def main(optimize_strategy, descent_strategy):

    # distribution configuration
    world_size, local_rank, device = distribution_initialize()
    global MyDataset
    global DATASET_NAME
    global print_log
    save_path = "./saved_lora/" + "_".join([
        DATASET_NAME,
        optimize_strategy,
        descent_strategy,
        datetime.now().strftime("%y%m%d.%H.%M")
    ])
    os.mkdir(save_path)
    torch_rank = torch.distributed.get_rank()
    print_log = wrap_print_function(file_path = save_path + "/log."+str(torch_rank))
    print_log(f">> world size:{world_size}; torch rank:{torch_rank}; OS local rank:{local_rank}; device:{device};\n", end = "")

    candidate_configs = {

        "sgd": {
            "number_expert": 20,        # 18
            "expert_rank": 4,           # 4
            "top_k": 10,                # 9
            "initial_expert_lr": 3e-5,  #3e-5,
            "initial_gate_lr": 3e-8,
            "warm_up": 0,
            "total_gradient_update_steps": 2000,    # 500   #"training_epochs": 20,
            "eval_steps": 100,                      # 100
            "per_gpu_batch_size": {                 # SGD takes less memory, so we use large batch.
                512+10: 10,
                256+10: 20,                         # For those dataset max_window_length=256 & max_generated_token=10 & 3090GPU
                128+10: 40,                         # For those dataset max_window_length=128 & max_generated_token=10 & 3090GPU
                64+10:  80,                          # For those dataset max_window_length=64 & max_generated_token=10 & 3090GPU
                
                140+10: 5,
                400+10: 2
            },
            "accumulated_steps": {                  # total_steps = acculumated_steps * total_gradient_update_steps
                512+10: 8,
                256+10: 4,
                128+10: 2,
                64+10:  1,
                
                140+10: 8,
                400+10: 2 
            },                                      # logic batch size = 80
            "scheduler_max_steps": int(2000/9*10),  # so that the learning rate will decrease to 0.1*Initial_LR
            "max_gate_grad_norm": 1.0
        },

        "adamw": {
            "number_expert": 20,        #50         # For AdamW, we only managed improves when training lots of experts.
            "expert_rank": 4,           #4
            "top_k": 10,                #25
            "initial_expert_lr": 1e-5,  #3e-5,
            "initial_gate_lr": 3e-8,
            "warm_up": 0,
            "total_gradient_update_steps": 2000,    # 250   #"training_epochs": 5,
            "eval_steps": 100,                      # 50
            "per_gpu_batch_size": {                 # AdamW takes more memory, so we use small batch.
                512+10: 5,
                256+10: 10,                          # For those dataset max_window_length=256 & max_generated_token=10 & 3090GPU
                128+10: 20,                         # For those dataset max_window_length=128 & max_generated_token=10 & 3090GPU
                64+10:  40,                          # For those dataset max_window_length=64 & max_generated_token=10 & 3090GPU

                140+10: 2,
                400+10: 1
            },
            "accumulated_steps": {                  # total_steps = acculumated_steps * total_gradient_update_steps
                512+10: 16,
                256+10: 8,
                128+10: 4,
                64+10:  2,

                140+10: 20,
                400+10: 4
            },                                      # logic batch size = 80
            "scheduler_max_steps": int(2000/9*10),  # so that the learning rate will decrease to 0.1*Initial_LR
            "max_gate_grad_norm": 1.0
        }

    }

    # model configuration
    tokenizer, model = load_model(
        pretrained_model_path = './llava/llava-1.5-7b-hf',
        lora_r = candidate_configs[optimize_strategy]["expert_rank"],
        lora_alpha = 16,
        lora_dropout = 0.05,
        lora_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'out_proj'],
        number_experts = candidate_configs[optimize_strategy]["number_expert"],
        top_k = candidate_configs[optimize_strategy]["top_k"],
        torch_type = torch.float16,
        descent_strategy = descent_strategy
    )
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model, 
        device_ids = [str(device)],
        find_unused_parameters = True,
        static_graph = True
    )

    # training configuration
    max_window_length = MyDataset.MAX_SAMPLE_INPUT_LENGTH           # 128/256/...
    max_generated_tokens = MyDataset.MAX_SAMPLE_OUTPUT_LENGTH       # 10/...
    test_before_training = True                                    # True/False
    eval_before_training = True                                    # True/False
    initial_expert_learning_rate = candidate_configs[optimize_strategy]["initial_expert_lr"]
    initial_gate_learning_rate = candidate_configs[optimize_strategy]["initial_gate_lr"]                                  
    warmup_steps = candidate_configs[optimize_strategy]["warm_up"]
    total_gradient_update_steps = candidate_configs[optimize_strategy]["total_gradient_update_steps"]
    per_gpu_batch_size = candidate_configs[optimize_strategy]["per_gpu_batch_size"][max_window_length+max_generated_tokens]
    accumulated_steps = candidate_configs[optimize_strategy]["accumulated_steps"][max_window_length+max_generated_tokens]
    scheduler_max_steps = candidate_configs[optimize_strategy]["scheduler_max_steps"]
    eval_steps = candidate_configs[optimize_strategy]["eval_steps"]
    max_gate_grad_norm = candidate_configs[optimize_strategy]["max_gate_grad_norm"]

    world_batch_size = per_gpu_batch_size * world_size

    # dataset configuration
    global DATASET_PATH
    train_dataset, val_dataset, test_dataset = MyDataset.load_dataset(
        data_path = DATASET_PATH,
        processor = tokenizer,
        max_length = max_window_length
    )
    train_loader, val_loader, test_loader = (
        DataLoader(
            dataset_split,
            batch_size = per_gpu_batch_size,
            sampler = DistributedSampler(
                dataset_split,
                num_replicas = world_size,
                rank = torch.distributed.get_rank()
            )
        ) for dataset_split in [train_dataset, val_dataset, test_dataset]
    )

    # optimizer configuration
    expert_parameters = [param for name, param in model.named_parameters() if name.find('lora_') != -1]
    gate_parameters = [param for name, param in model.named_parameters() if name.find('gate.') != -1]
    optimizer_expert = ({
        "adamw|riemannian": AdamWr,
        "sgd|riemannian": SGDr,
        "adamw|moe-riemannian": AdamWr,
        "sgd|moe-riemannian": SGDr,
    }[f"{optimize_strategy}|{descent_strategy}"])(expert_parameters, lr = initial_expert_learning_rate)
    
    optimizer_gate = ({"sgd": SGD, "adamw": AdamW})[optimize_strategy](gate_parameters, lr = initial_gate_learning_rate)
    #optimizer_gate = SGD(gate_parameters, lr = initial_gate_learning_rate)
    
    scheduler_expert = get_linear_schedule_with_warmup(
        optimizer_expert, 
        num_warmup_steps = warmup_steps, 
        num_training_steps = scheduler_max_steps
    )
    scheduler_gate = get_linear_schedule_with_warmup(
        optimizer_gate, 
        num_warmup_steps = warmup_steps, 
        num_training_steps = scheduler_max_steps
    )
    optimizer = {"gate": optimizer_gate, "expert": optimizer_expert}
    scheduler = {"gate": scheduler_gate, "expert": scheduler_expert}

    model.eval()
    if eval_before_training:
        valid(0, -1, val_loader, model)
        torch.distributed.barrier()
    if test_before_training:
        test(0, -1, test_loader, model, tokenizer, device, max_window_length, max_generated_tokens)
        torch.distributed.barrier()

    #_ = input("Wait to stop ...")
    #exit()

    model = train(
        train_loader = train_loader, 
        val_loader = val_loader, 
        test_loader = test_loader, 
        model = model, 
        per_gpu_batch_size = per_gpu_batch_size,
        accumulated_steps = accumulated_steps, 
        optimizer = optimizer, 
        scheduler = scheduler,
        max_gate_grad_norm = max_gate_grad_norm, 
        total_gradient_update_steps = total_gradient_update_steps, 
        eval_steps = eval_steps,
        generate_param = {
            "tokenizer": tokenizer, 
            "device": device, 
            "max_window_length": max_window_length, 
            "max_generated_tokens": max_generated_tokens
        }
    )


if __name__ == "__main__":

    assert len(sys.argv) == 5

    assert sys.argv[2] in CANDIDATE_DATASETS
    DATASET_NAME = sys.argv[2]
    MyDataset = CANDIDATE_DATASETS[DATASET_NAME][0]
    DATASET_PATH = CANDIDATE_DATASETS[DATASET_NAME][1]

    assert sys.argv[3] in {"sgd", "adamw"}
    optimize_strategy = sys.argv[3]

    assert sys.argv[4] in {"riemannian", "ourmethod"}
    descent_strategy = sys.argv[4] if (sys.argv[4] != "ourmethod") else "moe-riemannian"

    main(optimize_strategy, descent_strategy)
