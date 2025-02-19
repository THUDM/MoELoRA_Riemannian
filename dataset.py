from transformers import AutoTokenizer
from torch.utils.data import Dataset
from datasets import load_from_disk
from io import BytesIO
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
import torch
import random
import csv
import json
import re
import os

class ImageText2TextDataset(Dataset):

    MAX_SAMPLE_INPUT_LENGTH = 140
    MAX_SAMPLE_OUTPUT_LENGTH =10
    # In + Out: 43-140

    @classmethod
    def get_response(cls, output: str):
        try:
            return output.split("\nASSISTANT:")[1].strip()
        except:
            print("Rank:", torch.distributed.get_rank(), "Found sample length out of max_window_size:", {"sample": output})
            exit()

    @classmethod
    def evaluate(cls, inputs: torch.tensor, labels: torch.tensor, preds: torch.tensor, tokenizer):
        assert preds.size(0) == labels.size(0)
        assert preds.size(0) == inputs.size(0)
        pairs = []
        for i in range(preds.size(0)):
            label_str = tokenizer.decode(labels[i], skip_special_tokens=True)
            pred_str = tokenizer.decode(preds[i], skip_special_tokens=True)
            pred_str = cls.get_response(pred_str)
            pairs.append((pred_str, label_str))
        count = 0
        for pair in pairs:
            pattern = re.compile(r'The answer is ([A-Z]).')
            res = pattern.findall(pair[0])
            if len(res) == 1:
                answer = res[0] # "A" / "B"
            else:
                answer = "FAILD"
            res = pattern.findall(pair[1])
            assert len(res) == 1
            label = res[0]      # "A" / "B"
            if str(answer) == str(label):
                count += 1
        return count / len(pairs)

    @classmethod
    def load_dataset(cls, data_path, processor, max_length):
        qa_path = os.path.join(data_path, "qa.json")
        image_folder = os.path.join(data_path, "images")
        f = open(qa_path)
        data = json.loads(f.read())
        f.close()
        train_data = []
        val_data = []
        test_data = []
        for item in tqdm(data['images']):
            image_path = os.path.join(image_folder, item['filename'])
            image = Image.open(image_path)
            for qa_pair in item['qa_pairs']:
                q = qa_pair['question']
                wrong_choices = qa_pair['multiple_choices']
                correct_choice = qa_pair['answer']
                candidates = wrong_choices + [correct_choice]
                random.shuffle(candidates)
                options = [chr(65+i) for i in range(len(candidates))]
                tmp = [i for i in range(len(candidates)) if candidates[i] == correct_choice]
                assert len(tmp) == 1
                correct_option = options[tmp[0]]
                prompt = q + "\n" + ("\n".join([options[i]+". "+candidates[i] for i in range(len(candidates))])) + "\n"
                prompt = "USER: <image>\n" + prompt + "\nASSISTANT:"
                response = "The answer is " + correct_option + "."
                sample = {"prompt": prompt, "response": response, "image": image}
                if item['split'] == "train":
                    train_data.append(sample)
                elif item['split'] == "val":
                    val_data.append(sample)
                elif item['split'] == 'test':
                    test_data.append(sample)
                else:
                    raise RuntimeError
        train_dataset = ImageText2TextDataset(data = train_data, processor = processor, max_length = max_length, dataset_type = "train")
        val_dataset = ImageText2TextDataset(data = val_data[:100], processor = processor, max_length = max_length, dataset_type = "validation")
        test_dataset = ImageText2TextDataset(data = test_data[:100], processor = processor, max_length = max_length, dataset_type = "generate")
        return train_dataset, val_dataset, test_dataset
    
    def __init__(self, data, processor, max_length, dataset_type):
        self.data = data
        self.processor = processor
        self.max_length = max_length
        self.dataset_type = dataset_type
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.dataset_type in {"train", "validation", "valid", "test"}:
            return self.get_input_connected_with_label(idx)
        elif self.dataset_type == "generate":
            return self.get_input_disconnected_with_label(idx)
        else:
            raise NotImplementedError

    def get_input_connected_with_label(self, idx):
        item = self.data[idx]
        # input_ids, attention_mask, pixel_values
        inputs = self.processor(
            text = item["prompt"] + item["response"], 
            images = item["image"], 
            return_tensors='pt'
        )
        prompts = self.processor(
            text = item["prompt"], 
            images = item["image"], 
            return_tensors='pt'
        )
        tokens = inputs["input_ids"].flatten()
        attmsk = inputs["attention_mask"].flatten()
        prompt_tokens = prompts["input_ids"].flatten()
        labels = torch.ones(tokens.size(), dtype = tokens.dtype)
        labels[:] = -100
        labels[prompt_tokens.size(0):] = tokens[prompt_tokens.size(0):]
        
        #global data_lens
        #data_lens.append(tokens.size(0))
        
        if tokens.size(0) > self.max_length:
            ret = {
                "input_ids": tokens[-self.max_length:],
                "attention_mask": attmsk[-self.max_length:],
                "labels": labels[-self.max_length:],
                "pixel_values": inputs["pixel_values"][0, :, :, :]
            }
        else:
            tmp = torch.ones(self.max_length, 1, dtype = tokens.dtype).flatten()
            pad_tokens = tmp*32001
            pad_attmsk = tmp*0
            pad_labels = tmp*(-100)
            pad_tokens[-tokens.size(0):] = tokens
            pad_attmsk[-attmsk.size(0):] = attmsk
            pad_labels[-labels.size(0):] = labels
            ret = {
                "input_ids": pad_tokens,
                "attention_mask": pad_attmsk,
                "labels": pad_labels,
                "pixel_values": inputs["pixel_values"][0, :, :, :]
            }
        return ret
    
    def get_input_disconnected_with_label(self, idx): 
        item = self.data[idx]
        prompts = self.processor(
            text = item["prompt"], 
            images = item["image"], 
            truncation = True,
            max_length = self.max_length,
            padding = 'max_length',
            return_tensors='pt'
        )
        if "response" in item:
            outputs = self.processor(
                text = item["response"],
                truncation = True,
                max_length = self.max_length,
                padding = 'max_length',
                return_tensors='pt'
            )
            return {
                "input_ids": prompts["input_ids"].flatten(),
                "attention_mask": prompts["attention_mask"].flatten(),
                "pixel_values": prompts["pixel_values"][0, :, :, :],
                "labels": outputs["input_ids"].flatten()
            }
        else:
            return {
                "input_ids": prompts["input_ids"].flatten(),
                "attention_mask": prompts["attention_mask"].flatten(),
                "pixel_values": prompts["pixel_values"][0, :, :, :]
            } 

class VMCBenchDataset(ImageText2TextDataset):

    MAX_SAMPLE_INPUT_LENGTH = 400
    MAX_SAMPLE_OUTPUT_LENGTH = 10

    @classmethod
    def load_dataset(cls, data_path, processor, max_length):
        file_path = os.path.join(data_path, 'dev-00000-of-00001.parquet')
        df = pd.read_parquet(file_path)
        data = []
        for i in range(len(df)):
            image = Image.open(BytesIO(df.loc[i]['image']['bytes']))
            q = df.loc[i]['question']
            options = '\n'.join([O+'. '+df.loc[i][O] for O in ['A','B','C','D']])
            prompt = q + "\n" + options + "\n"
            prompt = "USER: <image>\n" + prompt + "\nASSISTANT:"
            response = "The answer is " + df.loc[i]['answer'] + "."
            sample = {"prompt": prompt, "response": response, "image": image}
            data.append(sample)
        train_dataset = VMCBenchDataset(data = data[:900], processor = processor, max_length = max_length, dataset_type = "train")
        val_dataset = VMCBenchDataset(data = data[900:], processor = processor, max_length = max_length, dataset_type = "validation")
        test_dataset = VMCBenchDataset(data = data[900:], processor = processor, max_length = max_length, dataset_type = "generate")
        return train_dataset, val_dataset, test_dataset

class MultipleChoiceQADataset(Dataset):

    MAX_SAMPLE_INPUT_LENGTH = None
    MAX_SAMPLE_OUTPUT_LENGTH = None
    
    prompt_template = {
        "prompt_input": None,
        "prompt_no_input": None,
        "response_split": None,
        "response_template": None,
        "failed_str": None
    }

    @classmethod
    def get_response(cls, output: str):
        try:
            return output.split(cls.prompt_template["response_split"])[1].strip()
        except:
            print("Rank:", torch.distributed.get_rank(), "Found sample length out of max_window_size:", {"sample": output})
            exit()

    @classmethod
    def evaluate(cls, inputs: torch.tensor, labels: torch.tensor, preds: torch.tensor, tokenizer):
        assert preds.size(0) == labels.size(0)
        assert preds.size(0) == inputs.size(0)
        pairs = []
        for i in range(preds.size(0)):
            label_str = tokenizer.decode(labels[i], skip_special_tokens=True)
            pred_str = tokenizer.decode(preds[i], skip_special_tokens=True)
            pred_str = cls.get_response(pred_str)
            pairs.append((pred_str, label_str))
        count = 0
        for pair in pairs:
            pattern = re.compile(cls.prompt_template["response_template"])
            res = pattern.findall(pair[0])
            if len(res) == 1:
                answer = res[0] # "A" / "B"
            else:
                answer = cls.prompt_template["failed_str"]
            res = pattern.findall(pair[1])
            assert len(res) == 1
            label = res[0]      # "A" / "B"
            if str(answer) == str(label):
                count += 1
        return count / len(pairs)
    
    @classmethod
    def load_dataset(cls, data_path, tokenizer, max_length):
        raise NotImplementedError

    def __init__(self, data, tokenizer, max_length, dataset_type):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length 
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.dataset_type in {"train", "validation", "valid", "test"}:
            return self.get_input_connected_with_label(idx)
        elif self.dataset_type == "generate":
            return self.get_input_disconnected_with_label(idx)
        else:
            raise NotImplementedError
    
    def get_input_connected_with_label(self, idx):
        input_text = self.prompt_template["prompt_input"].format(
            instruction = self.data[idx]["instruction"], 
            input = self.data[idx]["input"]
        ) if (self.data[idx]["input"]) else self.prompt_template["prompt_no_input"].format(
            instruction = self.data[idx]["instruction"]
        )
        output_text = self.data[idx]["output"]
        text = input_text + output_text
        tokens = self.tokenizer(
            text, 
            truncation = True, 
            max_length = self.max_length, 
            padding = 'max_length', 
            return_tensors = "pt"
        )
        output_tokens = self.tokenizer(
            output_text, 
            truncation = True, 
            max_length = self.max_length, 
            padding = 'max_length', 
            return_tensors = "pt"
        )
        tokens["input_ids"] = tokens["input_ids"].flatten()
        tokens["attention_mask"] = tokens["attention_mask"].flatten()
        tokens["labels"] = output_tokens["input_ids"].flatten()
        tokens["labels"][tokens["labels"] == self.tokenizer.pad_token_id] = -100 # -100 means not counted in loss.
        return tokens
        
    def get_input_disconnected_with_label(self, idx): 
        input_text = self.prompt_template["prompt_input"].format(
            instruction = self.data[idx]["instruction"], 
            input = self.data[idx]["input"]
        ) if (self.data[idx]["input"]) else self.prompt_template["prompt_no_input"].format(
            instruction = self.data[idx]["instruction"]
        )
        text = input_text
        tokens = self.tokenizer(
            text, 
            truncation = True, 
            max_length = self.max_length, 
            padding = 'max_length', 
            return_tensors = "pt"
        )
        if "output" in self.data[idx]: # data should have a label if it is a testing sample; Or, it should be a inferring sample.
            output_text = self.data[idx]["output"]
            output_tokens = self.tokenizer(
                output_text, 
                truncation = True, 
                max_length = self.max_length, 
                padding = 'max_length', 
                return_tensors = "pt"
            )
            return {
                "input_ids": tokens["input_ids"].flatten(),
                "attention_mask": tokens["attention_mask"].flatten(),
                "labels": output_tokens["input_ids"].flatten()
            }
        else:
            return {
                "input_ids": tokens["input_ids"].flatten(), 
                "attention_mask": tokens["attention_mask"].flatten()
            }

class ScienceQADataset(MultipleChoiceQADataset):
    MAX_SAMPLE_INPUT_LENGTH = 256
    MAX_SAMPLE_OUTPUT_LENGTH = 10
    prompt_template = {
        "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
        "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
        "response_split": "### Response:",
        "response_template": r'The answer is ([A-Z]).',
        "failed_str": "FAILD"
    }
    @classmethod
    def load_dataset(cls, data_path, tokenizer, max_length):
        data = load_from_disk(data_path)
        train_dataset = ScienceQADataset(data = data["train"], tokenizer = tokenizer, max_length = max_length, dataset_type = "train")
        val_dataset = ScienceQADataset(data = data["validation"], tokenizer = tokenizer, max_length = max_length, dataset_type = "validation")
        test_dataset = ScienceQADataset(data = data["test"], tokenizer = tokenizer, max_length = max_length, dataset_type = "generate")
        return train_dataset, val_dataset, test_dataset

class BoolQDataset(MultipleChoiceQADataset):
    MAX_SAMPLE_INPUT_LENGTH = 256
    MAX_SAMPLE_OUTPUT_LENGTH = 10
    prompt_template = {
        "prompt_no_input": "Below is a passage followed by a question. Read the whole passage, and write a response to answer the question by Yes or No.\n\n### Passage And Question:\n\n{instruction}\n\n### Response:\n",
        "response_split": "### Response:",
        "response_template": r'The answer is ([YN][eo][s]?)',
        "failed_str": "FAILD"
    }
    @classmethod
    def get_response(cls, output: str): # override to ignore samples longer than MAX_SAMPLE_INPUT_LENGTH
        try:
            return output.split(cls.prompt_template["response_split"])[1].strip()
        except:
            return ""
    @classmethod
    def load_dataset(cls, data_path, tokenizer, max_length):
        train_file = open(os.path.join(data_path, "train.jsonl"))
        train_set = []
        for line in train_file:
            sample = json.loads(line)
            instruction = "Title:" + sample["title"] + "\n" + sample["passage"] + "\nQuestion:" + sample["question"] + "?"
            assert sample["answer"] in {True, False}
            output = "The answer is " + ("Yes" if sample["answer"] else "No")
            train_set.append({"instruction": instruction, "output": output, "input": ""})
        train_file.close()
        dev_file = open(os.path.join(data_path, "dev.jsonl"))
        dev_set = []
        for line in dev_file:
            sample = json.loads(line)
            instruction = "Title:" + sample["title"] + "\n" + sample["passage"] + "\nQuestion:" + sample["question"] + "?"
            assert sample["answer"] in {True, False}
            output = "The answer is " + ("Yes" if sample["answer"] else "No")
            dev_set.append({"instruction": instruction, "output": output, "input": ""})
        dev_file.close()
        train_dataset = BoolQDataset(data = train_set, tokenizer = tokenizer, max_length = max_length, dataset_type = "train")
        val_dataset = BoolQDataset(data = dev_set, tokenizer = tokenizer, max_length = max_length, dataset_type = "validation")
        test_dataset = BoolQDataset(data = dev_set, tokenizer = tokenizer, max_length = max_length, dataset_type = "generate")
        return train_dataset, val_dataset, test_dataset

class CommonsenseQADataset(MultipleChoiceQADataset):
    MAX_SAMPLE_INPUT_LENGTH = 256
    MAX_SAMPLE_OUTPUT_LENGTH = 10
    prompt_template = {
        "prompt_no_input": "Below is a question followed by several possible answers. Please choose the correct answer.\n\n### Question and Possible Answers:\n\n{instruction}\n\n### Correct Answer:\n",
        "response_split": "### Correct Answer:",
        "response_template": r'The correct answer is ([A-Z]).',
        "failed_str": "FAILD"
    }
    @classmethod
    def load_dataset(cls, data_path, tokenizer, max_length):
        train_file = open(os.path.join(data_path, "train_rand_split.jsonl"))
        train_set = []
        for line in train_file:
            sample = json.loads(line)
            instruction = sample["question"]["stem"] + "\n" + "\n".join([
                "("+sample["question"]["choices"][i]["label"].strip()+"). "+sample["question"]["choices"][i]["text"] 
                for i in range(len(sample["question"]["choices"]))])
            output = "The correct answer is " + sample["answerKey"] + "."
            train_set.append({"instruction": instruction, "output": output, "input": ""})
        train_file.close()
        dev_file = open(os.path.join(data_path, "dev_rand_split.jsonl"))
        dev_set = []
        for line in dev_file:
            sample = json.loads(line)
            instruction = sample["question"]["stem"] + "\n" + "\n".join([
                "("+sample["question"]["choices"][i]["label"].strip()+"). "+sample["question"]["choices"][i]["text"] 
                for i in range(len(sample["question"]["choices"]))])
            output = "The correct answer is " + sample["answerKey"] + "."
            dev_set.append({"instruction": instruction, "output": output, "input": ""})
        dev_file.close()
        train_dataset = CommonsenseQADataset(data = train_set, tokenizer = tokenizer, max_length = max_length, dataset_type = "train")
        val_dataset = CommonsenseQADataset(data = dev_set, tokenizer = tokenizer, max_length = max_length, dataset_type = "validation")
        test_dataset = CommonsenseQADataset(data = dev_set, tokenizer = tokenizer, max_length = max_length, dataset_type = "generate")
        return train_dataset, val_dataset, test_dataset

class CoLADataset(MultipleChoiceQADataset):
    MAX_SAMPLE_INPUT_LENGTH = 128
    MAX_SAMPLE_OUTPUT_LENGTH = 10
    prompt_template = {
        "prompt_no_input": "Below is a sentence. Please check whether the sentence is grammatically correct or wrong.\n\n### Sentence:\n\n{instruction}\n\n### Answer:\n",
        "response_split": "### Answer:",
        "response_template": r'The sentence is grammatically ([cw][or][ro][rn][eg][c]?[t]?).',
        "failed_str": "FAILD"
    }
    @classmethod
    def load_dataset(cls, data_path, tokenizer, max_length):
        train_file = open(os.path.join(data_path, "train.tsv"))
        dev_file = open(os.path.join(data_path, "dev.tsv"))
        res = []
        data_files = [train_file, dev_file]
        for i in range(2):
            data_file = data_files[i]
            data_reader = csv.reader(data_file, delimiter = "\t")
            subsets = {0:[], 1:[]}
            for line in data_reader:
                if sum([1 if (item.find("\t") != -1) else 0 for item in line]) > 0:
                    continue
                label = int(line[1])
                assert label in {0, 1}
                answer = "correct" if (label == 1) else "wrong"
                instruction = line[-1]
                output = "The sentence is grammatically " + answer + "."
                sample = {"instruction": instruction, "output": output, "input": ""}
                subsets[label].append(sample)
            data_file.close()
            if i == 1:
                data_split = subsets[0] + subsets[1]
                random.shuffle(data_split)
            else:
                assert len(subsets[1]) > len(subsets[0])
                p = len(subsets[1]) // len(subsets[0])
                set0 = subsets[0]*p
                set0 = set0 + random.sample(subsets[0], len(subsets[1]) - len(set0))
                data_split = subsets[1] + set0
                random.shuffle(data_split)
            res.append(data_split)
        train_dataset = CoLADataset(data = res[0], tokenizer = tokenizer, max_length = max_length, dataset_type = "train")
        val_dataset = CoLADataset(data = res[1], tokenizer = tokenizer, max_length = max_length, dataset_type = "validation")
        test_dataset = CoLADataset(data = res[1], tokenizer = tokenizer, max_length = max_length, dataset_type = "generate")
        return train_dataset, val_dataset, test_dataset
    @classmethod
    def evaluate(cls, inputs: torch.tensor, labels: torch.tensor, preds: torch.tensor, tokenizer):
        assert preds.size(0) == labels.size(0)
        assert preds.size(0) == inputs.size(0)
        pairs = []
        for i in range(preds.size(0)):
            label_str = tokenizer.decode(labels[i], skip_special_tokens=True)
            pred_str = tokenizer.decode(preds[i], skip_special_tokens=True)
            pred_str = cls.get_response(pred_str)
            pairs.append((pred_str, label_str))
        TP, FP, TN, FN, Failed = 0, 0, 0, 0, 0
        for pair in pairs:
            pattern = re.compile(cls.prompt_template["response_template"])
            res = pattern.findall(pair[0])
            if len(res) == 1:
                answer = res[0] # "correct" / "wrong"
            else:
                answer = cls.prompt_template["failed_str"]
            res = pattern.findall(pair[1])
            assert len(res) == 1
            label = res[0]      # "correct" / "wrong"
            if (str(label) == "correct") and (str(answer) == "correct"):
                TP += 1
            elif (str(label) == "correct") and (str(answer) == "wrong"):
                FN += 1
            elif (str(label) == "wrong") and (str(answer) == "wrong"):
                TN += 1
            elif (str(label) == "wrong") and (str(answer) == "correct"):
                FP += 1
            else:
                Failed += 1
        valid_rate = (len(pairs) - Failed) / len(pairs)
        try:
            mcc = (TP*TN - FP*FN)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5)
        except:
            assert TP*TN - FP*FN == 0
            mcc = 0.0
        return mcc*valid_rate

class SST2Dataset(MultipleChoiceQADataset):
    MAX_SAMPLE_INPUT_LENGTH = 128
    MAX_SAMPLE_OUTPUT_LENGTH = 10
    prompt_template = {
        "prompt_no_input": "Below is a short comment of a movie. Please judge whether the author has a positive or negative attitude of this movie.\n\n### Comment:\n\n{instruction}\n\n### Answer:\n",
        "response_split": "### Answer:",
        "response_template": r"The author has a ([pn][oe][sg][ia])tive attitude",
        "failed_str": "FAILD"
    }
    @classmethod
    def load_dataset(cls, data_path, tokenizer, max_length):
        train_file = open(os.path.join(data_path, "train.tsv"))
        dev_file = open(os.path.join(data_path, "dev.tsv"))
        res = []
        data_files = [train_file, dev_file]
        for i in range(2):
            dataset = []
            data_file = data_files[i]
            data_reader = csv.reader(data_file, delimiter = "\t")
            for line in data_reader:
                try:
                    label = int(line[1])
                except:
                    continue
                assert label in {0, 1}
                instruction = line[0]
                answer = "positive" if (label == 1) else "negative"
                output = "The author has a " + answer + " attitude."
                sample = {"instruction": instruction, "output": output, "input": ""}
                dataset.append(sample)
            data_file.close()
            res.append(dataset)
        train_dataset = SST2Dataset(data = res[0], tokenizer = tokenizer, max_length = max_length, dataset_type = "train")
        val_dataset = SST2Dataset(data = res[1], tokenizer = tokenizer, max_length = max_length, dataset_type = "validation")
        test_dataset = SST2Dataset(data = res[1], tokenizer = tokenizer, max_length = max_length, dataset_type = "generate")
        return train_dataset, val_dataset, test_dataset

class MRPCDataset(MultipleChoiceQADataset):
    MAX_SAMPLE_INPUT_LENGTH = 128
    MAX_SAMPLE_OUTPUT_LENGTH = 10
    prompt_template = {
        "prompt_no_input": "Below are two sentences. Please tell me if they are equivalent or not.\n\n### Sentences:\n\n{instruction}\n\n### Answer:\n",
        "response_split": "### Answer:",
        "response_template": r"They are ([n]?[o]?[t]?[\s]?[e][q][u][i][v][a][l][e][n][t]).",
        "failed_str": "FAILD"
    }
    @classmethod
    def get_response(cls, output: str): # override to ignore samples longer than MAX_SAMPLE_INPUT_LENGTH
        try:
            return output.split(cls.prompt_template["response_split"])[1].strip()
        except:
            print("Found an overflow sample.")
            return ""
    @classmethod
    def load_dataset(cls, data_path, tokenizer, max_length):
        data = load_from_disk(data_path)
        train_dataset = [{
            "instruction": data["train"][i]["text1"]+"\n"+data["train"][i]["text2"], 
            "output": "They are " + data["train"][i]["label_text"] + ".", 
            "input":""
        } for i in range(len(data["train"]))]
        val_dataset = [{
            "instruction": data["validation"][i]["text1"]+"\n"+data["validation"][i]["text2"], 
            "output": "They are " + data["validation"][i]["label_text"] + ".", 
            "input":""
        } for i in range(len(data["validation"]))]
        test_dataset = [{
            "instruction": data["test"][i]["text1"]+"\n"+data["test"][i]["text2"], 
            "output": "They are " + data["test"][i]["label_text"] + ".",
            "input":""
        } for i in range(len(data["test"]))]
        train_dataset = MRPCDataset(data = train_dataset, tokenizer = tokenizer, max_length = max_length, dataset_type = "train")
        val_dataset = MRPCDataset(data = val_dataset, tokenizer = tokenizer, max_length = max_length, dataset_type = "validation")
        test_dataset = MRPCDataset(data = test_dataset, tokenizer = tokenizer, max_length = max_length, dataset_type = "generate")
        return train_dataset, val_dataset, test_dataset

class QQPDataset(MultipleChoiceQADataset):
    MAX_SAMPLE_INPUT_LENGTH = 256
    MAX_SAMPLE_OUTPUT_LENGTH = 10
    prompt_template = {
        "prompt_no_input": "Below are two questions. Please tell me if they are equivalent or not.\n\n### Questions:\n\n{instruction}\n\n### Answer:\n",
        "response_split": "### Answer:",
        "response_template": r"They are ([n]?[o]?[t]?[\s]?[e][q][u][i][v][a][l][e][n][t]).",
        "failed_str": "FAILD"
    }
    @classmethod
    def load_dataset(cls, data_path, tokenizer, max_length):
        train_file = open(os.path.join(data_path, "train.tsv"))
        dev_file = open(os.path.join(data_path, "dev.tsv"))
        res = []
        for data_file in [train_file, dev_file]:
            data_reader = csv.reader(data_file, delimiter = "\t")
            data_set = []
            for line in data_reader:
                if sum([1 if (item.find("\t") != -1) else 0 for item in line]) > 0:
                    continue
                try:
                    label = int(line[-1])
                except:
                    continue
                assert label in {0, 1}
                q1 = line[3]
                q2 = line[4]
                instruction = "1. " + q1 + "\n" + "2. " + q2
                answer = "They are equivalent." if (label == 1) else "They are not equivalent."
                sample = {"instruction": instruction, "output": answer, "input": ""}
                data_set.append(sample)
            data_file.close()
            res.append(data_set)
        res[1] = res[1][::40]
        train_dataset = QQPDataset(data = res[0], tokenizer = tokenizer, max_length = max_length, dataset_type = "train")
        val_dataset = QQPDataset(data = res[1], tokenizer = tokenizer, max_length = max_length, dataset_type = "validation")
        test_dataset = QQPDataset(data = res[1], tokenizer = tokenizer, max_length = max_length, dataset_type = "generate")
        return train_dataset, val_dataset, test_dataset

class QNLIDataset(MultipleChoiceQADataset):
    MAX_SAMPLE_INPUT_LENGTH = 256
    MAX_SAMPLE_OUTPUT_LENGTH = 10
    prompt_template = {
        "prompt_no_input": "Below are a question and a sentence. Please tell me if the sentence contains the answer to the question.\n\n### Question and Sentence:\n\n{instruction}\n\n### Response:\n",
        "response_split": "### Response:",
        "response_template": r"The sentence (contains|does not contain) the answer",
        "failed_str": "FAILD"
    }
    @classmethod
    def get_response(cls, output: str): # override to ignore samples longer than MAX_SAMPLE_INPUT_LENGTH
        try:
            return output.split(cls.prompt_template["response_split"])[1].strip()
        except:
            print("Found an overflow sample.")
            return ""
    @classmethod
    def load_dataset(cls, data_path, tokenizer, max_length):
        train_file = open(os.path.join(data_path, "train.tsv"))
        dev_file = open(os.path.join(data_path, "dev.tsv"))
        res = []
        for data_file in [train_file, dev_file]:
            data_reader = csv.reader(data_file, delimiter = "\t")
            data_set = []
            for line in data_reader:
                if sum([1 if (item.find("\t") != -1) else 0 for item in line]) > 0:
                    continue
                try:
                    idx = int(line[0])
                except:
                    continue
                label = line[-1].strip()
                assert label in {"entailment", "not_entailment"}
                answer = ({"entailment": "The sentence contains the answer", "not_entailment": "The sentence does not contain the answer"})[label]
                instruction = "Question: " + line[1] + "\n" + "Sentence: " + line[2]
                sample = {"instruction": instruction, "output": answer, "input": ""}
                data_set.append(sample)
            data_file.close()
            res.append(data_set)
        train_dataset = QNLIDataset(data = res[0], tokenizer = tokenizer, max_length = max_length, dataset_type = "train")
        val_dataset = QNLIDataset(data = res[1], tokenizer = tokenizer, max_length = max_length, dataset_type = "validation")
        test_dataset = QNLIDataset(data = res[1], tokenizer = tokenizer, max_length = max_length, dataset_type = "generate")
        return train_dataset, val_dataset, test_dataset
    
class STSBDataset(MultipleChoiceQADataset):
    MAX_SAMPLE_INPUT_LENGTH = 256
    MAX_SAMPLE_OUTPUT_LENGTH = 10
    prompt_template = {
        "prompt_no_input": "Below are two sentences. Please rate the semantic similarity of these two sentences. (1-5 scale, 1 being least similar and 5 being most similar.\n\n### Sentences:\n\n{instruction}\n\n### Answer:\n",
        "response_split": "### Answer:",
        "response_template": r"Their similarity can be rated as ([12345])",
        "failed_str": "FAILD"
    }
    @classmethod
    def load_dataset(cls, data_path, tokenizer, max_length):
        train_file = open(os.path.join(data_path, "train.tsv"))
        dev_file = open(os.path.join(data_path, "dev.tsv"))
        res = []
        for data_file in [train_file, dev_file]:
            data_reader = csv.reader(data_file, delimiter = "\t")
            data_set = []
            for line in data_reader:
                if sum([1 if (item.find("\t") != -1) else 0 for item in line]) > 0:
                    continue
                try:
                    idx = int(line[0])
                except:
                    continue
                score = float(line[-1])
                assert score >= 0.0
                assert score <= 5.0
                sent1 = line[7]
                sent2 = line[8]
                if score <= 1.0:
                    label = "1"
                elif score <= 2.0:
                    label = "2"
                elif score <= 3.0:
                    label = "3"
                elif score <= 4.0:
                    label = "4"
                else:
                    label = "5"
                answer = "Their similarity can be rated as " + label
                instruction = "1. " + sent1 + "\n" + "2. " + sent2
                sample = {"instruction": instruction, "output": answer, "input": ""}
                data_set.append(sample)
            data_file.close()
            res.append(data_set)
        train_dataset = STSBDataset(data = res[0], tokenizer = tokenizer, max_length = max_length, dataset_type = "train")
        val_dataset = STSBDataset(data = res[1], tokenizer = tokenizer, max_length = max_length, dataset_type = "validation")
        test_dataset = STSBDataset(data = res[1], tokenizer = tokenizer, max_length = max_length, dataset_type = "generate")
        return train_dataset, val_dataset, test_dataset

class WNLIDataset(MultipleChoiceQADataset):
    MAX_SAMPLE_INPUT_LENGTH = 256
    MAX_SAMPLE_OUTPUT_LENGTH = 10
    prompt_template = {
        "prompt_no_input": "Given a sentence and a possible resolution of its pronoun reference, determine if the resolution is correct.\n\n### Sentence and Resolution:\n\n{instruction}\n\n### Answer:\n",
        "response_split": "### Answer:",
        "response_template": r'The resolution is (correct|wrong).',
        "failed_str": "FAILED"
    }
    @classmethod
    def load_dataset(cls, data_path, tokenizer, max_length):
        train_file = open(os.path.join(data_path, "train.tsv"), encoding='utf-8')
        dev_file = open(os.path.join(data_path, "dev.tsv"), encoding='utf-8')
        res = []
        for data_file in [train_file, dev_file]:
            data_reader = csv.reader(data_file, delimiter='\t')
            data_set = []
            for line in data_reader:
                if sum([1 if (item.find("\t") != -1) else 0 for item in line]) > 0:
                    continue
                try:
                    label = int(line[-1])
                except:
                    continue
                sentence1 = line[1].strip()
                sentence2 = line[2].strip()
                instruction = f"Sentence: {sentence1}\nProposed Resolution: {sentence2}"
                output = "The resolution is correct." if (label == 1) else "The resolution is wrong."
                sample = {"instruction": instruction, "output": output, "input": ""}
                data_set.append(sample)
            data_file.close()
            res.append(data_set)
        test_set = res[1].copy()
        train_dataset = WNLIDataset(data=res[0], tokenizer=tokenizer, max_length=max_length, dataset_type="train")
        val_dataset = WNLIDataset(data=res[1], tokenizer=tokenizer, max_length=max_length, dataset_type="validation")
        test_dataset = WNLIDataset(data=test_set, tokenizer=tokenizer, max_length=max_length, dataset_type="generate")
        return train_dataset, val_dataset, test_dataset

class MixedQADataset(MultipleChoiceQADataset):
    MAX_SAMPLE_INPUT_LENGTH = 256
    MAX_SAMPLE_OUTPUT_LENGTH = 10
    prompt_template = {
        "response_split": "### Response:",
        "response_template": r'The answer is ([A-Z]|Yes|No)',
        "failed_str": "FAILD"
    }
    @classmethod
    def get_response(cls, output: str): # override to ignore samples longer than MAX_SAMPLE_INPUT_LENGTH
        try:
            return output.split(cls.prompt_template["response_split"])[1].strip()
        except:
            return ""
    @classmethod
    def load_dataset(cls, data_path, tokenizer, max_length):
        ScienceQA_train, ScienceQA_val, ScienceQA_test = ScienceQADataset.load_dataset(data_path + "/scienceqa/science_qa.hf", tokenizer, max_length)
        BoolQ_train, BoolQ_val, BoolQ_test = BoolQDataset.load_dataset(data_path + "/BoolQ", tokenizer, max_length)
        CommonsenseQA_train, CommonsenseQA_val, CommonsenseQA_test = CommonsenseQADataset.load_dataset(data_path + "/CommonsenseQA", tokenizer, max_length)
        for data_set in [CommonsenseQA_train, CommonsenseQA_val, CommonsenseQA_test]:
            for item in data_set.data:
                if "output" in item:
                    item["output"] = item["output"].replace("The correct answer", "The answer")
        train_dataset = MixedQADataset(ScienceQA_train, BoolQ_train, CommonsenseQA_train)
        val_dataset = MixedQADataset(ScienceQA_val, BoolQ_val, CommonsenseQA_val)
        test_dataset = MixedQADataset(ScienceQA_test, BoolQ_test, CommonsenseQA_test)
        return train_dataset, val_dataset, test_dataset
    
    def __init__(self, ScienceQA_set, BoolQ_set, CommonsenseQA_set):
        self.scienceqa = ScienceQA_set
        self.boolq = BoolQ_set
        self.commonsenseqa = CommonsenseQA_set
        self.inner_idx = [i for i in range(len(ScienceQA_set))] + [i for i in range(len(BoolQ_set))] + [i for i in range(len(CommonsenseQA_set))]
        self.idx_subset = [self.scienceqa for i in range(len(ScienceQA_set))] + [self.boolq for i in range(len(BoolQ_set))] + [self.commonsenseqa for i in range(len(CommonsenseQA_set))]

    def __len__(self):
        return len(self.inner_idx)
    
    def __getitem__(self, idx):
        return (self.idx_subset[idx])[self.inner_idx[idx]]


if __name__ == "__main__":

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained('./llava/llava-1.5-7b-hf')

    train_dataset, val_dataset, test_dataset = ImageText2TextDataset.load_dataset(
        data_path = "./datasets/visual7w",
        processor = processor,
        max_length = 64
    )

    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))