# coding=utf-8
# model_name = "mistralai/Mistral-7B-v0.1"
# model_name = "openchat/openchat-3.5-1210"
# model_name = "tiiuae/falcon-7b"
# model_name = "TheBloke/Llama-2-7B-Chat-GPTQ"
# model_name = "NousResearch/Llama-2-7b-chat-hf"
import os
import torch
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)

# from transformers import Seq2SeqTrainingArguments as TrainingArguments


from trl import SFTTrainer
from data_read import convert_csv_to_traindataset
import numpy as np
from sklearn.metrics import accuracy_score
torch.set_float32_matmul_precision('high')

def normalize(text):
    return text.lower().strip().replace("-", " ")

def normalize(text):
    """统一格式用于对比：小写、去空格、去破折号"""
    return text.lower().strip().replace("-", " ")

# def compute_metrics(eval_preds, tokenizer=None):
#     preds, labels = eval_preds
#     preds = preds[:]
#     print(preds.shape)
#     print(labels.shape)
#     # exit(1)
#     # 有可能是 tensor，转成 list
#     # preds = preds.tolist() if hasattr(preds, "tolist") else preds
#     labels = labels.tolist() if hasattr(labels, "tolist") else labels

#     # 解码
#     # pred_strs = tokenizer.batch_decode(preds[0], skip_special_tokens=True)
#     label_strs = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
#     # print(pred_strs)
#     print(label_strs)
#     exit(1)

#     # 计算准确率（忽略大小写和空格）
#     correct = 0
#     total = len(pred_strs)

#     print("\n=== Example Predictions ===")
#     for i, (p, l) in enumerate(zip(pred_strs, label_strs)):
#         norm_p = normalize(p)
#         norm_l = normalize(l)
#         match = norm_p == norm_l
#         correct += match

#         # 打印前5条
#         if i < 5:
#             print(f"[{i}]")
#             print(f"Prediction: {p.strip()}")
#             print(f"Label     : {l.strip()}")
#             print(f"{'✅ Correct' if match else '❌ Incorrect'}\n")

#     acc = correct / total
#     return {"accuracy": acc}





@dataclass
class ScriptArguments:
    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})
    per_device_train_batch_size: Optional[int] = field(default=8)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    max_seq_length: Optional[int] = field(default=512)
    model_name: Optional[str] = field(default="openchat/openchat-3.5-1210")
    dataset_name: Optional[str] = field(default="timdettmers/openassistant-guanaco")
    use_4bit: Optional[bool] = field(default=True)
    use_nested_quant: Optional[bool] = field(default=False)
    bnb_4bit_compute_dtype: Optional[str] = field(default="float16")
    bnb_4bit_quant_type: Optional[str] = field(default="nf4")
    num_train_epochs: Optional[int] = field(default=3)
    fp16: Optional[bool] = field(default=False)
    bf16: Optional[bool] = field(default=False)
    packing: Optional[bool] = field(default=False)
    gradient_checkpointing: Optional[bool] = field(default=False)
    optim: Optional[str] = field(default="adamw_torch")
    lr_scheduler_type: str = field(default="constant")
    max_steps: int = field(default=-1)
    warmup_ratio: float = field(default=0.03)
    group_by_length: bool = field(default=True)
    save_steps: int = field(default=None)
    save_strategy: str = field(default="epoch")
    logging_strategy: str = field(default="epoch")
    evaluation_strategy: str = field(default="epoch")
    logging_steps: int = field(default=10)
    merge_and_push: Optional[bool] = field(default=True)
    output_dir: str = field(default="./results")

def create_and_prepare_model(args):
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_nested_quant,
    )
    # Load the entire model onto GPU 0
    # Switch to device_map = "auto" for multi-GPU configurations
    device_map = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        quantization_config=bnb_config, 
        device_map=device_map, 
        use_auth_token=True
    )
    model.config.pretraining_tp = 1 
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM", 
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, peft_config, tokenizer

from tqdm import tqdm

def generate_from_dataset(model, tokenizer, dataset, labels, max_new_tokens=100):
    model.eval()
    outputs = []

    for example, label in zip(tqdm(dataset, desc="Generating"), labels):
        prompt = example["text"]
        
        prompt = "can you tell me what's 1+1?"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs)
            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"Prompt: {prompt}")
            print(f"Response: {response}")
            # print(label)
            exit(1)
            
            
            outputs.append({
                "input": prompt,
                "response": response
            })
    
    return outputs


def train_model(args):
    model, peft_config, tokenizer = create_and_prepare_model(args)
    model.config.use_cache = False
    
    # sample = tokenizer("This company is looking for a graduate software engineer.", return_tensors="pt").to(model.device)
    # output_ids = model.generate(**sample, max_length=32)
    # print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
    # exit(1)
    #######################################################################
    # dataset = load_dataset(args.dataset_name, split="train")
    # print(dataset[0])
    # exit(1)
    # dataset = dataset.select(range(100))  # ✅ 正确，保持 Dataset 类型
    ##############################################################
    train_dataset, eval_dataset, labels = convert_csv_to_traindataset("dataset/cleaned_seniority_labelled_development_set.csv")
    
    generate_from_dataset(model, tokenizer, eval_dataset, labels)
    
    # print(train_dataset[0])
    # print(test_dataset[0])
    # print(labels[0])
    # exit(1)
    
    
    
    train_dataset = train_dataset.select(range(10))
    ##########################################################################
    
    
  # Limit to 100 samples for testing
    # Fix the unusual overflow issue in fp16 training.
    tokenizer.padding_side = "right"
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        # save_steps=args.save_steps,
        # logging_steps=args.logging_steps,
        evaluation_strategy=args.evaluation_strategy,    # ✅ 每个 epoch 执行一次 eval
        save_strategy=args.save_strategy,
        logging_strategy=args.logging_strategy,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        bf16=args.bf16,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        group_by_length=args.group_by_length,
        # predict_with_generate=True,
        # generation_max_length=32,
        # generation_num_beams=1,
        lr_scheduler_type=args.lr_scheduler_type,
    )
    from functools import partial
    # wrapped_compute_metrics = partial(compute_metrics, tokenizer=tokenizer)
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # compute_metrics=wrapped_compute_metrics,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=args.packing,
    )
    trainer.train()
    # test_metrics = trainer.evaluate(eval_dataset=dataset)
    # print(">>> Test set metrics:", test_metrics)
    # exit(1)

    if args.merge_and_push:
        output_dir = os.path.join(args.output_dir, "final_checkpoints")
        trainer.model.save_pretrained(output_dir)
        # Free up memory for merging weights
        del model
        torch.cuda.empty_cache()
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
        model = model.merge_and_unload()
        output_merged_dir = os.path.join(args.output_dir, "final_merged_checkpoint")
        model.save_pretrained(output_merged_dir, safe_serialization=True)
    
    
    
    # merged_dir = os.path.join(args.output_dir, "final_merged_checkpoint")
    # model = AutoModelForCausalLM.from_pretrained(
    #     merged_dir,
    #     torch_dtype=torch.bfloat16,
    # ).to("cuda:0")  # 如果你有多个 GPU，指定一个 GPU
    # tokenizer = AutoTokenizer.from_pretrained(merged_dir)  # 如果你把 tokenizer 也保存在那里
    
    
    
    # with torch.no_grad():
    #     outputs = model.generate(**inputs)

    # # 2. 把 Trainer 的 model 替换成新加载的 model
    # trainer.model = model
    # trainer.tokenizer = tokenizer

    
    # real_train = convert_csv_to_traindataset
    # # 3. 评估训练集
    # train_metrics = trainer.evaluate(
    #     eval_dataset=trainer.train_dataset,   # 或者显式传入 train_dataset
    # )
    # print(">>> Train set metrics:", train_metrics)

    # # 4. 评估测试集
    # # 假设你已经有一个 HuggingFace Dataset 变量叫 test_dataset
    # test_metrics = trainer.evaluate(
    #     eval_dataset=test_dataset,
    # )
    # print(">>> Test set metrics:", test_metrics)
    
if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    train_model(script_args)
