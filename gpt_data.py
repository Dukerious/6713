import os
import pandas as pd
from datasets import Dataset, load_metric
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTTrainingArguments
from sklearn.metrics import accuracy_score

csv_path = './data/job_ads.csv'
df = pd.read_csv(csv_path)
# 将 DataFrame 转为 HuggingFace Dataset
dataset = Dataset.from_pandas(df)

