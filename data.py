
import pandas as pd
from datasets import Dataset


# csv_file_path = 'dataset/cleaned_seniority_labelled_development_set.csv'
csv_file_path = "dataset/cleaned_work_arrangements_development_set.csv"
csv_file_path2 = "dataset/cleaned_work_arrangements_test_set.csv"
# csv_file_path = "dataset/cleaned_salary_labelled_development_set.csv"    
# 读取CSV文件

def select_ytrue_candidate(df):

    # 获取'y_true'列的唯一值
    unique_values = df['y_true'].unique()
    return unique_values


# 2. Build English prompt and format into target structure
def build_prompt(row, y_true=None, name=None):
    
    if "seniority" in name:
    
        prompt = f"""You are given a job posting. Your task is to classify the required skill level of the position.
            Only respond with **one of the following options** (and nothing else): {y_true}.
            The job posting is as follows:

            - Job Title: {row['job_title']}
            - Summary: {row['job_summary']}
            - Job Details: {row['job_ad_details']}
            - Classification: {row['classification_name']}
            - Subclassification: {row['subclassification_name']}"""
    
    elif "work_arrangements" in name:
        prompt = f"""You are given a job posting. Your task is to classify the work arrangements of the position.
            Only respond with **one of the following options** (and nothing else): {y_true}.
            The job posting is as follows:
            -Job Details: {row['job_ad']}"""

    prompt = f"### Human: {prompt}\n### Assistant: {row['y_true']}"
    
    return prompt
    # return prompt
    # return f"### Human: {prompt}\n### Assistant: {row['y_true']}"



def convert_csv_to_traindataset(name):
    # 1. Read CSV
    df = pd.read_csv(name)

    # 2. 获取 y_true（例如标签列名）
    y_true = select_ytrue_candidate(df)  # 自定义函数，确保它返回的是 Series
    
    # 3. 构造 text_train 和 text_test（假设 build_prompt 返回的是元组）
    
    
    df['text'] = df.apply(lambda row: pd.Series(build_prompt(row, y_true, name)), axis=1)

    dataset = Dataset.from_pandas(df[['text']])
    return dataset



# build_dataset(csv_file_path)

# Load the dataset
# dataset = load_dataset('timdettmers/openassistant-guanaco')

# # Shuffle the dataset and slice it
# dataset = dataset['train'].shuffle(seed=42).select(range(1000))


# # print(dataset.type)
# print(len(dataset))
# print(dataset[0])
# print(dataset.type)



# Define a function to transform the data
def transform_conversation(example):
    conversation_text = example['text']
    segments = conversation_text.split('###')

    reformatted_segments = []

    # Iterate over pairs of segments
    for i in range(1, len(segments) - 1, 2):
        human_text = segments[i].strip().replace('Human:', '').strip()

        # Check if there is a corresponding assistant segment before processing
        if i + 1 < len(segments):
            assistant_text = segments[i+1].strip().replace('Assistant:', '').strip()

            # Apply the new template
            reformatted_segments.append(f'<s>[INST] {human_text} [/INST] {assistant_text} </s>')
        else:
            # Handle the case where there is no corresponding assistant segment
            reformatted_segments.append(f'<s>[INST] {human_text} [/INST] </s>')

    return {'text': ''.join(reformatted_segments)}



def build_dataset(name):
    dataset = convert_csv_to_traindataset(name)
    transformed_dataset = dataset.map(transform_conversation)
    return transformed_dataset


dataset = build_dataset(csv_file_path2)# Apply the transformation


print(len(dataset))