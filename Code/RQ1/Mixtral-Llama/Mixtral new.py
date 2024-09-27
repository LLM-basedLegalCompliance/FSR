"""Mixtral.ipynb
"""
from trl import SFTTrainer
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from datasets import load_dataset,Dataset
from peft import LoraConfig, prepare_model_for_kbit_training,AutoPeftModelForCausalLM,get_peft_model,PeftModel, PeftConfig
from transformers import LlamaTokenizer
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
# from transformers.generation.utils import top_k_top_p_filtering
from trl import SFTTrainer
#The one I created for Mixtral experiemnts in EMSE paper
access_token=""
mistral_checkpoint = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1",use_auth_token=access_token)

#prepare Train Data
# Load the Excel file
xlsx_file = pd.ExcelFile('SFCR_1.xlsx')

# Get the sheet names
sheet_names = xlsx_file.sheet_names

# Create dataframes with specific names
df_sentences = xlsx_file.parse(sheet_names[0])  # Assuming the first sheet contains sentences
df_paragraphs = xlsx_file.parse(sheet_names[1])  # Assuming the second sheet contains paragraphs

# Removing rows where 'paragraph_id' is NaN
df_paragraphs = df_paragraphs.dropna(subset=['paragraph_id'])
# Resetting the index (optional)
df_paragraphs = df_paragraphs.reset_index(drop=True)
# Replace NaN values with 0 in df_sentences
df_sentences.fillna(0, inplace=True)
# Create dataframes with specific names
# df = xlsx_file.parse(sheet_names[0])
# print("columns:", df_paragraphs.columns)


#prepare Test Data
test_xlsx_file = pd.ExcelFile('Annotation_1.xlsx')
# Get the sheet names
test_sheet_names = test_xlsx_file.sheet_names
# Create dataframes with specific names
test_df_sentences = test_xlsx_file.parse(test_sheet_names[0])  # Assuming the first sheet contains sentences
test_df_paragraphs = test_xlsx_file.parse(test_sheet_names[1])  # Assuming the second sheet contains paragraphs
# Removing rows where 'paragraph_id' is NaN
test_df_paragraphs = test_df_paragraphs.dropna(subset=['paragraph_id'])
# Resetting the index (optional)
test_df_paragraphs = test_df_paragraphs.reset_index(drop=True)
# Replace NaN values with 0 in df_sentences
test_df_sentences.fillna(0, inplace=True)
# Create dataframes with specific names
# test_df = test_xlsx_file.parse(test_sheet_names[0])

delimiter="%%"
# Specified columns
specified_columns = ['overall', 'Data', 'LabelData', 'Non-labelData', 'Measurement', 'Temperature', 'Size', 'Mass', 'Water Content', 'Pathogen', 'Firmness', 'Colour', 'Time Constraint']

# Convert paragraphs dataframe to dictionary
paragraphs = {int(row['paragraph_id']): row['paragraph'] for _, row in df_paragraphs.iterrows()}

# Convert sentences dataframe to dictionary
sentences = {}
for _, row in df_sentences.iterrows():
    pid = int(row['paragraph_id'])
    if pid not in sentences:
        sentences[pid] = []

    # Extract specified labels for each sentence
    # labels = [str(int(float(row[col]))) for col in specified_columns]
    # sentences[pid].append((row['Statement'], labels))
    labels = [specified_columns[i] for i, col in enumerate(specified_columns) if row[col] == 1]
    sentences[pid].append((row['Statement'], labels))



task_string = f"""You are an asssitsant tasked with extracting relevant text segments and their labels from the provided food safety paragraph. The entire paragraph is delimited within {delimiter} characters. Use only the provided labels in this exact order: 'Overall', 'Data', 'Label Data', 'Non-label Data', 'Measurement', 'Temperature', 'Size', 'Mass', 'Water Content', 'Pathogen', 'Firmness', 'Colour', 'Time Constraint'.
-Data: any information used to convey knowledge, provide assurance, or perform analysis. This includes 'Label Data' and 'Non-label Data'.-Label Data: a subtype of 'Data' that includes information that a food-product package or container must bear.-Non-label Data: a subtype of 'Data' that includes any food-safety-relevant data other than \
label data that needs to be collected and/or retained for inclusion in documents such as certificates, reports, guarantees, and letters.-Measurement: Association of numbers with physical quantities. This includes measurements of 'Colour', 'Firmness', 'Mass', 'Pathogen', 'Size', 'Temperature', and 'Water Content'.-Colour: a subtype of 'Measurement' \
that is self-evident.-Firmness: a subtype of 'Measurement' that refers to the degree of resistance to deformation.-Mass: a subtype of 'Measurement' that refers to the amount of substance by weight or volume.-Pathogen: a subtype of 'Measurement' that refers to a microorganism that causes disease.-Size: a subtype of 'Measurement' that refers to dimension \
(e.g., length or thickness) or surface area.-Temperature: a subtype of 'Measurement' that is self-evident.-Water Content: a subtype of 'Measurement' that refers to humidity or moisture.- Time Constraint: A temporal restriction, in our context, is expressed using intervals, deadlines or periodicity.-Overall: requirements-related provisions that include all the introduced concepts."""
train_data = []
for pid, section in paragraphs.items():
    user_message = section
    assistant_message = ' #### '.join([f"{sentence} ::: {', '.join(labels)}" for sentence, labels in sentences[pid]])
    dialogue = f"""{task_string}\nParagraph:\n{delimiter}{user_message}{delimiter}"""

    train_data.append({
        "dialogue": dialogue,
        "Target": assistant_message
    })

# Create DataFrame
df = pd.DataFrame(train_data)

train_df, val_df = train_test_split(df, test_size=0.1, shuffle=True, random_state=42)


#Test Data
# Generate Training data
delimiter="%%"
# Specified columns
specified_columns = ['Overall', 'Data', 'LabelData', 'Non-labelData', 'Measurement', 'Temperature', 'Size', 'Mass', 'Water Content', 'Pathogen', 'Firmness', 'Colour', 'Time Constraint']

# Convert paragraphs dataframe to dictionary
test_paragraphs = {int(row['paragraph_id']): row['Statement'] for _, row in test_df_paragraphs.iterrows()}

# Convert sentences dataframe to dictionary
test_sentences = {}
for _, row in test_df_sentences.iterrows():
    pid = int(row['paragraph_id'])
    if pid not in test_sentences:
        test_sentences[pid] = []

    # Extract specified labels for each sentence
    # labels = [str(int(float(row[col]))) for col in specified_columns]
    # test_sentences[pid].append((row['Statement'], labels))
    labels = [specified_columns[i] for i, col in enumerate(specified_columns) if row[col] == 1]
    sentences[pid].append((row['Statement'], labels))



test_data = []
for pid, section in test_paragraphs.items():
    user_message = section
    assistant_message = ' #### '.join([f"{sentence} ::: {', '.join(labels)}" for sentence, labels in test_sentences[pid]])
    dialogue = f"""{task_string}\nParagraph:\n{delimiter}{user_message}{delimiter}"""

    test_data.append({
        "dialogue": dialogue,
        "Target": assistant_message
    })

# test_Create DataFrame
test_df = pd.DataFrame(test_data)

train_df["text"] = train_df[["dialogue","Target"]].apply(lambda x: "<s>[INST]"+ x["dialogue"]+"[/INST]"+ x["Target"]+"</s>", axis=1)
test_df["text"] = test_df[["dialogue","Target"]].apply(lambda x: "<s>[INST]"+ x["dialogue"]+"[/INST]"+ x["Target"]+"</s>", axis=1)
val_df["text"] = val_df[["dialogue","Target"]].apply(lambda x: "<s>[INST]"+ x["dialogue"]+"[/INST]"+ x["Target"]+"</s>", axis=1)


# Convert each split to a Hugging Face dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
val_dataset = Dataset.from_pandas(val_df)


train_dataset=train_dataset.remove_columns(['__index_level_0__'])
# test_dataset=test_dataset.remove_columns(['__index_level_0__'])
val_dataset=val_dataset.remove_columns(['__index_level_0__'])


# tokenizer = AutoTokenizer.from_pretrained(mistral_checkpoint, use_auth_token=access_token)
tokenizer.pad_token = tokenizer.eos_token
#tokenizer.padding_side = 'right'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    load_in_8bit_fp32_cpu_offload=True  # Enable CPU offload
)

model = AutoModelForCausalLM.from_pretrained(mistral_checkpoint, quantization_config=bnb_config, device_map='auto',use_auth_token=access_token)
# model = AutoModelForCausalLM.from_pretrained(mistral_checkpoint, use_auth_token=access_token,quantization_config=bnb_config, device_map='auto')

model_size = sum(p.numel() for p in model.parameters())
print(f"Model Params: {model_size} parameters")

model_size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 3)
print(f"Model size in RAM: {model_size_gb:.4f} GB")



model.config.use_cache=False
model.config.pretraining_tp=1
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(r=16,lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()


training_arguments = TrainingArguments(
        output_dir="Mixtral_v1",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        optim="adamw_hf",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        num_train_epochs=20,
        max_grad_norm=0.3,
        warmup_ratio= 0.1,
		save_total_limit=1,
        fp16=True,
        do_eval=True,
        use_cpu=False
)

# Initialize SFTTrainer without unsupported arguments
trainer = SFTTrainer(
    model=peft_model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    args=training_arguments,
    tokenizer=tokenizer,
    packing=False,
    max_seq_length=4096
)

# trainer.train()

"""#PEFT_MODEL LOADING"""
peft_model = PeftModel.from_pretrained(model,
                                       'Mixtral_v1/checkpoint-1000',
                                       torch_dtype=torch.bfloat16,
                                       is_trainable=False)

print(peft_model)

peft_model_size = sum(p.numel() for p in peft_model.parameters())
print(f"PEFT Model Params: {peft_model_size} parameters")

peft_model_size_gb = sum(p.numel() * p.element_size() for p in peft_model.parameters()) / (1024 ** 3)
print(f"Model size in RAM: {peft_model_size_gb:.4f} GB")

temperature_value=0.2
peft_model_output= []
targets = []
for index, row in test_df[0:5].iterrows():
    target = row['Target']
    dialogue = row['dialogue']
    prompt = f"""
    {dialogue}
    Answer:
    """

    model_inputs = tokenizer([prompt], return_tensors="pt").to('cuda')
    peft_generated_ids = model.generate(**model_inputs, max_new_tokens=256, do_sample=True, temperature=temperature_value)
    output_p=tokenizer.batch_decode(peft_generated_ids, skip_special_tokens=True)[0]
    split_text_p = output_p.split("Answer:")
    peft_model_text_output=split_text_p[1].strip() if len(split_text_p) > 1 else None
    peft_model_output.append(peft_model_text_output)
    print("peft_model_text_output",peft_model_text_output)
    targets.append(target)

# Create DataFrame with predictions and actual targets
results_df = pd.DataFrame({'Model Output': peft_model_output, 'Actual Target': targets})
# Save to CSV
results_df.to_csv("Mixtralresults_comparison.csv", index=False)


