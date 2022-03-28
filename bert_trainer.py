from torch.utils.data.dataloader import DataLoader
import nltk
import pandas as pd
import numpy as np
# import string
import torch
from torch.utils.data import Dataset, DataLoader
# from torchvision import datasets
from torch.nn.modules.activation import Softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from torch.optim import AdamW
from transformers import get_scheduler, BertForSequenceClassification
from transformers import TrainingArguments
from torch.utils.tensorboard import SummaryWriter
import datasets
from datasets import load_dataset
import os
from transformers import Trainer


accuracy_metric = datasets.load_metric("accuracy")
f1_metric = datasets.load_metric("f1")

accuracy_for_class_0 = datasets.load_metric("accuracy")
accuracy_for_class_1 = datasets.load_metric("accuracy")


torch.manual_seed(9)

# test_dataset = pd.read_csv('datasets/sarcasm_test.csv')
# train_dataset = load_dataset('csv', data_files=["datasets/ArSarcasm_train.csv"])
# eval_dataset = load_dataset('csv', data_files=["datasets/sarcasm_test.csv"])
# train_dataset = train_dataset['train']
# eval_dataset = eval_dataset['eval']
# print(train_dataset[1])

train_dataset = pd.read_csv("datasets/ArSarcasm_train.csv")
eval_dataset = pd.read_csv("datasets/sarcasm_test.csv")

class CustomDataset(Dataset):
    def __init__(self, data, clip_zero_class = False):
        self.data = data
        self.data['labels'] = data['sarcasm'].apply(lambda x :int(x))

        if clip_zero_class:
            # find all indexes of 1 class
            indexes_class_1 = self.data[self.data['labels'] == 1].index.values

            # caluclate number of elements of 1 class
            len_indexes_class_1 = len(indexes_class_1)

            # randomly select elements of class 0 in required number
            indexes_class_0 = self.data[self.data['labels'] == 0].index.values
            class_zero_random_selection = np.random.choice(indexes_class_0, len_indexes_class_1, replace = False)
            # merge indexes
            merged = np.concatenate((class_zero_random_selection, indexes_class_1))

            # select elments by the indexes
            self.data = self.data[self.data.index.isin(merged)]


        self.data = self.data.reset_index()

    def __getitem__(self, idx):
        return self.data.iloc[idx]['tweet'] ,self.data.iloc[idx]['labels']
    def __len__(self):
        return len(self.data)



train_dataset = CustomDataset(train_dataset, clip_zero_class=True)
eval_dataset = CustomDataset(eval_dataset)


tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02-twitter")

class BatchPreprocessor:
  def __init__(self, tokenizer):

    self.tokenizer = tokenizer
    # self.device = device
  def collate_fn(self, batch):
    list_of_texts = [x[0] for x in batch]
    list_of_labels = [x[1] for x in batch]
    labels = torch.Tensor(list_of_labels).long()
    tokenized_input = self.tokenizer(list_of_texts,
                                return_tensors="pt",
                                truncation=True,
                                padding=True)
    tokenized_input  = {k : v for k, v in tokenized_input.items()}
    tokenized_input["labels"] = labels
    return tokenized_input

# def tokenize_function(examples):
#     return tokenizer(examples["tweet"])
#
#
# train_tokenized_datasets = train_dataset.map(tokenize_function, batched=True)['train']
# eval_tokenized_datasets = eval_dataset.map(tokenize_function, batched=True)['train']
# print(train_tokenized_datasets[2])


model = AutoModelForSequenceClassification.from_pretrained("aubmindlab/bert-base-arabertv02-twitter", num_labels=2)


def compute_metrics(eval_pred):
    metrics_dict = {}
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    metrics_dict.update(accuracy_metric.compute(predictions=predictions, references=labels))
    metrics_dict.update(f1_metric.compute(predictions=predictions, references=labels))


    idx_0 = labels == 0
    idx_1 = labels == 1
    result_acc_0 = accuracy_for_class_0.compute(predictions=predictions[idx_0], references=[0] * sum(idx_0))
    result_acc_1 = accuracy_for_class_1.compute(predictions=predictions[idx_1], references=[1] * sum(idx_1))
    accuracy_0 = {k + '_0': v for k, v in result_acc_0.items()}
    accuracy_1 = {k + '_1': v for k, v in result_acc_1.items()}
    metrics_dict.update(accuracy_0)
    metrics_dict.update(accuracy_1)


    return metrics_dict

OUTPUT_DIR = "using_trainer"
TRAIN_BS = 32
EVAL_BS = 16
LR = 5e-3
EPOCHS = 3
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    per_device_train_batch_size=TRAIN_BS,
    per_device_eval_batch_size=EVAL_BS,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    save_strategy="no"
)

# data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=BatchPreprocessor(tokenizer=tokenizer).collate_fn
)

trainer.train()