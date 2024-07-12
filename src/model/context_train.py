import pandas as pd
import torch.nn as nn
import torch
import os
import re
from datasets import Dataset, load_metric
from transformers import RobertaTokenizer, Trainer, TrainingArguments, RobertaConfig, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Define the base directory and model paths
current_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(current_dir, '../../data/processed')
dict_dir = os.path.join(current_dir, '../../data/raw/misc')

keyword_dict = {
    # Add your keyword mappings here
}

class RobertaForMultiTaskSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config, num_numerical_features):
        super().__init__(config)
        # Replace the classifier with a new one for multi-label (binary per label)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.num_numerical_features = num_numerical_features
        self.numerical_classifier = nn.Linear(config.hidden_size, num_numerical_features)

    def forward(self, input_ids=None, attention_mask=None, labels=None, numerical_labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0][:, 0, :]  # Take the first token (CLS token) for classification tasks

        logits = self.classifier(sequence_output)  # (batch_size, num_labels)
        numerical_logits = self.numerical_classifier(sequence_output)  # (batch_size, num_numerical_features)

        logits = torch.sigmoid(logits)
        numerical_logits = torch.sigmoid(numerical_logits)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            categorical_loss = loss_fct(logits, labels.float())
            numerical_loss = nn.MSELoss()(numerical_logits, numerical_labels.float())
            loss = categorical_loss + numerical_loss

        return (loss, logits, numerical_logits) if loss is not None else (logits, numerical_logits)

# Load and pre-process scraped article dataset
def load_descriptions(filepath):
    df = pd.read_csv(filepath)
    df['context'] = df['article'].apply(lambda x: x.strip().lower())
    return df

# Load and aggregate unique values from multiple categorical columns across datasets.
def load_unique_values(data_paths, categorical_columns):
    unique_values = {col: set() for col in categorical_columns}  # Initialize with empty sets for each category

    for path in data_paths:
        df = pd.read_csv(path)
        for col in categorical_columns:
            if col in df.columns:  # Check if the column exists in the dataframe
                unique_values[col].update(df[col].unique())  # Use set to avoid duplicate values
    
    for col in unique_values:
        unique_values[col] = list(unique_values[col])
        
    return unique_values

# Prepare article data for model by creating binary labels for each unique value in each categorical column.
def prepare_data(df, unique_values, numerical_columns):
    # Applying preprocessing to the 'processed_context'
    df['processed_context'] = df['context'].apply(lambda x: preprocess_description(x, keyword_dict))

    # Initialize an empty DataFrame to hold all labels
    label_columns = []

    # Create binary labels for each unique value in each categorical column
    for col, values in unique_values.items():
        for value in values:
            label_name = f"label_{col}_{value}"
            df[label_name] = df['processed_context'].apply(lambda x: 1 if value.lower() in x else 0)
            label_columns.append(label_name)

    # Normalize numerical columns
    scaler = MinMaxScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    df['labels'] = df[label_columns].values.tolist()  # This would create a list of labels per example
    df['numerical_labels'] = df[numerical_columns].values.tolist()

    return df, label_columns, numerical_columns

# Process descriptions to allow for easy finding of data
def preprocess_description(description, keyword_dict):
    # Lowercase the description for uniformity
    description = description.lower()
    # Replace variations with standard terms
    for key, value in keyword_dict.items():
        description = re.sub(r'\b{}\b'.format(key), value, description)
    return description

# Tokenize the descriptions
def tokenize_data(examples, tokenizer, label_columns, numerical_columns):
    tokenized_inputs = tokenizer(examples['processed_context'], padding="max_length", truncation=True, max_length=512)
    
    # Here we should reshape the labels to fit the expected model input for multi-label classification
    # Flatten the list of labels into a single list per example if not already done
    labels = [[int(examples[col][i]) for col in label_columns] for i in range(len(examples['processed_context']))]
    numerical_labels = [examples[numerical_columns].values.tolist()[i] for i in range(len(examples['processed_context']))]
    tokenized_inputs['labels'] = labels
    tokenized_inputs['numerical_labels'] = numerical_labels

    return tokenized_inputs

def main():
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Define categorical columns:
    categorical_columns = [
        'Building Project Type', 'Primary Foundation Type', 'Primary Ground Floor Type',
        'Primary Vertical Element Type', 'Primary Horizontal Element Type', 'Primary Slab Type',
        'Primary Cladding Type', 'Primary Heating Type', 'Primary Cooling Type',
        'Primary Finishes Type', 'Primary Ventilation Type', 'Building Use Type',
        'Building Use Subtype', 'Continent', 'Country', 'Structure Type'
    ]

    # Define numerical columns:
    numerical_columns = ['Total Users', 'Floors Above Ground', 'Floors Below Ground']

    desc_path = os.path.join(dict_dir, 'building_descriptions.csv')
    if os.path.exists(desc_path):
        desc_data = load_descriptions(desc_path)
    else:
        raise FileNotFoundError(f"Description file not found at {desc_path}")

    train_data_paths = [
        os.path.join(processed_dir, 'inspect/cleaned_becd.csv'),
        os.path.join(processed_dir, 'inspect/cleaned_carbenmats.csv')
    ]
    unique_values = load_unique_values(train_data_paths, categorical_columns)

    processed_data, label_columns, numerical_columns = prepare_data(desc_data, unique_values, numerical_columns)
    dataset = Dataset.from_pandas(processed_data)

    split_datasets = dataset.train_test_split(test_size=0.2)
    train_dataset = split_datasets['train'].map(lambda examples: tokenize_data(examples, tokenizer, label_columns, numerical_columns), batched=True)
    test_dataset = split_datasets['test'].map(lambda examples: tokenize_data(examples, tokenizer, label_columns, numerical_columns), batched=True)

    config = RobertaConfig.from_pretrained('roberta-base', num_labels=len(label_columns))
    model = RobertaForMultiTaskSequenceClassification(config, num_numerical_features=len(numerical_columns))

    training_args = TrainingArguments(
        output_dir=os.path.join(processed_dir, 'model/context/results'),
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    trainer.train()

    model.save_pretrained(os.path.join(processed_dir, 'model/context/saved_model'))
    tokenizer.save_pretrained(os.path.join(processed_dir, 'model/context/saved_model'))

if __name__ == '__main__':
    main()
