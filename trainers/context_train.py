import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import os
import re

from datasets import Dataset, load_metric
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define the base directory and model paths
current_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(current_dir, '../src/model')
inspect_dir = os.path.join(current_dir, '../data/processed/inspect')
dict_dir = os.path.join(current_dir, '../data/raw/misc')
log_dir = os.path.join(current_dir, '../data/logs')

keyword_dict = {
    # Add any specific keyword mappings here if needed
}

class RobertaForMultiLabelSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0][:, 0, :]
        logits = self.classifier(sequence_output)
        logits = torch.sigmoid(logits)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
        return (loss, logits) if loss is not None else logits

# Load and pre-process scraped article dataset
def load_descriptions(filepath):
    df = pd.read_csv(filepath)
    # Ensure that the 'article' column is a string and handle NaN values
    df['article'] = df['article'].fillna('').astype(str)
    df['context'] = df['article'].apply(lambda x: x.strip().lower())
    return df

# Load and aggregate unique values from multiple categorical columns across datasets.
def load_unique_values(data_paths, categorical_columns):
    unique_values = {col: set() for col in categorical_columns}
    for path in data_paths:
        df = pd.read_csv(path)
        for col in categorical_columns:
            if col in df.columns:
                unique_values[col].update(df[col].unique())
    for col in unique_values:
        unique_values[col] = list(unique_values[col])
    return unique_values

# Prepare article data for model by creating binary labels for each unique value in each categorical column.
def prepare_data(df, unique_values):
    df['processed_context'] = df['context'].apply(lambda x: preprocess_description(x, keyword_dict))
    # Initialize an empty dictionary to hold new label columns
    label_data = {}
    
    # List to keep track of the new label column names
    label_columns = []
    
    # Populate the dictionary with new label columns
    for col, values in unique_values.items():
        for value in values:
            label_name = f"label_{col}_{value}"
            label_data[label_name] = df['processed_context'].apply(lambda x: 1 if value.lower() in x else 0)
            label_columns.append(label_name)  # Add the column name to the list
    
    # Convert the dictionary of new columns into a DataFrame
    label_df = pd.DataFrame(label_data)
    
    # Concatenate the new DataFrame with the original DataFrame
    df = pd.concat([df, label_df], axis=1)
    
    # Create a list of labels for each row, combining all label columns
    df['labels'] = df[label_columns].values.tolist()
    
    # Return both the updated DataFrame and the list of label columns
    return df, label_columns


# Process descriptions to allow for easy finding of data
def preprocess_description(description, keyword_dict):
    description = description.lower()
    for key, value in keyword_dict.items():
        description = re.sub(r'\b{}\b'.format(key), value, description)
    return description

# Tokenize the descriptions
def tokenize_data(examples, tokenizer, label_columns):
    tokenized_inputs = tokenizer(examples['processed_context'], padding="max_length", truncation=True, max_length=512)
    labels = [[int(examples[col][i]) for col in label_columns] for i in range(len(examples['processed_context']))]
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

def main(LIMITER):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    categorical_columns = [
        'Sector', 'Sub-Sector', 'Piles Material', 'Pile Caps Material', 'Capping Beams Material', 
        'Raft Foundation Material', 'Basement Walls Material', 'Lowest Floor Slab Material', 
        'Ground Insulation Material', 'Core Structure Material', 'Columns Material', 
        'Beams Material', 'Secondary Beams Material', 'Floor Slab Material', 'Joisted Floors Material', 
        'Roof Material', 'Roof Insulation Material', 'Roof Finishes Material', 'Facade Material', 
        'Wall Insulation Material', 'Glazing Material', 'Window Frames Material', 
        'Partitions Material', 'Ceilings Material', 'Floors Material', 'Services'
    ]

    numerical_columns = [
        'Gross Internal Area (m2)', 'Building Perimeter (m)', 'Building Footprint (m2)', 
        'Building Width (m)', 'Floor-to-Floor Height (m)', 'Storeys Above Ground', 
        'Storeys Below Ground', 'Glazing Ratio (%)'
    ]

    desc_path = os.path.join(dict_dir, 'building_descriptions.csv')
    if os.path.exists(desc_path):
        desc_data = load_descriptions(desc_path)
    else:
        raise FileNotFoundError(f"Description file not found at {desc_path}")

    train_data_paths = [os.path.join(inspect_dir, 'cleaned_synthetic.csv')]
    unique_values = load_unique_values(train_data_paths, categorical_columns)

    processed_data, label_columns = prepare_data(desc_data, unique_values)
    processed_data = processed_data[:LIMITER]
    dataset = Dataset.from_pandas(processed_data)

    split_datasets = dataset.train_test_split(test_size=0.2)
    train_dataset = split_datasets['train'].map(lambda examples: tokenize_data(examples, tokenizer, label_columns), batched=True)
    test_dataset = split_datasets['test'].map(lambda examples: tokenize_data(examples, tokenizer, label_columns), batched=True)

    config = RobertaConfig.from_pretrained('roberta-base', num_labels=len(label_columns))
    model = RobertaForMultiLabelSequenceClassification(config)

    # Define compute_metrics function using sklearn's r2_score
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.sigmoid(torch.tensor(logits)).numpy() > 0.5  # Applying threshold to get binary predictions
        labels = np.array(labels)
        
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='macro')
        recall = recall_score(labels, predictions, average='macro')
        f1 = f1_score(labels, predictions, average='macro')
        
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    # Setup TrainingArguments and Trainer as before, incorporating the new compute_metrics
    training_args = TrainingArguments(
        output_dir=os.path.join(processed_dir, 'context/results'),
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",  # Save model at the end of each epoch
        logging_dir=log_dir,  # Log metrics in the specified directory
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # Train the model and capture the result
    train_result = trainer.train()

    # Save model and tokenizer
    model_path = os.path.join(processed_dir, 'context/saved_model')
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    # Extract R² scores from trainer's state and write them to a file
    r2_train = trainer.state.log_history[-2]['eval_r2'] if 'eval_r2' in trainer.state.log_history[-2] else "No training R2 recorded"
    r2_eval = trainer.state.log_history[-1]['eval_r2'] if 'eval_r2' in trainer.state.log_history[-1] else "No evaluation R2 recorded"
    
    with open(os.path.join(log_dir, 'context_performance.txt'), 'w') as file:
        file.write(f"Training R²: {r2_train}\n")
        file.write(f"Validation R²: {r2_eval}\n")

    # Print confirmation message
    print("Results have been written to:", os.path.join(log_dir, 'context_performance.txt'))
    print("Model and tokenizer have been saved to:", model_path)

if __name__ == '__main__':
    main(100)