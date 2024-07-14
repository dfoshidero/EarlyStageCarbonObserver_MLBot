import os
import pandas as pd
import numpy as np
import re
import torch

from alive_progress import alive_bar
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.optim import AdamW

class ArticleDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        # Assuming the 'description' column contains text and the rest are binary labels
        self.texts = dataframe['description'].values  # Convert to numpy array for easier indexing
        self.labels = torch.FloatTensor(dataframe.drop(columns=['description']).values)  # Convert all other columns to a tensor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),  # Remove extra dimensions
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': label
        }


def train_model(model, train_loader, optimizer, device):
    model.train()  # Set the model to training mode
    total_loss = 0
    with alive_bar(len(train_loader), title='Training') as bar:
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            model.zero_grad()  # Clear previously calculated gradients
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]  # The model outputs are always tuple in transformers (loss, logits)

            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            total_loss += loss.item()
            bar()  # Update the progress bar
    return total_loss / len(train_loader)

def evaluate_model(model, val_loader, device, threshold=0.5):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_true_labels = []
    
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        total_loss += loss.item()
        
        predictions = torch.sigmoid(logits) > threshold
        all_predictions.append(predictions.cpu().numpy())
        all_true_labels.append(labels.cpu().numpy())

    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_true_labels = np.concatenate(all_true_labels, axis=0)

    # Calculate accuracy and other metrics
    accuracy = accuracy_score(all_true_labels, all_predictions)
    accuracy_per_label = [accuracy_score(all_true_labels[:, i], all_predictions[:, i]) for i in range(all_true_labels.shape[1])]
    average_label_accuracy = np.mean(accuracy_per_label)

    precision_samples = precision_score(all_true_labels, all_predictions, average='samples', zero_division=0)
    recall_samples = recall_score(all_true_labels, all_predictions, average='samples', zero_division=0)
    f1_samples = f1_score(all_true_labels, all_predictions, average='samples', zero_division=0)

    # Calculating macro averages
    precision_macro = precision_score(all_true_labels, all_predictions, average='macro', zero_division=0)
    recall_macro = recall_score(all_true_labels, all_predictions, average='macro', zero_division=0)
    f1_macro = f1_score(all_true_labels, all_predictions, average='macro', zero_division=0)

    print("\n")
    print(f"Overall Accuracy: {accuracy}")
    print(f"Average Label-wise Accuracy: {average_label_accuracy}")
    print(f"Precision (Samples): {precision_samples}")
    print(f"Recall (Samples): {recall_samples}")
    print(f"F1 Score (Samples): {f1_samples}")
    print(f"Precision (Macro): {precision_macro}")
    print(f"Recall (Macro): {recall_macro}")
    print(f"F1 Score (Macro): {f1_macro}")

    return total_loss / len(val_loader), accuracy, average_label_accuracy, precision_samples, recall_samples, f1_samples, precision_macro, recall_macro, f1_macro


def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def normalize_and_split(text):
    """Normalize and split text into individual words, preserving '%' and removing other punctuation."""
    # Remove punctuation except '%'
    normalized_text = re.sub(r'[^\w\s%]', ' ', text)
    normalized_text = normalized_text.lower()  # Convert to lowercase
    # Split the text into words and remove duplicates
    words = normalized_text.split()
    return list(set(words))

def extract_unique_terms(data):
    """Extract unique terms from given DataFrame columns and their values, return as a dictionary."""
    terms_dict = {}
    unique_terms = set()

    for col in data.columns:
        # Split the column name into words
        terms_dict[col] = normalize_and_split(col)
        unique_terms.update(normalize_and_split(col))

        # Process categorical data
        if data[col].dtype == 'object':  # Ensure we only process textual data
            unique_values = data[col].dropna().unique()
            for value in unique_values:
                value_terms = normalize_and_split(str(value))
                terms_dict[str(value)] = value_terms
                unique_terms.update(value_terms)

    return terms_dict, list(unique_terms)

def label_data(text, terms_to_search):
    """Label the data by checking the presence of terms in the text."""
    labels = {}
    if not isinstance(text, str):
        text = ""
    for term in terms_to_search:
        labels[term] = bool(re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE))
    return labels

def process_descriptions(desc_path, terms_to_search, save_path, limiter=None):
    """Process all descriptions and label them based on the presence of terms."""
    if not os.path.exists(desc_path):
        raise FileNotFoundError(f"Description file not found at {desc_path}")
    desc_data = load_data(desc_path)
    
    if limiter:
        desc_data = desc_data.head(limiter)  # Limit the number of articles processed
    
    labeled_data = []
    with alive_bar(len(desc_data), title='Processing Descriptions') as bar:
        for _, row in desc_data.iterrows():
            # Combine title and article into a single string for processing
            article_text = f"{row.get('title', '')} {row.get('article', '')}".strip()
            labels = label_data(article_text, terms_to_search)
            labeled_data.append(labels)
            bar()  # Update the progress bar
    
    # Convert labeled data to DataFrame and include the combined text for inspection
    labeled_df = pd.DataFrame(labeled_data)
    labeled_df['description'] = desc_data.apply(lambda row: f"{row['title']} {row['article']}".strip(), axis=1)
    
    # Save the processed and labeled data for inspection
    save_path = os.path.join(save_path, 'labeled_data.csv')
    save_to_csv(labeled_df, save_path)
    
    return labeled_df

def save_to_csv(dataframe, file_path):
    """Save the DataFrame to a CSV file."""
    dataframe.to_csv(file_path, index=False)

def log_performance(log_path, epoch, train_loss, val_loss, val_accuracy, val_avg_accuracy, val_precision, val_recall, val_f1, val_precision_macro, val_recall_macro, val_f1_macro):
    with open(log_path, 'a') as f:
        f.write("\n")
        f.write(f"Epoch {epoch + 1}:\n")
        f.write(f"Training Loss: {train_loss}\n")
        f.write(f"Validation Loss: {val_loss}\n")
        f.write("\n")
        f.write(f"Validation Overall Accuracy: {val_accuracy}\n")
        f.write(f"Validation Average Label-wise Accuracy: {val_avg_accuracy}\n")
        f.write(f"Validation Precision: {val_precision}\n")
        f.write(f"Validation Recall: {val_recall}\n")
        f.write(f"Validation F1 Score: {val_f1}\n")
        f.write("\n")
        f.write(f"Validation Macro-Precision: {val_precision_macro}\n")
        f.write(f"Validation Macro-Recall: {val_recall_macro}\n")
        f.write(f"Validation Macro-F1 Score: {val_f1_macro}\n")
        f.write("________________________________")
        f.write("\n")

def main():
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    inspect_dir = os.path.join(current_dir, '../data/processed/inspect')
    dict_dir = os.path.join(current_dir, '../data/raw/misc')
    log_dir = os.path.join(current_dir, '../data/logs')
    
    path_mdl_data = os.path.join(inspect_dir, 'cleaned_synthetic.csv')
    path_articles = os.path.join(dict_dir, 'building_descriptions.csv')
    
    # Load data
    mdl_data = load_data(path_mdl_data)
    
    # Extract terms
    _, terms_to_search = extract_unique_terms(mdl_data)
    
    LIMITER = None

    # Process descriptions and label data
    labeled_df = process_descriptions(path_articles, terms_to_search, current_dir, limiter=LIMITER)
    
    # Convert labeled data to DataFrame
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        labeled_df['description'], labeled_df.drop(columns=['description']), test_size=0.3)

    train_dataset = ArticleDataset(pd.concat([train_texts, train_labels], axis=1), tokenizer)
    val_dataset = ArticleDataset(pd.concat([val_texts, val_labels], axis=1), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Prepare model
    unique_labels_count = labeled_df.drop(columns=['description']).shape[1]
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=unique_labels_count)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Setup the optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Training and validation
    epochs = 6  # Number of training epochs. Adjust as needed.
    for epoch in range(epochs):
        print("-------------------------------")
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Training
        train_loss = train_model(model, train_loader, optimizer, device)
        print(f"Training loss: {train_loss}")
        
        # Validation
        val_loss, val_accuracy, val_avg_accuracy, val_precision, val_recall, val_f1, val_precision_macro, val_recall_macro, val_f1_macro = evaluate_model(model, val_loader, device)

        # Log performance
        log_path = os.path.join(log_dir, 'context_log.txt')
        log_performance(os.path.join(log_dir, 'performance_log.txt'), epoch, train_loss, val_loss, val_accuracy, val_avg_accuracy, val_precision, val_recall, val_f1, val_precision_macro, val_recall_macro, val_f1_macro)
    
    # Save the model and tokenizer
    model_path = os.path.join(current_dir, '../src/model/context')
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

if __name__ == "__main__":
    main()