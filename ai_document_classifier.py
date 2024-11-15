import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

class CustomDataset(torch.utils.data.Dataset):
    """Custom dataset class for handling encodings and labels."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx]).to("cpu")  # Force labels to CPU
        return item

    def __len__(self):
        return len(self.labels)

class DocumentClassifier:
    def __init__(self, model_name="distilbert-base-uncased", initial_labels=None):
        # Force CPU usage
        self.device = torch.device("cpu")
        self.model_name = model_name
        # Ensure model is loaded on the CPU
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(initial_labels)).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.categories = initial_labels if initial_labels else []

    def preprocess(self, documents):
        """Tokenize documents for model input."""
        encodings = self.tokenizer(documents, padding=True, truncation=True, return_tensors="pt")
        return {key: tensor.to(self.device) for key, tensor in encodings.items()}  # Ensure tensors are on CPU

    def load_documents(self, file_path):
        """Load documents from a directory or a single file.
        
        Args:
            file_path (str): Path to a file or directory of files.
        
        Returns:
            List[str]: A list of document texts.
            List[str]: A list of document names.
        """
        documents = []
        doc_names = []
        if os.path.isdir(file_path):
            # Load each file's content as a separate document
            for filename in os.listdir(file_path):
                full_path = os.path.join(file_path, filename)
                with open(full_path, 'r', encoding='utf-8') as f:
                    documents.append(f.read())
                    doc_names.append(filename)
        elif os.path.isfile(file_path):
            # Load each line as a separate document
            with open(file_path, 'r', encoding='utf-8') as f:
                documents = f.readlines()
                doc_names = [f"Line {i+1}" for i in range(len(documents))]
        else:
            raise ValueError(f"The path '{file_path}' is not a valid file or directory.")
        
        return [doc.strip() for doc in documents if doc.strip()], doc_names  # Remove empty lines

    def train(self, train_texts, train_labels):
        """Fine-tune the classifier model."""
        train_encodings = self.preprocess(train_texts)
        
        # Create the dataset
        train_dataset = CustomDataset(train_encodings, train_labels)
        
        # Trainer arguments
        training_args = TrainingArguments(
            output_dir="./results", 
            num_train_epochs=3,
            per_device_train_batch_size=4,
            logging_dir='./logs',  
            logging_steps=10  
        )
        
        # Trainer initialization
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset
        )
        
        # Training
        trainer.train()

    def predict(self, file_path):
        """Predict categories for documents loaded from files.
        
        Args:
            file_path (str): Path to a file or directory of files.
        
        Returns:
            List[int]: Predicted category indices.
            List[float]: Confidence scores.
            List[str]: Document names.
        """
        documents, doc_names = self.load_documents(file_path)
        inputs = self.preprocess(documents)
        
        # Ensure model is on CPU for inference
        self.model.to(self.device)
        outputs = self.model(**inputs)  # Ensure inputs are on CPU
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()  # Ensure output is on CPU
        predictions = probs.argmax(axis=1)
        
        return predictions.tolist(), doc_names

    def visualize_predictions(self, predictions, doc_names):
        """Visualize the number of documents per category as a bar chart with document names."""
        # Group documents by category
        category_to_docs = defaultdict(list)
        for pred, doc_name in zip(predictions, doc_names):
            category_to_docs[self.categories[pred]].append(doc_name)
        
        # Prepare data for plotting
        categories = list(category_to_docs.keys())
        counts = [len(category_to_docs[cat]) for cat in categories]
        doc_lists = ["\n".join(category_to_docs[cat]) for cat in categories]

        # Plotting
        plt.figure(figsize=(12, 7))
        bars = plt.bar(categories, counts, color='skyblue')
        
        # Annotate bars with document names
        for bar, doc_list in zip(bars, doc_lists):
            plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), doc_list,
                     ha='center', va='bottom', fontsize=9, rotation=90)
        
        plt.xlabel('Categories')
        plt.ylabel('Number of Documents')
        plt.title('Document Count per Category with Document Names')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

# Main Code
if __name__ == "__main__":
    # Adding new categories to initial labels
    initial_labels = ["Finance", "Healthcare", "Technology", "Application", "Business"]
    classifier = DocumentClassifier(initial_labels=initial_labels)

    # Initial training data
    initial_docs = [
        "Financial report Q1", "Healthcare policy update", "Tech innovation 2021",
        "Application development trends", "Business strategy for growth"
    ]
    initial_labels_numeric = [0, 1, 2, 3, 4]
    
    # Train the classifier on initial data
    classifier.train(initial_docs, initial_labels_numeric)

    # Path to new documents for classification
    new_docs_path = "test"  # Replace with your file or directory path

    # Predict categories for documents in the file or directory
    preds, doc_names = classifier.predict(new_docs_path)

    # Visualize the document counts per category with document names
    classifier.visualize_predictions(preds, doc_names)
