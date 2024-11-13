import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

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
        """
        documents = []
        if os.path.isdir(file_path):
            # Load each file's content as a separate document
            for filename in os.listdir(file_path):
                full_path = os.path.join(file_path, filename)
                with open(full_path, 'r', encoding='utf-8') as f:
                    documents.append(f.read())
        elif os.path.isfile(file_path):
            # Load each line as a separate document
            with open(file_path, 'r', encoding='utf-8') as f:
                documents = f.readlines()
        else:
            raise ValueError(f"The path '{file_path}' is not a valid file or directory.")
        
        return [doc.strip() for doc in documents if doc.strip()]  # Remove empty lines

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
            Tuple[List[int], List[float]]: Predicted category indices and their confidence scores.
        """
        documents = self.load_documents(file_path)
        inputs = self.preprocess(documents)
        
        # Ensure model is on CPU for inference
        self.model.to(self.device)
        outputs = self.model(**inputs)  # Ensure inputs are on CPU
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()  # Ensure output is on CPU
        confidence_scores = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        
        return predictions.tolist(), confidence_scores.tolist()

    def embed_documents(self, documents):
        """Get embeddings for documents from the model."""
        inputs = self.preprocess(documents)
        
        # Ensure model is on CPU for embedding
        self.model.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()  # Use last hidden state and ensure it's on CPU
          
        return embeddings

# Sample Usage
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
    preds, confs = classifier.predict(new_docs_path)

    print("Predictions:", preds)
    print("Confidence Scores:", confs)
