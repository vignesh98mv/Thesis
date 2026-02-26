import os
import sys

# Disable TensorFlow completely
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_TENSORFLOW'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*TensorFlow.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*tensorflow.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Force transformers to use PyTorch
os.environ['USE_TORCH'] = '1'

# Import numpy first to avoid conflicts
import numpy as np

print(f"Using NumPy version: {np.__version__}")

# ==============================================
# Train BERT model to classify flight phases
# ==============================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
import torch
import joblib

# -----------------------------
# 1. Load your data
# -----------------------------

df = pd.read_csv(r"D:\FlightPhases_Better_Approach_Cruise.csv", encoding= 'cp1252')
print("Sample data:\n", df.head())
print(f"Dataset shape: {df.shape}")
print("Label distribution:\n", df["label"].value_counts(), "\n")

# -----------------------------
# 2. Encode labels
# -----------------------------
label_encoder = LabelEncoder()
df["label_id"] = label_encoder.fit_transform(df["label"])

print("Label classes:", list(label_encoder.classes_))
print("Label mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# -----------------------------
# 3. Prepare dataset (Change the path accordingly to the bert tokenizer)
# -----------------------------
tokenizer = BertTokenizer.from_pretrained(r"C:\CRV\bert-offline-install\models\bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=128)

dataset = Dataset.from_pandas(df[["text", "label_id"]])
dataset = dataset.train_test_split(test_size=0.2, seed=42)
dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label_id", "labels")
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

print(f"Training samples: {len(dataset['train'])}")
print(f"Test samples: {len(dataset['test'])}")

# -----------------------------
# 4. Load model
# -----------------------------
model = BertForSequenceClassification.from_pretrained(
    r"C:\CRV\bert-offline-install\models\bert-base-uncased",
    num_labels=len(label_encoder.classes_)
)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

# -----------------------------
# 5. Training setup
# -----------------------------
training_args = TrainingArguments(
    output_dir="./flight_phase_results",
    eval_strategy="epoch",  # Changed from evaluation_strategy to eval_strategy
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./flight_phase_logs",
    logging_steps=10,
    save_total_limit=2,
    report_to=None  # Disable wandb/etc if not configured
)

# Custom function to compute metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# -----------------------------
# 6. Train
# -----------------------------
print("\nStarting training...")
trainer.train()

# -----------------------------
# 7. Evaluate on test set
# -----------------------------
print("\nEvaluating model on test set...")
test_metrics = trainer.evaluate()
print("Test set metrics:", test_metrics)

# -----------------------------
# 8. Calculate detailed metrics
# -----------------------------
print("\nCalculating detailed performance...")

# Training set predictions
train_predictions = trainer.predict(dataset["train"])
train_preds = np.argmax(train_predictions.predictions, axis=1)
train_labels = train_predictions.label_ids
train_accuracy = accuracy_score(train_labels, train_preds)

# Test set predictions
test_predictions = trainer.predict(dataset["test"])
test_preds = np.argmax(test_predictions.predictions, axis=1)
test_labels = test_predictions.label_ids
test_accuracy = accuracy_score(test_labels, test_preds)

print("\n" + "="*60)
print("DETAILED PERFORMANCE REPORT")
print("="*60)
print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

print("\nClassification Report (Test Set):")
print(classification_report(test_labels, test_preds, 
                          target_names=label_encoder.classes_))

# -----------------------------
# 9. Save everything
# -----------------------------
model.save_pretrained("./flight_phase_model")
tokenizer.save_pretrained("./flight_phase_model")
joblib.dump(label_encoder, "./flight_phase_model/label_encoder.pkl")

print("\n✅ Model saved to './flight_phase_model'")

# -----------------------------
# 10. Inference function
# -----------------------------
def predict_phase(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move to device
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class_id = outputs.logits.argmax().item()
    
    return label_encoder.inverse_transform([predicted_class_id])[0]

# -----------------------------
# 11. Test predictions
# -----------------------------
print("\n" + "="*40)
print("TEST PREDICTIONS")
print("="*40)

test_sentences = [
    "v1, rotate",
    "prepare for landing",
    "boarding completed",
    "descending to 10000 feet"
]

for sentence in test_sentences:
    predicted_phase = predict_phase(sentence)
    print(f"Input: '{sentence}' -> Predicted: {predicted_phase}")

# -----------------------------
# 12. Final summary
# -----------------------------
print("\n" + "="*60)
print("FINAL PERFORMANCE SUMMARY")
print("="*60)
print(f"Training samples: {len(dataset['train'])}")
print(f"Test samples: {len(dataset['test'])}")
print(f"Number of classes: {len(label_encoder.classes_)}")
print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Model saved to: ./flight_phase_model")
print("="*60)