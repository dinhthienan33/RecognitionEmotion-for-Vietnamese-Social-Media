from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
import torch.nn as nn
from pyvi import ViTokenizer
import os
from transformers import logging
import warnings
from huggingface_hub import HfApi, HfFolder

# Suppress warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# Define the model class
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained("uitnlp/visobert")
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        _, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False  # Dropout will throw errors if not disabled
        )
        x = self.drop(output)
        x = self.fc(x)
        return x

# Load tokenizer and device configuration
tokenizer = AutoTokenizer.from_pretrained("uitnlp/visobert", use_fast=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model and load weights
model = SentimentClassifier(n_classes=7).to(device)
state_dict = torch.load('models/visobert_fold6.pth', map_location=device)
model.load_state_dict(state_dict, strict=False)  # `strict=False` ignores unexpected keys

# Save model and tokenizer locally
save_directory = "./visobert_model"
os.makedirs(save_directory, exist_ok=True)

# Save model components
torch.save(model.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))  # Save model weights
tokenizer.save_pretrained(save_directory)  # Save tokenizer
config = AutoConfig.from_pretrained("uitnlp/visobert", num_labels=7)  # Update config if needed
config.save_pretrained(save_directory)

# Push to Hugging Face Hub
from huggingface_hub import HfApi

api = HfApi()
repo_id = "andt123/VisoBert"

# Push files to the hub
api.upload_folder(
    folder_path=save_directory,
    repo_id=repo_id,
    repo_type="model",
    token="hf_YOUR_TOKEN"  # Replace with your Hugging Face token
)

print(f"Model uploaded to https://huggingface.co/{repo_id}")
