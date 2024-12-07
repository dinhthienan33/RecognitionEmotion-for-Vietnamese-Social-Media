from transformers import AutoModelForMaskedLM, AutoTokenizer,AutoModelForSequenceClassification, AutoModel
import torch
import torch.nn as nn
from pyvi import ViTokenizer
import warnings
warnings.filterwarnings("ignore")

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModelForMaskedLM.from_pretrained("andt123/VisoBert")
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)
    def forward(self, input_ids, attention_mask):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False # Dropout will errors if without this
        )

        #x = self.drop(output)
        #x = self.fc(x)
        return output
tokenizer =AutoTokenizer.from_pretrained("andt123/VisoBert", use_fast=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = AutoModelForMaskedLM.from_pretrained('andt123/VisoBert').to(device)
model = SentimentClassifier(n_classes=7).to(device)
class_names = ['Enjoyment', 'Disgust', 'Sadness', 'Anger', 'Surprise', 'Fear', 'Other']

def infer(text, tokenizer, max_len=120):
    text = ViTokenizer.tokenize(text)

    encoded_review = tokenizer.encode_plus(
        text,
        max_length=max_len,
        truncation=True,
        add_special_tokens=True,
        padding='max_length',
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='pt',
    )

    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)

    output = model(input_ids, attention_mask)
    _, y_pred = torch.max(output, dim=1)

    return class_names[y_pred]   # Return the predicted class


print(infer('Cảm ơn bạn đã chạy thử model của mình. Chúc một ngày tốt lành nha!', tokenizer))