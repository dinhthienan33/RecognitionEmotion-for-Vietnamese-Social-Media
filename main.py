############ 1. IMPORTING LIBRARIES ############
import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
from bertviz import head_view
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
from pyvi import ViTokenizer
from dotenv import load_dotenv
import os
import warnings
from groq import Groq

# Suppress warnings and logging
warnings.filterwarnings("ignore")
from transformers import logging
logging.set_verbosity_error()

# Load environment variables
load_dotenv()
viso_model_path = os.getenv('VISO_MODEL_PATH')
phobert_model_path = os.getenv('PHOBERT_MODEL_PATH')
############ MODEL SETUP ############
class EmotionClassifier(nn.Module):
    def __init__(self, model_name, n_classes=7):
        super(EmotionClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        _, output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        x = self.drop(output)
        return self.fc(x)

def initialize_model(model_name, model_path, n_classes=7):
    model = EmotionClassifier(model_name, n_classes).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def infer(text, tokenizer, model, max_len=120):
    text = ViTokenizer.tokenize(text)
    encoded = tokenizer.encode_plus(
        text, max_length=max_len, truncation=True, add_special_tokens=True,
        padding='max_length', return_attention_mask=True, return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    output = model(input_ids, attention_mask)
    _, y_pred = torch.max(output, dim=1)
    return class_names[y_pred]    
def llm_infer(query):
    prompt= f"""
    Hãy phân loại câu dưới đây thành 1 trong 7 lớp sau : 'Enjoyment', 'Disgust', 'Sadness', 'Anger', 'Surprise', 'Fear', 'Other'
    Chỉ trả lời bằng 1 trong 7 lớp trên.Không trả lời bằng câu hỏi hoặc câu trả lời không liên quan.
    Query:
    {query}
    """
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",
    )

    return chat_completion.choices[0].message.content
# Initialize device and class names
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_names = ['Enjoyment', 'Disgust', 'Sadness', 'Anger', 'Surprise', 'Fear', 'Other']

############ 2. STREAMLIT SETUP ############
st.set_page_config(
    layout="centered", page_title="Vietnamese Emotion Classifier", page_icon="❄️"
)

# Logo and heading
c1, c2 = st.columns([1, 3])
with c1:
    st.image("images/logouit.webp", width=150)
with c2:
    st.title("Phân loại cảm xúc tiếng Việt trên mạng xã hội")

############ TABBED NAVIGATION ############
MainTab, VisualizeTab, InfoTab = st.tabs(["Main", "Visualize Attention", "Info"])

with InfoTab:
    st.subheader("Đồ án môn NLP_CS221.P12")
    st.markdown("[GitHub Repository](https://github.com/dinhthienan33/Classification-for-Vietnamese-Text)")
    st.subheader("Thành viên")
    st.markdown("""
    - [Lê Trần Gia Bảo](MSSV: 22520105)
    - [Đinh Thiên Ân](MSSV: 22520010)
    - [Huỳnh Trọng Nghĩa](MSSV: 22520003)
    - [Nguyễn Vũ Khai Tâm](MSSV: 22521293)
    """)

with MainTab:
    st.markdown("""
    Ứng dụng phân loại câu thành 1 trong 7 cảm xúc: Enjoyment, Disgust, Sadness, Anger, Surprise, Fear, Other.
    """)
    genre = st.radio("Choose model", ["VisoBert", "PhoBert", "LLM"], index=0)
    st.write("You selected:", genre)

    # Load the selected model and tokenizer
    if genre == "VisoBert":
        tokenizer = AutoTokenizer.from_pretrained("5CD-AI/Vietnamese-Sentiment-visobert",output_attentions=True)
        model = initialize_model("5CD-AI/Vietnamese-Sentiment-visobert", viso_model_path)        
    elif genre == "PhoBert":
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base",output_attentions=True)
        model = initialize_model("vinai/phobert-base", phobert_model_path)
    else :
        client = Groq(
            api_key=os.getenv("GROQ_API_KEY"),
        )

    with st.form(key="classifier_form"):
        pre_defined_keyphrases = [
            "mỗi lần coi lại sởn gai ốc",
            "ngưỡng mộ ông bà á . ông bà đáng yêu ghê",
            "nghe đi rồi khóc 1 trận cho thoải mái . đừng cố gồng mình lên nữa"
        ]
        text = st.text_area(
            "Enter keyphrases to classify", "\n".join(pre_defined_keyphrases),
            height=200, help="Enter one keyphrase per line (max 50)."
        )
        submit_button = st.form_submit_button("Submit")

    if submit_button:
        lines = list(filter(None, map(str.strip, text.split("\n"))))[:50]
        if not lines:
            st.warning("❄️ No keyphrases provided!")
        else:
            if(genre == "LLM"):
                results = [{"Phrase": phrase, "Predicted Class": llm_infer(phrase)} for phrase in lines]
            else:
                results = [{"Phrase": phrase, "Predicted Class": infer(phrase, tokenizer, model)} for phrase in lines]
            df = pd.DataFrame(results)
            st.success("✅ Classification Completed!")
            st.write(df)

            # Download results
            @st.cache_data
            def convert_df(dataframe):
                return dataframe.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Results as CSV", convert_df(df),
                file_name="classification_results.csv", mime="text/csv"
            )
    ############ SIDEBAR HISTORY ############
    st.sidebar.header("Lịch sử truy vấn")
    if "history" not in st.session_state:
        st.session_state.history = []

    if submit_button:
        st.session_state.history.extend(lines)
        for i, entry in enumerate(st.session_state.history[::-1], 1):  # Reverse the list to show recent first
            st.sidebar.markdown(f"**{i}.** {entry}")
    if st.sidebar.button("Clear History"):
        st.session_state.history = []
        st.sidebar.success("History cleared.")