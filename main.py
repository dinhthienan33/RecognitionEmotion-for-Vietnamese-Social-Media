############ 1. IMPORTING LIBRARIES ############

import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from pyvi import ViTokenizer
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModel, logging

import warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error()
############ MODEL SETUP ############
class VisoBert(nn.Module):
    def __init__(self, n_classes):
        super(VisoBert, self).__init__()
        self.bert = AutoModel.from_pretrained("uitnlp/visobert")
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)
    def forward(self, input_ids, attention_mask):
        last_hidden_state, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False # Dropout will errors if without this
        )

        x = self.drop(output)
        x = self.fc(x)
        return x

tokenizer = AutoTokenizer.from_pretrained("uitnlp/visobert")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VisoBert(n_classes=7).to(device)
state_dict = torch.load('models/visobert_fold6.pth', map_location=device)
model.load_state_dict(state_dict, strict=False)  # `strict=False` ignores unexpected keys
model.eval()

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

    return class_names[y_pred]  # Return the predicted class


############ 2. SETTING UP THE PAGE LAYOUT AND TITLE ############

st.set_page_config(
    layout="centered", page_title="Zero-Shot Text Classifier on VietNamese social network comments", page_icon="❄️"
)

############ CREATE THE LOGO AND HEADING ############

c1, c2 = st.columns([0.32, 2])

with c1:
    st.image("images/logo.png", width=85)

with c2:
    st.caption("")
    st.title("Zero-Shot Text Classifier")

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

############ TABBED NAVIGATION ############

MainTab, InfoTab = st.tabs(["Main", "Info"])

with InfoTab:
    st.subheader("Đồ án môn NLP_CS221.P12")
    st.markdown(
        "[Link github ](https://github.com/) "
    )
    st.subheader("Thành viên")
    st.markdown(
        """
    - [Lê Trần Gia Bảo ](https://docs.streamlit.io/)
    - [Đinh Thiên Ân ](https://docs.streamlit.io/library/cheatsheet)
    - [Huỳnh Trọng Nghĩa](https://www.amazon.com/dp/180056550X) (Getting Started with Streamlit for Data Science)
    """
    )
with MainTab:x  
    st.write("")
    st.markdown(
        """
    Classify keyphrases on the fly with this mighty app. No training needed!
    """
    )
    genre = st.radio(
    "What's your favorite movie genre",
    ["VisoBert", "PhoBert"],
    index=0,
)

    st.write("You selected:", genre)
    st.write("")

    with st.form(key="my_form"):
        MAX_KEY_PHRASES = 50
        new_line = "\n"

        pre_defined_keyphrases = [
            "lo học đi . yêu đương lol gì hay lại thích học sinh học",
            "uớc gì sau này về già vẫn có thể như cụ này :))",
            "per nghe đi rồi khóc 1 trận cho thoải mái . đừng cố gồng mình lên nữa"
        ]

        keyphrases_string = f"{new_line.join(map(str, pre_defined_keyphrases))}"

        text = st.text_area(
            "Enter keyphrases to classify",
            keyphrases_string,
            height=200,
            help=f"At least two keyphrases for the classifier to work, one per line, {MAX_KEY_PHRASES} keyphrases max.",
            key="1",
        )

        text = text.split("\n")
        linesList = list(dict.fromkeys(filter(None, text)))

        if len(linesList) > MAX_KEY_PHRASES:
            st.info(
                f"❄️ Only the first {MAX_KEY_PHRASES} keyphrases will be reviewed to preserve performance."
            )
            linesList = linesList[:MAX_KEY_PHRASES]

        submit_button = st.form_submit_button(label="Submit")

    if not submit_button and not st.session_state.valid_inputs_received:
        st.stop()

    elif submit_button and not linesList:
        st.warning("❄️ There is no keyphrase to classify")
        st.session_state.valid_inputs_received = False
        st.stop()

    elif submit_button or st.session_state.valid_inputs_received:

        if submit_button:
            st.session_state.valid_inputs_received = True
        if(genre == 'PhoBert'):
            tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
            class PhoBert(nn.Module):
                def __init__(self, n_classes):
                    super(PhoBert, self).__init__()
                    self.bert = AutoModel.from_pretrained("vinai/phobert-base")
                    self.drop = nn.Dropout(p=0.3)
                    self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
                    nn.init.normal_(self.fc.weight, std=0.02)
                    nn.init.normal_(self.fc.bias, 0)

                def forward(self, input_ids, attention_mask):
                    last_hidden_state, output = self.bert(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=False # Dropout will errors if without this
                    )

                    x = self.drop(output)
                    x = self.fc(x)
                    return x
            model = PhoBert(n_classes=7).to(device)
            state_dict = torch.load('models\phobert_fold1.pth', map_location=device)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
        
        results = []
        for phrase in linesList:
            sentiment = infer(phrase, tokenizer)
            results.append({"Phrase": phrase, "Predicted Class": sentiment})

        df = pd.DataFrame(results)

        st.success("✅ Classification Completed!")
        st.write(df)
    ############ CREATE SIDEBAR FOR HISTORY ############
    # Initialize the session state to track history if it doesn't already exist
    if "history" not in st.session_state:
        st.session_state.history = []

    # Sidebar layout
    st.sidebar.header("Lịch sử truy vấn")
    st.sidebar.write("Lịch sử truy vấn của người dùng sẽ được lưu ở đây.")

    # Display history in the sidebar
    if st.session_state.history:
        for i, entry in enumerate(st.session_state.history[::-1], 1):  # Reverse the list to show recent first
            st.sidebar.markdown(f"**{i}.** {entry}")
    else:
        st.sidebar.write("Chưa có gì hết.")

    # Clear history button
    if st.sidebar.button("Clear History"):
        st.session_state.history = []  # Reset the history
        st.sidebar.success("History cleared.")

    ############ RECORD INPUTS TO HISTORY ############
    # After processing inputs, add them to history
    if submit_button or st.session_state.valid_inputs_received:
        if submit_button:  # Add new inputs only when submitted
            st.session_state.history.extend(linesList)

        @st.cache_data
        def convert_df(dataframe):
            return dataframe.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Results as CSV",
            data=convert_df(df),
            file_name="classification_results.csv",
            mime="text/csv",
        )
