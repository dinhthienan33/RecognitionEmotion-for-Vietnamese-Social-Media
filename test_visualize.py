import streamlit as st
from bertviz import head_view, model_view
import torch
from transformers import AutoTokenizer, AutoModel
import os 
import uuid  # Import the uuid module
tokenizer = AutoTokenizer.from_pretrained("5CD-AI/Vietnamese-Sentiment-visobert")
model = AutoModel.from_pretrained("5CD-AI/Vietnamese-Sentiment-visobert",output_attentions=True)

# Load pre-trained BERT model and tokenizer


# Streamlit app
st.title("BERT Attention Head Visualization")

# Input text
text = st.text_area("Enter text to visualize BERT attention heads:", "The quick brown fox jumps over the lazy dog.")

# Tokenize input text
inputs = tokenizer.encode_plus(text, return_tensors='pt', add_special_tokens=True)
input_ids = inputs['input_ids']
attention = model(input_ids)[-1]  # Get attention weights
# Convert token ids to tokens
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# Visualize attention heads
# Generate visualization using bertviz
def generate_head_view(attention, tokens):
    attn_data = [{
        'name': None,
        'attn': attention.tolist(),
        'left_text': tokens,
        'right_text': tokens
    }]

    # Generate unique div id to enable multiple visualizations in one notebook
    vis_id = 'bertviz-%s' % (uuid.uuid4().hex)

    # Compose HTML
    vis_html = f"""
        <div id="{vis_id}" style="font-family:'Helvetica Neue', Helvetica, Arial, sans-serif;">
            <span style="user-select:none">
                Layer: <select id="layer"></select>
            </span>
            <div id='vis'></div>
        </div>
    """

    # Prepare parameters for JavaScript
    params = {
        'attention': attn_data,
        'default_filter': "0",
        'root_div_id': vis_id,
        'layer': 0,
        'heads': None,
        'include_layers': list(range(len(attention)))
    }

    # Load JavaScript for visualization
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    vis_js = open(os.path.join(__location__, 'head_view.js')).read().replace("PYTHON_PARAMS", json.dumps(params))

    # Combine HTML and JavaScript
    full_html = f"""
        <script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"></script>
        {vis_html}
        <script type="text/javascript">
            {vis_js}
        </script>
    """

    return full_html

# Generate and display the visualization
html_content = generate_head_view(attention, tokens)
st.components.v1.html(html_content, height=800, scrolling=True)