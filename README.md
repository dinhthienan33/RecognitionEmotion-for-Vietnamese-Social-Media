# Exploring ViSoBERT and its Application to Emotion Recognition with UIT-VSMEC

This document provides a comprehensive explanation of the ViSoBERT model, introduces the UIT-VSMEC dataset, and illustrates the model's performance in emotion recognition using this dataset.

---

## Understanding ViSoBERT: A Pre-trained Language Model for Vietnamese Social Media

**ViSoBERT**, short for *"Vietnamese Social BERT,"* is a specialized pre-trained language model explicitly designed for processing the nuances of Vietnamese social media text. It addresses the limitations of traditional language models trained on formal text and effectively handles:

- Informal language
- Emojis
- Slang
- Variations in diacritic usage

### Key Features and Advantages of ViSoBERT

- **Built upon XLM-R Architecture**  
  ViSoBERT leverages the robust multilingual capabilities of the XLM-R (Cross-lingual Language Model - RoBERTa) architecture, inheriting its powerful transformer-based design and masked language model pre-training approach.

  ![BERT Architecture](images/bert_diagram.png "BERT Architecture")


- **Custom Tokenizer Tailored to Social Media**  
  A crucial aspect of ViSoBERT's effectiveness is its use of a custom tokenizer built with SentencePiece. This tokenizer is specifically trained on a large corpus of Vietnamese social media text, enabling it to accurately handle emojis, teencode, and variations in diacritic usage.

- **Training on a Massive Social Media Dataset**  
  ViSoBERT is trained exclusively on a vast dataset of Vietnamese social media posts and comments collected from platforms like Facebook, TikTok, and YouTube. This targeted training makes ViSoBERT highly adept at understanding vocabulary, language patterns, and context in social media.

### Demonstrated Effectiveness of ViSoBERT

ViSoBERT has demonstrated superior performance in various Vietnamese social media processing tasks, including:

- Emotion recognition
- Hate speech detection
- Sentiment analysis
- Spam reviews detection
- Hate speech spans detection

It consistently outperforms strong baseline models, including monolingual and multilingual language models.

---

## Introducing UIT-VSMEC: A Benchmark Dataset for Vietnamese Emotion Recognition

**UIT-VSMEC** (*Vietnamese Social Media Emotion Corpus*) is a standardized dataset developed by researchers at the University of Information Technology in Vietnam. It serves as a valuable resource for training and evaluating emotion recognition models specifically designed for Vietnamese social media text.

### Key Characteristics of UIT-VSMEC

- **Emotion Labels**  
  The dataset consists of 6,927 Vietnamese sentences annotated with one of seven emotion labels:  
  `enjoyment`, `sadness`, `anger`, `surprise`, `fear`, `disgust`, and `other` (for neutral or ambiguous emotions).

- **Source of Data**  
  Sentences were collected from Facebook, ensuring the dataset reflects real-world social media communication.

- **Annotation Agreement**  
  To ensure quality and reliability, the annotation process involved multiple annotators and an agreement measure (Am) to assess consensus. The Am agreement for UIT-VSMEC was over **82%**.

---

## Utilizing ViSoBERT for Emotion Recognition on UIT-VSMEC

Given its specialization in Vietnamese social media text, **ViSoBERT** is well-suited for emotion recognition on the UIT-VSMEC dataset.

### Experimental Setup

1. **Corpus Preparation**  
   - The UIT-VSMEC corpus was divided into training, validation, and test sets using stratified sampling to ensure a balanced distribution of emotion labels.

2. **Fine-tuning ViSoBERT**  
   - ViSoBERT was fine-tuned on the UIT-VSMEC training set using the `simpletransformers` library. Standard fine-tuning procedures were followed.

3. **Evaluation Metrics**  
   - Performance was evaluated using **accuracy**, **weighted F1-score**, and **macro F1-score**.

### Results

ViSoBERT achieved the following results on the UIT-VSMEC emotion recognition task:

| Metric          | Score   |
|------------------|---------|
| **Accuracy**     | 66% |
| **Weighted F1**  | 66% |
| **Macro F1**     | 64% |

---

### Significance of the Results

- **State-of-the-Art Performance**  
  ViSoBERT demonstrates its effectiveness in capturing the nuances of emotion expression in Vietnamese social media text.

- **Value of Domain-Specific Training**  
  Its superior performance compared to general-purpose models like PhoBERT and multilingual models like TwHIN-BERT underscores the importance of training on domain-specific data.

---

## Conclusion

ViSoBERT, with its specialized tokenizer and training on a large-scale Vietnamese social media corpus, proves to be a powerful tool for emotion recognition in this domain. Its performance on the UIT-VSMEC dataset showcases its ability to accurately classify emotions expressed in social media text, setting a new benchmark for this task.

## How to Use This Repository

1. **Download Models**  
   Download the model weights from Google Drive and move them to your project directory:  
   [Model Weights](https://drive.google.com/drive/folders/1aoaLvEJSlU6hr2F-bB6ls085CATnckKb)

2. **Clone the Repository**  
   Clone this repository and navigate to the project directory:
   ```bash
   git clone https://github.com/your-repository-name/Classification-for-Vietnamese-Text.git
   cd Classification-for-Vietnamese-Text
3. **Run application**
```bash
streamlit run main.py