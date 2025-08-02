# LLM_FINETUNING_GPT2_SENTIMENT_CLASSIFIER
# GPT-2 Sentiment Extraction from Tweets

This project demonstrates how to fine-tune the GPT-2 model for **sequence classification** using the [Tweet Sentiment Extraction dataset](https://huggingface.co/datasets/mteb/tweet_sentiment_extraction) from Hugging Face 🤗.

## 🔍 Objective

To extract sentiments from tweets (e.g., positive, negative, neutral) by fine-tuning a pre-trained GPT-2 model on a real-world sentiment classification dataset.

## 🚀 Technologies Used

- **Hugging Face Transformers**
- **Datasets (Hugging Face)**
- **Evaluate (for accuracy metrics)**
- **GPT-2 Model**
- **Google Colab / Jupyter Notebook**

## 📦 Dataset

- Dataset: [`mteb/tweet_sentiment_extraction`](https://huggingface.co/datasets/mteb/tweet_sentiment_extraction)
- Contains tweets and corresponding sentiment labels.

## 🧠 Model

- Pre-trained **GPT-2** model is fine-tuned for a **sequence classification** task with 3 output labels (positive, negative, neutral).
- Tokenizer: `GPT2TokenizerFast`
- Model: `GPT2ForSequenceClassification`

## ⚙️ How to Run

1. Clone the repo:
    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Run the training script:
    ```bash
    python train_gpt2_sentiment.py
    ```

## 📈 Results

- The model is trained on a small subset (1000 samples) for demonstration purposes.
- Accuracy is computed using the `evaluate` library.

## 📌 Note

Since GPT-2 was originally trained for text generation, using it for classification may require more layers and training time. For production use or higher accuracy, consider models like BERT or RoBERTa.

---

📬 For any issues or suggestions, feel free to raise an issue or open a pull request!
