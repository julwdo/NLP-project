# Multimodal X Bot Detection

This project replicates and analyzes a multimodal approach to detecting bots on X (formerly Twitter), combining user-level and tweet-level features to classify accounts as human or bot.

## Key Components

- **Dataset and Preprocessing:**  
  Utilized the TwiBot-22 dataset, fully labeled with 1,000,000 users. Features were extracted from user profiles and tweets, including numerical, boolean, and text features. Due to computational constraints, a subset of users and one chunk of tweets (~10 GB) was used.

- **Feature Engineering:**  
  - **User-level features:** 47 numerical/boolean features including account metrics, ratios, growth rates, text characteristics, and sentiment, plus user descriptions (texts, not embeddings).  
  - **Tweet-level features:** Aggregated metrics from recent tweets (e.g., likes, retweets, replies, sensitive content), along with truncated and full tweet texts.

- **Text Embeddings:**  
  User descriptions encoded with `distiluse-base-multilingual-cased-v1`. Tweets encoded with four embedding strategies, varying the model (`distiluse-base-multilingual-cased-v1` or `paraphrase-multilingual-MiniLM-L12-v2`) and aggregation method (concatenation vs mean-pooling).

- **Oversampling:**  
  SMOTE was used to address class imbalance.

- **Neural Network Architecture:**  
  Multi-branch network with separate branches for user features, description texts, and tweet texts. Each branch passes through a linear layer, Leaky ReLU, and dropout before concatenation and final logit output.

- **Results:**  
  - Feature importance highlighted `followers_count`, `tweet_count`, and `has_description` as top predictors.  
  - Best hyperparameters: batch size 32, dropout 0.5, BCE loss with integrated sigmoid, Adam optimizer (lr=1e-3, weight decay=1e-5). Test F1: 0.4372.  
  - Observed discrepancy between validation and test F1 suggests previous studies may have overestimated performance due to information leakage.
