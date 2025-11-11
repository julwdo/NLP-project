# Multimodal X Bot Detection

This project replicates and analyzes a multimodal approach to detecting bots on X (formerly Twitter), combining user-level and tweet-level features to classify accounts as human or bot.

## Key Components

- **Dataset and Preprocessing:**  
  Utilized the TwiBot-22 dataset, fully labeled with 1,000,000 users. Features were extracted from user profiles and tweets, including numerical, boolean, and text features. Due to computational constraints, a subset of users and one chunk of tweets (~10 GB) was used.

- **Feature Engineering:**  
  - **User-level features:** 47 numerical/boolean features including account metrics, ratios, growth rates, text characteristics, and sentiment, plus user descriptions.   
  - **Tweet-level features:** Aggregated metrics from recent tweets (e.g., likes, retweets, replies), along with truncated and non-truncated tweet texts.  

- **Text Embeddings:**  
  User descriptions encoded with `distiluse-base-multilingual-cased-v1`. Tweets encoded with four different embedding strategies, varying the model (`distiluse-base-multilingual-cased-v1` or `paraphrase-multilingual-MiniLM-L12-v2`) and aggregation method (concatenation vs mean-pooling).

- **Neural Network Architecture:**  
  Multi-branch network with separate branches for user features, description embeddings, and tweet embeddings. Each branch passes through a linear layer, Leaky ReLU, and dropout before concatenation and final logit output.

- **Training and Evaluation:**  
  - Batch size: 32, Dropout: 0.5, BCE loss with integrated sigmoid, Adam optimizer (lr=1e-3, weight decay=1e-5)  
  - Oversampling using SMOTE to address class imbalance  
  - Best embedding variant selected based on F1 score  
  - Hyperparameter tuning via 5-fold cross-validation

- **Results:**  
  - Feature importance highlighted `followers_count`, `tweet_count`, and `has_description` as top predictors.  
  - Best F1 score on the test set: 0.4372.  
  - Observed discrepancy between validation and test F1 suggests previous studies may have overestimated performance due to information leakage.

- **Conclusion:**  
  The multimodal approach is effective in detecting bots, combining numerical, textual, and network data. Proper data splitting and unbiased evaluation are crucial for realistic performance assessment.
