##  Introduction
- Problem inspiration: why am I doing this? what is the potential impact of such a project? (write very shortly in 3-4 sentences)
- Datasets: 
	- Source, authors
	- Timeframe
	- Size
	- Features:
	- Diversity?
	- Categories
	- Labeling process
	- Reflecting Reality?
- BERTweet (in 1 sentence): https://huggingface.co/vinai/bertweet-base
	- Based on RoBERTA pre-training procedure
	- Trained on 850M English Tweets from 2012-2019
	- Suitable for tweet classification tasks that requires deeper sentence comprehension 
## Exploratory data analysis:
The data is already splitted into train-test-validation sets by the authors. Their intention was for everyone to use a consistent split, making comparison between models made by different people easier. The data split was well-stratified:![[HumAID tweet classification with BERTweet-1777726212548.webp]]

There are minimal features, of which most are within the text. Since the main objective is to use BERT which will only takes the raw text as input, no feature engineering was done.

#### Data Imbalances:
The split is well-stratified, but there is extreme class imbalance, with the most extreme one is the class "missing_or_found_people" with less than 1% of total observation. 
![[HumAID tweet classification with BERTweet-1777725892738.webp]]
## Data Preprocessing:
- For SVM and LR:
	- Stripping stopwords and punctuations and hashtag (but keep the hashtag word)
	- TF-IDF with parameters: ngram_range= (1, 2), sublinear_tf= True
- For BERTweet:
	- The text are kept intact without preprocessing
## Evaluation metrics: 
Macro f1-score was used as the main evaluation metrics due to the problem being multi-class and has great class imbalances. Class-specific f1-score are also looked at, especially ones that performed poorly.
## Baseline modelling:
First, I will use two linear models as the baseline; their performance and their misclassification are then evaluated manually. The baseline models' weaknesses will be the motivation for using BERTweet in the next part.
 **Logistic regressions:** 
- Suitable for multiclass classification
 **Linear Support Vector Machine:**
- Suitable for multiclass classification
- (Vectorized) Text data is sparse, tree models won't perform well
=> Both may struggle to separate non-linear patterns
#### Result
- Logistic Regression F1: 0.6817
- SVM F1: 0.6927
- SVM did a slightly better job than LR in terms of macro f1-score
- LR performed worse for the minority class "missing_or_found_people"
- Both performed poorly for 4 classes: "caution_and_advice", "not_humanitarian", "other_relevant_information", "requests_or_urgent_needs"
#### Error Analysis for SVM
Since SVM has better overall performance, I did not do error analysis on Logistic Regression.
**Confusion Matrix**
![[HumAID tweet classification with BERTweet-1777727543912.webp]]
- "**not_humanitarian**" is mainly misclassified as **"other_relevant_information"**
- "**other_relevant_information**" is mainly misclassified as "**rescue_volunteering_or_donation_effort**", "**infrastructure_and_utility_damage**", and "**caution_and_advice**". 

**Manually inspecting the errors**:
Upon manual inspection of the most commonly misclassified classes, I saw one important pattern leading to these errors: each tweet can contain many types of information: 
- "other_relevant_information" can contains anything related to humanitarian.
- "caution_and_advice" and "requests_or_urgent_needs" normally include information about the disasters such as casualties, infrastructure damage, etc. 
A linear model like LinearSVM may struggle to separate such unclear boundaries. Furthermore, the TF-IDF do not "understand" the entire sentence and merely counting the importance of individual keywords (or pair of keywords). These factors contributed to the misclassification of ambiguous classes.

Therefore, using BERT may improve performance due to its context-understanding and minimal pre-processing required.

## BERTweet Modelling:
- Hardware: GPU T4 x 2 on Kaggle server.
- Implemented using PyTorch
- Model downloaded from HuggingFace
- Tokenized with max_length = 128 since this is enough for tweets
- Evaluation metrics: macro f1-score
- Training configuration: 
	- 4 epochs with early stopping on validation F1
	- Learning rate 2e-5 (conservative to preserve pretrained weights)
	- Warmup over 10% of steps to stabilise early training
	- Batch size 32 for training efficiency on T4 GPU.
#### Result: 

![[HumAID tweet classification with BERTweet-1777730282065.webp]]
The Macro F1 score plateaued after epoch 4 and ended up at **0.7559**
This is significantly higher than the baseline model:
- Logistic Regression F1: 0.6817
- Support Vector Machine F1: 0.6927
Looking at each class's f1-score, there are still some underperforming classes:
- not_humanitarian F1: 0.41
- other_relevant_information: 0.59
- requests_or_urgent_need: 0.61
- caution_and_advice: 0.70
These are the same 4 classes that performed poorly in the baseline models but each class improved significantly. "not_humanitarian" F1 doubled from 0.21, while the others increased more modestly. "other_relevant_information" F1 only improved by 0.02, which may signify that this is a data problem rather than a modelling one.

#### **Error Analysis**:
![[HumAID tweet classification with BERTweet-1777731261954.webp|520x473]]
The four classes with the most errors:
1. **Not_humanitarian** is mostly misclassified as **other_relevant_information**.
2. **Other_relevant_information** is misclassified as a range of different classes.
3. **Requests_or_urgent_needs** is mostly misclassified as **rescue_volunteering_or_donation_effort**.
4. **Caution_and_advice** is mostly misclassified as **other_relevant_information**.

After manually inspecting the errors in these four classes, I found some valuable insights:
- "other_relevant_information": misclassification mainly comes from the ambiguious definition. "Other" can means any information related to humanitarian and does not have an inherent meaning, hence this class does not have clear features that can be separated from other classes. 
-  "requests_or_urgent_needs": misclassification mainly due to confusion with the class "rescue_volunteering_or_donation_effort". In real life, rescue and donation effort is a natural response to urgent need. Tweets asking for rescue or donations can also contain information about the rescue response; hence, the confusion between the 2 classes. 
- "caution_and_advice": in order to raise caution and awareness, these tweets often contains information about different aspects of a disaster: casualties, damage, area affected. These aspects can be confused with other classes.
