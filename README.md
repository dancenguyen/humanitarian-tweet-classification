##  Introduction
### Problem Inspiration:
During humanitarian crises, social media platforms become critical communication channels; however, the sheer volume of posts makes it nearly impossible to manually identify which messages signal genuine emergencies. This project explores automated classification of disaster-related tweets across 10 categories, ranging from rescue requests to infrastructure damage. The goal is to build a system that not only classifies tweets accurately but knows when it is uncertain: flagging ambiguous predictions for human review rather than forcing a confident answer.

### Datasets: 
This project uses the [HumAID](https://arxiv.org/abs/2104.03090) dataset, the tweets span across 2016-2019 and 13 different major disasters, containing 42K labelled English tweets of 10 classes: 
	- Caution and advice
	- Sympathy and support
	- Requests or urgent needs
	- Displaced people and evacuations
	- Injured or dead people
	- Missing or found people
	- Infrastructure and utility damage
	- Rescue, volunteering, or donation effort 
	- Other relevant information
	- Not humanitarian

The data is already splitted into train-test-validation sets by the authors. Their intention was for everyone to use a consistent split, making comparison between models of different people easier. The data split was well-balanced:
<img width="531" height="390" alt="HumAID tweet classification with BERTweet-1777726212548" src="https://github.com/user-attachments/assets/2cc96fc8-d6c0-4f9a-83f3-10837f61178a" />

There are only three columns: tweet_id, tweet_text, and the label of the tweet. While many features can be extracted from the tweet text, no feature engineering was done, as the main objective is implementing BERT which will only takes the raw text as input.
#### Data Imbalances:
The classes are well-stratified between the train, test, and validation set. However, there are some minority classes, with "missing_or_found_people" being the most extreme minority - accounting for only ~0.5% of the dataset.
<img width="989" height="490" alt="HumAID tweet classification with BERTweet-1777725892738" src="https://github.com/user-attachments/assets/4523f47c-328e-4f00-9e73-1b2f8d4a802b" />

### BERTweet: 
[BERTweet is the first large-scale language model](https://aclanthology.org/2020.emnlp-demos.2/) pretrained specifically on English tweets, trained on 850 million tweets spanning 2012 to 2019. Unlike generic BERT models pretrained on formal text like Wikipedia, BERTweet understands Twitter-specific language patterns including slang, informal abbreviations, and hashtags. This makes it a natural fit for disaster tweet classification, where the informal tweet language would disadvantage models pretrained on formal corpora."

## Data Preprocessing:
- For SVM and Logistic Regression:
	- Stripping stopwords, punctuations, and hashtag symbol. Negation words ("no", "not") are kept
	- TF-IDF with parameters: ngram_range= (1, 2), sublinear_tf= True
- For BERTweet:
	- The text are normalised using [BERTweet's normalizer](https://github.com/VinAIResearch/BERTweet#-normalize-raw-input-tweets). This makes the texts consistent with BERTweet's tokenization. These changes are purely syntactic and preserve all original meanings. For example: "cannot" into "can not", and "…" into "..."
## Evaluation metrics: 
Macro f1-score was used as the main evaluation metrics as we have 10 different classes with extreme class imbalance. For underperforming classes, their individual f1-score are examined more closely. 
## Baseline modelling:
First, two linear models are used as the baseline:
 **Logistic regressions:** 
- Suitable for multiclass classification
 **Linear Support Vector Machine:**
- Suitable for multiclass classification
- (Vectorized) Text data is sparse, tree models won't perform well
The misclassifications of the two models are then examined manually, revealing their weaknesses which will be the motivation for using BERTweet in the next part.
### Result
<img width="489" height="390" alt="download" src="https://github.com/user-attachments/assets/cf10c6f8-4eb4-439d-a171-8c7b35e1ba8f" />
<img width="785" height="490" alt="download" src="https://github.com/user-attachments/assets/a0eff745-8761-458a-b450-eaf7cd4b88f8" />
SVM did a slightly better job than LR in terms of macro f1-score, and LR performed worse for the minority class "missing_or_found_people". However, both models performed poorly for 4 classes: "caution_and_advice", "not_humanitarian", "other_relevant_information", "requests_or_urgent_needs"

### Detailed Error Analysis for SVM
Since SVM has better overall performance, error analyses are only done on SVM and not on Logistic Regression. 

**Confusion Matrix**
<img width="1059" height="963" alt="HumAID tweet classification with BERTweet-1777727543912" src="https://github.com/user-attachments/assets/d66e13fb-0cf4-4f47-8da4-6e11c390011f" />

- "**not_humanitarian**" is mainly misclassified as **"other_relevant_information"**
- "**other_relevant_information**" is mainly misclassified as "**rescue_volunteering_or_donation_effort**", "**infrastructure_and_utility_damage**", and "**caution_and_advice**". 

**Manually inspecting the errors**:
Upon manual inspection of the most commonly misclassified classes, we can see one important pattern : these classes are ambiguous by definition and contain information about other
 classes. 
- "other_relevant_information" can contains anything related to humanitarian.
- "caution_and_advice" and "requests_or_urgent_needs" normally include information about the disasters such as casualties, infrastructure damage, etc.

Such information confuses the classifier. For example, a misclassified "other_relevant_information" tweet is: "rural area small villages roads challenges rescue workers face amatrice" (filler words removed). It was misclassified as rescue efforts which make sense since it contain keywords about rescue workers. 

The TF-IDF vectorizer does not "understand" the entire sentence, it strips individual words from their context and count how many times they appeared. Furthermore, the preprocessing may have removed important filler words, the above sentence can be read very differently if the filler words were not removed. Therefore, using BERTweet may improve performance due to its context comprehension and minimal pre-processing required. 

## BERTweet Implementation:
- Model: BERTweet from Hugging Face
- Framework: PyTorch with the Hugging Face Transformers library
- Hardware: 2× NVIDIA T4 GPUs on Kaggle
- Maximum sequence length: 128 tokens (sufficient for tweet-length inputs)
- Evaluation metric: Macro F1-score
- **Training configuration: **
	- 4 epochs with early stopping on validation F1
	- Learning rate 2e-5 (conservative to preserve pretrained weights)
	- Warmup over 10% of steps to stabilise early training
	- Batch size 32 for training efficiency on T4 GPU.
   
### Result: 

<img width="576" height="455" alt="download" src="https://github.com/user-attachments/assets/a36cfcdb-0522-4751-af18-2f8dc4286237" />

Peaking at epoch 3, BERTweet achieved a validation Macro F1-score of 0.7625 and a test Macro F1-score of 0.7636. The similar validation and test performance suggests limited overfitting.

Compared to the SVM baseline (Macro F1 = 0.69), BERTweet achieved an improvement of approximately 10.7%.

*Insert graph comparing the F1 Score*

BERTweet outperformed the SVM model across all individual classes:

<img width="785" height="490" alt="download" src="https://github.com/user-attachments/assets/f2ee0c9e-115e-4852-8de4-03a09fc4dba4" />

1. **"Not_humanitarian"** remained the weakest-performing class, although its F1-score nearly doubled relative to SVM.
2. **"Missing_or_found_people"** improved by roughly 10%
3. **"Requests_or_urgent_needs", and "Caution_and_advice"** improved by roughly 15%
4. **"Other_relevant_information"** showed only marginal improvement, signifying a data problem rather than a modelling problem: the class is highly broad and semantically ambiguous.

BERTweet also outperformed BERT, RoBERTa, and DistilBERT reported in the original dataset paper in terms of Weighted F1-score. Macro F1-score is not compared since the paper did not report this score.

### **Error Analysis**:  Four classes with the most misclassification:

<img width="1059" height="963" alt="download" src="https://github.com/user-attachments/assets/d3ef1cb4-0487-4500-b058-1440cb73dd6e" />

1. **Not_humanitarian** is mostly misclassified as **other_relevant_information**.
2. **Other_relevant_information** is misclassified across multiple classes.
3. **Requests_or_urgent_needs** is most frequently misclassified as **rescue_volunteering_or_donation_effort**.
4. **Caution_and_advice** is most frequently misclassified as **other_relevant_information**.

These patterns are largely explained by semantic ambiguity between classes. In particular, other_relevant_information functions as a broad catch-all category without a clearly defined semantic boundary, making it difficult for the model to learn consistent patterns. This ambiguity likely also contributes to the confusion between not_humanitarian, caution_and_advice, and other_relevant_information, as tweets without a strong distinguishing signal may be absorbed into the broader "other" category.

The confusion between requests_or_urgent_needs and rescue_volunteering_or_donation_effort likely reflects genuine semantic overlap, since discussions of urgent needs are often accompanied by references to rescue activities, volunteering, or donation efforts.

### Flagging predictions for human review:

BERTweet outputs a probability distribution across all classes for each tweet and assigns the label with the highest probability. However, the highest-probability class is not always a reliable prediction.

Consider two examples:

- A tweet with probabilities of 70% for Class 1 and 30% for Class 2 is a relatively confident prediction.
- A tweet with probabilities of 50.01% for Class 1 and 49.99% for Class 2 is also assigned to Class 1, despite the model being highly uncertain and the two classes being nearly indistinguishable.

To account for this uncertainty, predictions can be flagged for human review. Instead of returning only the top predicted class, the model also returns:

- the second most likely class
- the probabilities associated with both classes

Two types of flags are introduced:

**1. Borderline Flag: if the first-ranking class is close to the second-ranking class in terms of probabilities. Examples of borderline tweets:**

| Tweet                                                                                                                                                                                                                  | Actual Label               | 1st class                | 1st prob. | 2nd class                              | 2nd prob. |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- | ------------------------ | --------: | -------------------------------------- | --------: |
| We are really sitting here on a water , sanitation and hygiene ticking bomb , ” - - @USER Secretary General @USER appeals for more immediate assistance for victims of #CycloneIdai in southern Africa .	 #CycloneIdai | requests_or_urgent_needs   | requests_or_urgent_needs |  0.372223 | rescue_volunteering_or_donation_effort |  0.371868 |
| Hillary knows what to do . Will anyone listen ? Were paying for a bloated military , lets at least get victims some help ! #PuertoRicoRelief                                                                           | other_relevant_information | not_humanitarian         |  0.390868 | other_relevant_information             |  0.390062 |


**2. Low-confidence Flag:** if the highest class probability is below 0.7. After testing multiple thresholds, 0.7 was selected as a balance between:

- minimizing the number of flagged tweets
- maintaining high accuracy among non-flagged predictions

Examples of low-confidence tweets:

| Tweet                                                                                                                                                                                                | Actual Label               | 1st class                  | 1st prob. |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- | -------------------------- | --------: |
| Greece fire : relatives of victims demand answers #Greece #news                                                                                                                                      | other_relevant_information | other_relevant_information |  0.270956 |
| Poor show @USER . I thought you are a phone call away from @USER & bring assistance as you promised instead of this showboating & comic behaviour.Victims of #CycloneIdai need help not this . Shame | not_humanitarian           | other_relevant_information |  0.279923 |

### Is flagging effective?
The flags are effective if the non-flagged predictions are mostly correct and the flagged predictions are mostly wrong. 

<img width="790" height="490" alt="download" src="https://github.com/user-attachments/assets/55629809-98b9-4701-89d6-a63921cdebf0" />

<img width="790" height="490" alt="download" src="https://github.com/user-attachments/assets/5f972e46-227a-49c9-848a-2deaa9787a3e" />






