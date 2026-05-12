##  Introduction
### Problem Inspiration:
During humanitarian crises, social media platforms become critical communication channels; however, the sheer volume of posts makes it nearly impossible to manually identify which messages signal genuine emergencies. This project explores automated classification of disaster-related tweets across 10 humanitarian categories, ranging from rescue requests to infrastructure damage. The goal is to build a system that not only classifies tweets accurately but knows when it is uncertain: flagging ambiguous predictions for human review rather than forcing a confident answer.

### Datasets: 
This project uses the Humaid dataset from ..., containing 42K labelled English tweets across 10 humanitarian categories: 
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
The classes are well-stratified between the train, test, and validation set. However, there are some minority classes: "requests_or_urgent_needs", "not_humanitarian", "displaced_people_and_evacuations", "caution_and_advice", and "missing_or_found_people" being the most extreme - accounting for only ~0.5% of the dataset.
<img width="989" height="490" alt="HumAID tweet classification with BERTweet-1777725892738" src="https://github.com/user-attachments/assets/4523f47c-328e-4f00-9e73-1b2f8d4a802b" />

### BERTweet (in 1 sentence): https://huggingface.co/vinai/bertweet-base
	- Based on RoBERTA pre-training procedure
	- Trained on 850M English Tweets from 2012-2019
	- Suitable for tweet classification tasks that requires deeper sentence comprehension 
	

## Data Preprocessing:
- For SVM and LR:
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
Since SVM has better overall performance, deeper analyses are only done on SVM and not on Logistic Regression. 

**Confusion Matrix**
<img width="1059" height="963" alt="HumAID tweet classification with BERTweet-1777727543912" src="https://github.com/user-attachments/assets/d66e13fb-0cf4-4f47-8da4-6e11c390011f" />

- "**not_humanitarian**" is mainly misclassified as **"other_relevant_information"**
- "**other_relevant_information**" is mainly misclassified as "**rescue_volunteering_or_donation_effort**", "**infrastructure_and_utility_damage**", and "**caution_and_advice**". 

**Manually inspecting the errors**:
Upon manual inspection of the most commonly misclassified classes, I saw one important pattern: these classes are ambiguous by definition and contain information about other classes. 
- "other_relevant_information" can contains anything related to humanitarian.
- "caution_and_advice" and "requests_or_urgent_needs" normally include information about the disasters such as casualties, infrastructure damage, etc.

Such information confuses the classifier. For example, a misclassified "other_relevant_information" tweet is: "rural area small villages roads challenges rescue workers face amatrice" (filler words removed). It was misclassified as rescue efforts which make sense since it contain keywords about rescue workers. 

The TF-IDF vectorizer does not "understand" the entire sentence, it strips individual words from their context and count how many times they appeared. Furthermore, the preprocessing may have removed important filler words, the above sentence can be read very differently if the filler words were not removed. Therefore, using BERTweet may improve performance due to its context comprehension and minimal pre-processing required. 

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
### Result: 
The Macro F1 score peaked at epoch 4: **0.7636**. SVM's score was 0.69, so **BERTweet improved by 10% compared to SVM.** 

*Insert graph comparing the F1 Score*

Furthermore, **All individual f1-score of BERTweet is higher than SVM's** as can be seen from the figure below: 

<img width="785" height="490" alt="download" src="https://github.com/user-attachments/assets/f2ee0c9e-115e-4852-8de4-03a09fc4dba4" />

1. **"Not_humanitarian"** is still the worst-performing class; however, its f1-score almost doubled compared to the SVM case. As this class is a catch-all class, it shows BERT's capability of understanding full-sentence context compared to SVM. 
2. **"Missing_or_found_people"** improved by around 10%
3. **"Requests_or_urgent_needs", and "Caution_and_advice"** improved by around 15%
4. **"Other_relevant_information"** improved by only 0.02 point, signifying a data problem rather than a modelling problem: this class is too generic and cannot be meaningfully separated from other classes. 

### **Error Analysis**:  Four classes with the most errors:

<img width="1059" height="963" alt="download" src="https://github.com/user-attachments/assets/d3ef1cb4-0487-4500-b058-1440cb73dd6e" />

1. **Not_humanitarian** is mostly misclassified as **other_relevant_information**.
2. **Other_relevant_information** is misclassified as a range of different classes.
3. **Requests_or_urgent_needs** is mostly misclassified as **rescue_volunteering_or_donation_effort**.
4. **Caution_and_advice** is mostly misclassified as **other_relevant_information**.

Interpretation of these patterns:

- The misclassification of **other_relevant_information** is the easiest to explain. As a broad catch-all category, it lacks a clearly defined semantic boundary and may contain highly heterogeneous content, the model struggles to learn consistent patterns.
- The confusion between **not_humanitarian** and **other_relevant_information**, as well as between **caution_and_advice** and **other_relevant_information**, may also stem from the ambiguous nature of the catch-all category. Tweets that do not strongly express a distinctive humanitarian signal may be absorbed into other_relevant_information, even when they belong to more specific categories.
- The confusion between **requests_or_urgent_needs** and **rescue_volunteering_or_donation_effort** is likely due to semantic overlap between the two classes. Discussions of urgent needs are often accompanied by references to rescue activities, volunteering, or donation efforts, making the boundary between these categories less distinct.

### Flagging low-confidence predictions for human review:

No model perfectly predict the unknown, especially for tweets that are not even clear to human readers; therefore, flagging low-confidence predictions for human reviewers is an important step in correcting the model's behaviour. Since for each observation, BERT-type models output real logits - probabilities of the tweet being in each classes. The model then simply choose the class with the highest probability as the label for that tweet. This can lead to major problems, consider the two following two cases:

- A tweet has 70% of being in Class 1 and 30% of being in Class 2. BERTweet will label it as Class 1, with relatively large confidence. This make sense.
- A tweet has 50.01% of being in Class 1 and 49.99% of being in Class 2. BERTweet will still label it as Class 1 since 50.01 > 49.99. However, this does not make much sense since the model is only 50% sure about the 1st class - which is as good as a 50-50 random guess . Furthermore, the odds of the first-ranking class is not meaningfully higher than the second-ranking one. So labeling it as the 1st-class is misleading.

To account for this problem, the model should flag predictions similar to the second case above where it is not so "confident" about which is the most likely class. To implement this, each predictions also return the 2nd most likely classes and not only the most likely one. It also return the probabilities of being in these 2 classes. Two kind of labels are then created for each predictions:

1. Borderline Label: if the first-ranking class is close to the second-ranking class in terms of probabilities.

| Tweet | Actual Label | 1st class | 1st prob. | 2nd class | 2nd prob. |
|---|---|---|---:|---|---:|
|We are really sitting here on a water , sanitation and hygiene ticking bomb , ” - - @USER Secretary General @USER appeals for more immediate assistance for victims of #CycloneIdai in southern Africa | requests_or_urgent_needs | requests_or_urgent_needs | 0.372223 | rescue_volunteering_or_donation_effort | 0.371868 |
|Hillary knows what to do . Will anyone listen ? Were paying for a bloated military , lets at least get victims some help ! #PuertoRicoRelief | other_relevant_information | not_humanitarian | 0.390868 | other_relevant_information | 0.390062 |

3. Low-confidence Label: if the first-ranking class's probabilities is smaller than 0.7 (note)

**Borderline Label:**
6135	We are really sitting here on a water , sanitation and hygiene ticking bomb , ” - - @USER Secretary General @USER appeals for more immediate assistance for victims of #CycloneIdai in southern Africa .	requests_or_urgent_needs	requests_or_urgent_needs	0.372223	rescue_volunteering_or_donation_effort	0.371868
7989	Hillary knows what to do . Will anyone listen ? Were paying for a bloated military , lets at least get victims some help ! #PuertoRicoRelief	other_relevant_information	not_humanitarian	0.390868	other_relevant_information	0.39006
**Low-confidence Label:**
