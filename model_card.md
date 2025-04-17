# Model Card
- 
## Model Details
- Model Type: RandomForrestClassifier
- Framework: scikit-learn
- Preprocessing: One-hot encoding of categorical variables, scaling of continuous features using StandardScaler
- Training Script: train_and_save.py
- Serving Interface: FastAPI RESTful API via main.py  

## Intended Use
- Purpose: Predict whether an individual's income exceeds $50K based on demographic and employment features
- User: Developers or Data Scientists
- Usage: Edcuational, research, or prototype environments. 

## Training Data
- Source: U.S. Census Adult Income Dataset
- Formate: CSV File with 15 columns including categorical and numeric features. 

## Evaluation Data
- Same dataset as training, split 80/20
- Preprocessed using same pipeline

## Metrics
- Precision: 0.5093
- Recall   : 0.8540
- F1 Score : 0.6381

## Ethical Considerations
- Bias Risk: This dataset reflects historical societal biases (gender, race, education)
- Fairness: Consider evaluating model performance across slices to ensure fairness using gender or race. 
- Privacy: The model is trained on anonymized public data but should not be used on real personal data without reviews for compliance. 
## Caveats and Recommendations'
- Do not use in high-stakes decision without auditing. 
- Retrain model if using live or updated data. 
- Use slice performance evaluations to monitor fairness and bias over time. 
- Log inputs and outputs to catch unexpected behavior. 