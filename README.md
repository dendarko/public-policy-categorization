
# Public Policy Categorization and Recommendation System

## Project Overview

This project focuses on optimizing the analysis of public policies by creating an **NLP-based categorization model** that categorizes public policies based on user interests and demographic attributes. The aim is to enhance public policy recommendations, improving accuracy by **25%**, and automate the process using a **Flask API**, reducing policy review time by **40%**.

## Key Features

- **Public Policy Categorization**: Policies are categorized into predefined categories such as Housing, Environment, Education, and more, based on keywords extracted from policy documents.
- **Handling Missing Data**: Advanced techniques such as filling missing values, tokenization, and stemming ensure robust text preprocessing.
- **Feature Engineering**: Combines textual data with demographic information to create a comprehensive feature set for model training.
- **Class Imbalance Handling**: Utilized **SMOTE** (Synthetic Minority Over-sampling Technique) to address class imbalances during model training.
- **Model Training**: Developed using a **Gradient Boosting Classifier**, evaluated with metrics such as accuracy, precision, recall, and F1-score.
- **API Development**: A Flask-based API provides policy recommendations based on user inputs (interest, city, state).

## Project Structure

- **Notebooks**: Jupyter notebooks demonstrating the entire workflow, from data preprocessing to model training.
- **Scripts**: Python scripts for model training, data processing, and API setup.
- **Models**: The trained machine learning model (`best_policy_model.pkl`) and vectorizers (`tfidf_vectorizer.pkl`, `onehot_encoder.pkl`).
- **Visualization**: Confusion matrices and performance metrics for model evaluation.

## Datasets

The datasets used include:
- **Orders Dataset**: Contains policy details such as classification numbers, passage dates, and effective dates.
- **Personnel Dataset**: Includes demographic information (e.g., age, location) and policy interests of users.

## Technical Implementation

1. **Preprocessing**:
   - Tokenization and stemming of policy document texts.
   - Handling missing values with median imputation and placeholder values.
2. **Categorization**:
   - Keyword-based categorization using expanded keyword lists for each category.
   - Assign dominant and minor categories based on frequency of matched keywords.
3. **Model Training**:
   - A **Gradient Boosting Classifier** is used to train on combined text and demographic data.
   - Class imbalance is handled with **SMOTE** to oversample minority categories.
4. **API Development**:
   - Flask API that accepts user input and returns a predicted policy category along with relevant documents.

## Model Performance

The model demonstrates high accuracy across all policy categories, with precision, recall, and F1-scores near **1.0**. The confusion matrix highlights minimal misclassifications, showcasing the robustness of the model.

![Confusion Matrix](./confusion_matrix.png)

## API Endpoints

1. `/predict` - Accepts a POST request with user inputs (interest, city, state) and returns the best matching policy category along with relevant policy documents.
2. `/download/<path:filename>` - Provides access to download the recommended policy documents.

## How to Run

### Prerequisites

- Python 3.8 or higher
- Required Libraries: `Flask`, `joblib`, `scikit-learn`, `nltk`, `pandas`, `seaborn`, `matplotlib`
  
### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/public-policy-categorization.git
   cd public-policy-categorization
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask API:
   ```bash
   python civa_project_g1(3).py
   ```

4. Test the API:
   Use tools like **Postman** or **curl** to send POST requests to the `/predict` endpoint.

## Future Work

- Improve location-based categorization accuracy.
- Extend the model to support more diverse policy categories.
- Integrate feedback mechanisms to continuously refine the model.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
