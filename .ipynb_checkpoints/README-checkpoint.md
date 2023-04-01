# Language Detection Using Natural Language Processing

This is a project for detecting the language of a given text using natural language processing techniques and Python. The project supports three languages: English, French, and Darija.

## Files Structure

The project files are organized as follows:


```
.
├── data
│   ├── english
│   │   ├── train
│   │   └── test
│   ├── french
│   │   ├── train
│   │   └── test
│   └── darija
│       ├── train
│       └── test
├── models
├── notebooks
├── README.md
└── requirements.txt

```


- **data**: This directory contains the text data used for training and testing the language detection model. Each language has its own subdirectory with separate train and test sets.
- **models**: This directory contains the trained language detection models.
- **notebooks**: This directory contains the Jupyter notebooks used for data preparation, model training, and evaluation.
- **README.md**: This file is the main documentation for the project.
- **requirements.txt**: This file lists the Python dependencies required for running the project.
- **scraping**: This directory contains the raw text data collected from web scraping.

## Steps

### 1. Collect Data

The first step is to collect text data for each language. In this project, we collected text data using web scraping techniques. We scraped texts from various websites and social media platforms for each language.

The Jupyter notebook for this step is located at `notebooks/01-Data-Collection.ipynb`. This notebook contains the code for web scraping and saving the raw text data to disk.

### 2. Prepare Data

The next step is to prepare the raw text data for training and testing the language detection model. We cleaned and preprocessed the text data by removing non-alphabetic characters, normalizing the text, and tokenizing the text.

The Jupyter notebook for this step is located at `notebooks/02-Data-Preparation.ipynb`. This notebook loads the raw text data, preprocesses it, and saves the cleaned data to disk.

### 3. Feature Extraction

After data preparation, we extract features from the cleaned text data. We used the Bag-of-Words and TF-IDF techniques to convert the text into numerical features that can be used to train the machine learning model.

The Jupyter notebook for this step is located at `notebooks/03-Feature-Extraction.ipynb`. This notebook loads the cleaned text data, applies feature extraction techniques, and saves the feature vectors to disk.

### 4. Split the Data

After feature extraction, we split the data into training and testing sets. The training set is used to train the machine learning model, while the testing set is used to evaluate its performance.

The Jupyter notebook for this step is located at `notebooks/04-Data-Splitting.ipynb`. This notebook loads the feature vectors, splits the data into training and testing sets, and saves the split data to disk.

### 5. Train a Model

Once the data is split, we can train a machine learning model using various algorithms such as Naive Bayes, SVM, or Logistic Regression.

The Jupyter notebook for this step is located at `notebooks/05-Model-Training.ipynb`. This notebook loads the split data, trains a machine learning model using the scikit-learn library, and saves the trained model to disk.

### 6. Evaluate the Model

After training the model, we evaluate its performance on the testing set. We calculate metrics such as accuracy, precision, recall, and F1-score to assess the model's performance.

The Jupyter notebook for this step is located at `notebooks/06-Model-Evaluation.ipynb`. This notebook loads the trained model and the testing set, applies the model to the testing set, and calculates the evaluation metrics.

### 7. Deploy the Model
After the model is evaluated, we can deploy it for language detection on new texts. We load the trained model and apply it to new texts to predict the language of the text.

The Jupyter notebook for this step is located at `notebooks/07-Model-Deployment.ipynb`. This notebook loads the trained model, applies it to new texts, and saves the language predictions to disk.

### Conclusion
This project demonstrates the use of natural language processing and machine learning techniques for language detection. By following the steps outlined above, you can train and deploy a language detection model for the three languages supported in this project.