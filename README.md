# Project-NLP-Business-Case-Automated-Customer-Reviews

#Steps
01 Read the dataset and clean data
02  Categorize data into categories 
(positive, negative, neutral)
03  use pretrined models Bert model triean model And Model Evaluation 
04 Deployment

#Task 1
#Classification
#1️⃣ Sentiment Classification (BERT-based)
In this part, we built a sentiment classification model using a pre-trained BERT model fine-tuned on our dataset.
The dataset was balanced into three sentiment categories:

Positive

Neutral

Negative

📊 Evaluation Results:
Class	Precision	Recall	F1-Score	Support
NEGATIVE	0.971	0.975	0.973	1980
NEUTRAL	0.946	0.888	0.916	1993
POSITIVE	0.902	0.954	0.928	2027
✅ Overall Accuracy: 0.91

✅ The model showed high accuracy, particularly in detecting Positive and Negative sentiments.
 #Task2
#2️⃣Clustering (KMeans, PCA, t-SNE)
We applied clustering techniques on the text embeddings (extracted using BERT) to discover natural groupings within the data.

📊 KMeans Clustering (with PCA)
Original Shape: (4987, 3074)

After PCA (50 components): (4987, 50)

Clustering Evaluation Metrics:

Silhouette Score: 0.204

Davies-Bouldin Index: 1.597

Calinski-Harabasz Index: 895.547


📉 Visualization:
We used PCA and t-SNE to reduce the embeddings to 2D and visualize clusters. Each cluster was labeled and color-coded, revealing natural data structure and overlaps between categories.


#Task3
#3️⃣ Summarization (AI-based)
We generated automatic summaries for customer reviews using a transformer-based summarization model. The summaries captured the key points in customer feedback in a short, coherent format — useful for management and quick insights.


#Task4
🚀 Flask Deployment & Running the App
📂 Folder Structure:
swift
نسخ
تحرير
/project/
 ├── model/
 │    └── bert_sentiment_model/
 ├── app.py
 ├── templates/
 │    └── index.html
 ├── static/
 │    └── style.css
 └── requirements.txt
📝 Example app.py
python
نسخ
تحرير
from flask import Flask, request, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

model = BertForSequenceClassification.from_pretrained('./model/bert_sentiment_model')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['user_input']
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits).item()
    labels = ['Negative', 'Neutral', 'Positive']
    result = labels[pred]
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
✅ To Run the Flask App:
Install dependencies:

bash
نسخ
تحرير
pip install -r requirements.txt
Run the app:

bash
نسخ
تحرير
python app.py
Open your browser and go to: http://127.0.0.1:5000/

#✅ Conclusion:
The project successfully combined:

Deep learning-based sentiment classification

Clustering and visualization of customer feedback

Automatic summarization

Deployment using Flask

We achieved high performance in classification and insightful clustering visualizations, delivering a complete AI-driven text analysis pipeline.
