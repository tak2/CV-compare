from flask import Flask, render_template, url_for, request, redirect
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():
    print ("ggg")
    if request.method == 'POST':
        try:

            textjd = request.form['jdtxt']
            textcv = request.form['cvtxt']
            #print (textjd)
            #print (textcv)
            documents = [textjd, textcv]
            count_vectorizer = CountVectorizer()
            sparse_matrix = count_vectorizer.fit_transform(documents)
            doc_term_matrix = sparse_matrix.todense()
            df = pd.DataFrame(doc_term_matrix, 
                        columns=count_vectorizer.get_feature_names(), 
                        index=['textjd', 'textcv'])
            answer = cosine_similarity(df, df)
            answer = pd.DataFrame(answer)
            answer = answer.iloc[[1],[0]].values[0]
            answer = round(float(answer),4)*100
            return ("Your resume matched " + str(answer) + " %" + " of the job-description!")
        except:
            return render_template('index.html')
    else:
            return render_template('index.html')
        


if __name__ == "__main__":
    app.run(debug=True)