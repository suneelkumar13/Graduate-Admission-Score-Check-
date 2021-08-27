from flask import Flask, render_template, request
import pandas as pd 

from flask_chance_of_admit import predict_newsample

app = Flask(__name__)

@app.route('/')

def home():
    return render_template('search.html')

@app.route('/search/resultscore', methods=['GET', 'POST'])

def search_request():
    new_sample = pd.DataFrame()
    new_sample['GRE Score'] = [request.form.get("GS")]
    new_sample['TOEFL Score'] = [request.form.get("TS")]
    new_sample['University Rating'] = [request.form.get("UR")]
    new_sample['SOP'] = [request.form.get("SOP")]
    new_sample['LOR '] = [request.form.get("LOR")]
    new_sample['CGPA'] = [request.form.get("CGPA")]
    new_sample['Research'] = [request.form.get("RE")]
    
    result = predict_newsample(new_sample)
    result = '%.2f' %((result[0])*100)
    
    return render_template('resultscore.html', res=result )

if __name__ == "__main__":
    app.run(debug=True)