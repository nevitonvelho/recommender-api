from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)
CORS(app)

products = pd.read_csv('products.csv')
products['description'] = products['description'].fillna('').str.lower()

nltk.download('stopwords')

stop_words_pt = stopwords.words('portuguese')

tfidf = TfidfVectorizer(stop_words=stop_words_pt)

tfidf_matrix = tfidf.fit_transform(products['description'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
of_index = pd.Series(products.index, index=products['product_id']).drop_duplicates()

def recommend(product_id, top_n=5):
    if product_id not in of_index:
        raise ValueError(f"Product ID '{product_id}' não encontrado no dataset.")
    
    idx = of_index[product_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1: top_n + 1]
    indices = [i[0] for i in sim_scores]
    
    results = products.iloc[indices][['product_id', 'product_name']]
    return results.to_dict(orient='records')

@app.route('/recommend', methods=['GET'])
def get_recommendations():
    product_id = request.args.get('product_id')
    top_n = request.args.get('top_n', default=5, type=int)

    if not product_id:
        return jsonify({'error': 'Parâmetro product_id é obrigatório'}), 400
    
    try:
        recs = recommend(product_id, top_n)
        return jsonify(recs)
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    
@app.route('/products', methods=['GET'])
def get_all_products():
    return jsonify(products.to_dict(orient='records'))


if __name__ == '__main__':
    app.run(debug=True)
