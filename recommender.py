import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse

# 1. Carrega dados
# Espera um CSV 'products.csv' com colunas: 'product_id', 'product_name', 'description'
products = pd.read_csv('products.csv')

# 2. Preprocessamento de texto
products['description'] = products['description'].fillna('').str.lower()

# 3. Vetorização com TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(products['description'])

# 4. Matriz de similaridade de cosseno
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 5. Mapas de índices
of_index = pd.Series(products.index, index=products['product_id']).drop_duplicates()

# 6. Função de recomendação
def recommend(product_id, top_n=5):
    """
    Retorna até top_n produtos similares com base na descrição.
    """
    if product_id not in of_index:
        raise ValueError(f"Product ID '{product_id}' não encontrado no dataset.")

    idx = of_index[product_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1: top_n + 1]
    indices = [i[0] for i in sim_scores]
    return products.iloc[indices][['product_id', 'product_name']]

# 7. Execução via CLI
def main():
    parser = argparse.ArgumentParser(description="Sistema de recomendação de produtos")
    parser.add_argument('--product_id', type=str, required=True, help='ID do produto (ex: P001)')
    parser.add_argument('--top_n', type=int, default=5, help='Número de recomendações')
    args = parser.parse_args()

    try:
        recs = recommend(args.product_id, top_n=args.top_n)
        print(f"Recomendações para '{args.product_id}':")
        print(recs.to_string(index=False))
    except ValueError as e:
        print(e)

if __name__ == '__main__':
    main()

