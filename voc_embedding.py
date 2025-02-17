import numpy as np

def load_embeddings(matrix_path=r"D:\newpy\1231_module\module\embedding_matrix.npy", wordtoidx=None, vocab_size=None):
   try:
       loaded_matrix = np.load(matrix_path)
       embeddings_index = {word: loaded_matrix[i] for word, i in wordtoidx.items()}
       
       embedding_dim = 300
       embedding_matrix = np.zeros((vocab_size, embedding_dim))
       for word, i in wordtoidx.items():
           if (vector := embeddings_index.get(word)) is not None:
               embedding_matrix[i] = vector
               
       return embedding_matrix
   except Exception as e:
       print(f"Error loading embeddings: {str(e)}")
       return None