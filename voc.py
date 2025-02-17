import pickle

def load_vocab(vocab_path=r"D:\newpy\1231_module\module\vocab.pkl"):
   try:
       with open(vocab_path, 'rb') as f:
           vocab = pickle.load(f)
       print(f"Loaded {len(vocab)} words")
       return vocab
   except Exception as e:
       print(f"Error loading vocabulary: {str(e)}")
       return None