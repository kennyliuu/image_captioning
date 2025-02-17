import pickle

def load_word_mappings(mappings_path=r"D:\newpy\1231_module\module\word_mappings.pkl"):
   try:
       with open(mappings_path, 'rb') as f:
           mappings = pickle.load(f)
       return mappings['wordtoidx'], mappings['idxtoword'], mappings['vocab_size']
   except Exception as e:
       print(f"Error loading mappings: {str(e)}")
       return None, None, None