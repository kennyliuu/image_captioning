from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from vocab_utils import load_word_mappings
from image_encoder import ImageEncoder
from config import OUTPUT_DIM, max_length
from caption_model import caption_model


def analyze_attention_for_caption(img_path, model):
    """分析字幕生成過程中的注意力權重，支持測試集圖片和外部圖片"""
    wordtoidx, idxtoword, _ = load_word_mappings()
    encoder = ImageEncoder()
    image_features = encoder.encodeImage(img_path).reshape((1, OUTPUT_DIM))
    
    
    # 獲取注意力層和創建注意力模型
    attention_layer = None
    for layer in model.layers:
        if 'attention' in layer.name:
            attention_layer = layer
            break
    
    attention_model = Model(
        inputs=model.inputs,
        outputs=[attention_layer.output, model.output]
    )
    
    # 生成序列並收集注意力權重
    sequence = [wordtoidx['startseq']]
    attention_weights = []
    generated_words = []
    
    for i in range(max_length):
        sequence_padded = pad_sequences([sequence], maxlen=max_length)
        attention_output, predictions = attention_model.predict([image_features, sequence_padded], verbose=0)
        
        predicted_id = np.argmax(predictions[0])
        word = idxtoword[predicted_id]
        
        attention_weights.append(attention_output[0])
        generated_words.append(word)
        
        if word == 'endseq':
            break
            
        sequence.append(predicted_id)
    
    # 移除特殊標記
    words = [w for w in generated_words if w not in ['startseq', 'endseq']]
    
    # 定義完整動作片段
    actions = {
        'peeking': ['peeking'],
        'sleeping': ['sleeping'],
        'standing': ['standing'],
        'turning around': ['turning', 'around'],
        'raising a hand': ['raising', 'a', 'hand'],
        'using a cell phone': ['using', 'a', 'cell', 'phone'],
        'writing a test paper': ['writing', 'a', 'test', 'paper']
    }
    
    # 檢查是否包含完整動作
    final_words = ['A', 'student', 'is']
    action_found = False
    
    # 先檢查特殊情況
    text = ' '.join(words)
    if 'around' in text:
        final_words.extend(['turning', 'around'])
        action_found = True
    elif 'phone' in text:
        final_words.extend(['using', 'a', 'cell', 'phone'])
        action_found = True
    else:
        # 檢查完整動作
        for action, action_seq in actions.items():
            action_text = ' '.join(action_seq)
            if action_text in text:
                final_words.extend(action_seq)
                action_found = True
                break
    
    # 如果沒找到完整動作，使用原始生成的文字
    if not action_found:
        final_words.extend(words)
    
    caption = ' '.join(final_words)
    if not caption.endswith('.'):
        caption += '.'
   

    return caption