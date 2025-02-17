import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class ImageEncoder:
    def __init__(self):
        # 初始化模型
        self.encode_model = ResNet50(weights='imagenet', 
                                   include_top=False, 
                                   pooling='avg')
        self.preprocess_input = tf.keras.applications.resnet50.preprocess_input

    def encode_image(self, img_path):
        try:
            # 載入圖像
            img = load_img(img_path, target_size=(224, 224))
            
            # 將圖像轉換為數組
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = self.preprocess_input(x)
            x = self.encode_model.predict(x, verbose=0)
            x = np.reshape(x, (2048,))
            
            return x
            
        except Exception as e:
            print(f"Error encoding image: {str(e)}")
            return None
            
    # 添加 encodeImage 作為 encode_image 的别名
    def encodeImage(self, img_path):
        return self.encode_image(img_path)