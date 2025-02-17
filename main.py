import sys
import cv2
import os
from caption_model import caption_model
from caption_generate import analyze_attention_for_caption
from vocab_utils import load_word_mappings
from voc import load_vocab
from voc_embedding import load_embeddings

class ImageCaptionCLI:
    def __init__(self):
        self.init_model()
        
    def init_model(self):
        """初始化所需的資源和模型"""
        print("正在初始化資源...")
        self.vocab = load_vocab()
        self.wordtoidx, self.idxtoword, vocab_size = load_word_mappings()
        self.embedding_matrix = load_embeddings(wordtoidx=self.wordtoidx, vocab_size=vocab_size)
        
        print("正在創建模型...")
        self.model = caption_model(self.wordtoidx, self.embedding_matrix)
        self.model.load_weights(r'D:\newpy\1231_module\module\bestaug5.weights.h5')
        #self.model.load_weights(r'D:\newpy\1229structure\caption_aug5_lr\bestaug5.weights.h5')
        print("模型載入成功")
        
    def generate_caption(self, image_path):
        """生成圖片描述功能"""
        if not os.path.exists(image_path):
            return "錯誤：找不到指定的圖片文件"
            
        try:
            # 使用原始圖片生成描述
            initial_caption = analyze_attention_for_caption(image_path, self.model)
            
            # 根據描述內容添加前綴
            if initial_caption == "A student is writing a test paper.":
                final_caption = "No exam misconduct was detected -- " + initial_caption
            else:
                final_caption = "Detecting suspected exam misconduct -- " + initial_caption
                
            return final_caption
            
        except Exception as e:
            return f"發生錯誤: {str(e)}"

def main():
    try:
        # 初始化類別實例
        print("正在初始化系統...")
        caption_generator = ImageCaptionCLI()
        
        while True:
            # 讓使用者輸入圖片路徑
            image_path = input("\n請輸入圖片路徑 (輸入 'q' 退出): ").strip()
            
            # 檢查是否要退出
            if image_path.lower() == 'q':
                print("程式結束")
                break
                
            # 生成描述
            result = caption_generator.generate_caption(image_path)
            print("\n生成的描述：")
            print(result)
            
    except Exception as e:
        print(f"程式執行發生錯誤: {str(e)}")

if __name__ == '__main__':
    main()