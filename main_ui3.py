from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt
import sys
import cv2
import os
from caption_model import caption_model
from caption_generate import analyze_attention_for_caption
from vocab_utils import load_word_mappings
from voc import load_vocab
from voc_embedding import load_embeddings

class ImageCaptionUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initModel()
        
    def initModel(self):
        """初始化所需的資源和模型"""
        print("正在初始化資源...")
        self.vocab = load_vocab()
        self.wordtoidx, self.idxtoword, vocab_size = load_word_mappings()
        self.embedding_matrix = load_embeddings(wordtoidx=self.wordtoidx, vocab_size=vocab_size)
        
        print("正在創建模型...")
        self.model = caption_model(self.wordtoidx, self.embedding_matrix)
        self.model.load_weights(r'D:\newpy\1229structure\caption_aug5_lr\bestaug5.weights.h5')
        #self.model.load_weights(r'D:\newpy\1227_aug_whether\aug5\aug5.weights.h5')
        print("模型載入成功")
        
    def initUI(self):
        """初始化使用者介面"""
        self.setWindowTitle('圖片動作描述生成系統')
        self.setGeometry(100, 100, 800, 600)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # 原始圖片區域
        original_image_container = QVBoxLayout()
        original_title = QLabel('原始影像')
        original_title.setAlignment(Qt.AlignCenter)
        original_title.setFont(QFont('微軟正黑體', 12, QFont.Bold))
        original_image_container.addWidget(original_title)

        self.original_image_label = QLabel()
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMinimumSize(400, 400)
        self.original_image_label.setStyleSheet("border: 2px solid black")
        original_image_container.addWidget(self.original_image_label)
        
        layout.addLayout(original_image_container)
        
        # 按鈕區域
        button_layout = QHBoxLayout()
        
        self.load_button = QPushButton('載入圖片')
        font = QFont('微軟正黑體', 16)
        font.setBold(True)
        self.load_button.setFont(font)
        self.load_button.clicked.connect(self.loadImage)
        button_layout.addWidget(self.load_button)
        
        self.generate_button = QPushButton('生成動作描述')
        font = QFont('微軟正黑體', 16)
        font.setBold(True)
        self.generate_button.setFont(font)
        self.generate_button.clicked.connect(self.generateCaption)
        self.generate_button.setEnabled(False)
        button_layout.addWidget(self.generate_button)
        
        layout.addLayout(button_layout)
        
        self.caption_label = QLabel('描述將在這裡顯示')
        font = QFont('微軟正黑體', 16)
        font.setBold(True)
        self.caption_label.setFont(font)
        self.caption_label.setAlignment(Qt.AlignCenter)
        self.caption_label.setWordWrap(True)
        self.caption_label.setStyleSheet("font-size: 14pt; margin: 10px")
        layout.addWidget(self.caption_label)
        
        self.current_image_path = None
        
    def loadImage(self):
        """載入圖片功能"""
        file_name, _ = QFileDialog.getOpenFileName(self, '選擇圖片', '', 'Images (*.png *.jpg *.jpeg)')
        if file_name:
            self.current_image_path = file_name
            pixmap = QPixmap(file_name)
            scaled_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.original_image_label.setPixmap(scaled_pixmap)
            self.generate_button.setEnabled(True)
            self.caption_label.setText('描述將在這裡顯示')
            
    def generateCaption(self):
        """生成圖片描述功能"""
        if self.current_image_path:
            try:
                # 使用原始圖片生成描述
                initial_caption = analyze_attention_for_caption(self.current_image_path, self.model)
                
                # 根據描述內容添加前綴
                if initial_caption == "A student is writing a test paper.":
                    final_caption = "No exam misconduct was detected -- " + initial_caption
                else:
                    final_caption = "Detecting suspected exam misconduct -- " + initial_caption
                    
                # 顯示描述
                self.caption_label.setText(final_caption)
                
            except Exception as e:
                self.caption_label.setText(f"發生錯誤: {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = ImageCaptionUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()