import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLineEdit, QWidget
from PyQt6.QtCore import Qt

class MyMainWindow(QMainWindow):
   def __init__(self, parent = None):
      super().__init__(parent)
      self.setWindowTitle("App")

      # The main scene
      main_widget = QWidget()
      main_layout = QVBoxLayout()
      main_widget.setLayout(main_layout)
      self.setCentralWidget(main_widget)

      # Add widgets
      self.add_model_combobox()
      self.add_language_combobox()
      self.add_buttons()
      
   def add_model_combobox(self):
      main_layout = self.centralWidget().layout()
      layout = QHBoxLayout()
      layout.setAlignment(Qt.AlignmentFlag.AlignRight)
      label = QLabel('Model')
      combo = QComboBox()
      combo.addItem('Model 1')
      combo.addItem('Model 2')
      layout.addWidget(label)
      layout.addWidget(combo)
      main_layout.addLayout(layout)

   def add_language_combobox(self):
      main_layout = self.centralWidget().layout()
      layout = QHBoxLayout()
      layout.setAlignment(Qt.AlignmentFlag.AlignRight)
      label = QLabel('Language')
      combo = QComboBox()
      combo.addItem('English')
      combo.addItem('Spanish')
      layout.addWidget(label)
      layout.addWidget(combo)
      main_layout.addLayout(layout)

   def add_buttons(self):
      main_layout = self.centralWidget().layout()
      layout = QHBoxLayout()
      layout.setAlignment(Qt.AlignmentFlag.AlignRight)
      button = QPushButton('Start')
      layout.addWidget(button)
      main_layout.addLayout(layout)

if __name__ == "__main__":
   app = QApplication(sys.argv)
   window = MyMainWindow()
   window.show()
   sys.exit(app.exec())