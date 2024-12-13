import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLineEdit, QWidget, QInputDialog
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
      combo = ModelComboBox()
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
      combo.addItem('Vietnamese')
      combo.addItem('Auto')
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

class ModelComboBox(QComboBox):
   def __init__(self, parent = None):
      super().__init__(parent)
      self.api_keys = {}
      self.addItem('OpenAI Whisper: Tiny')
      self.addItem('OpenAI Whisper: Base')
      self.addItem('OpenAI Whisper: Medium')
      self.addItem('OpenAI Whisper: Large')
      self.addItem('OpenAI Whisper: Turbo')
      self.addItem('DeepGram')
      self.addItem('AssemblyAI')
      self.currentTextChanged.connect(self.textChanged)
   
   # Overwrite
   def textChanged(self, text):
      api_needed = ['DeepGram', 'AssemblyAI']
      if text in api_needed:
         self.showAPIKeyInput(text)
         self.loadAPI(text, self.api_keys)
      else:
         self.loadModel(text)
   
   def showAPIKeyInput(self, text):
      if self.api_keys.get(text):
         return
      
      dialog = QInputDialog(parent=self)
      dialog.setInputMode(QInputDialog.InputMode.TextInput)
      dialog.setWindowTitle('API Key?')
      dialog.setLabelText('Please enter API Key')
      while True:
         dialog.exec()

         if dialog.result() == 1 and dialog.textValue():
            self.api_keys[text] = dialog.textValue()
            print(f'API Key: {dialog.textValue()}')
            break

         dialog.setLabelText('Please enter a valid API Key')

   def loadModel(self, text):
      print(f'Loading model: {text}')

   def loadAPI(self, text, api_key):
      print(f'Loading API: {text} with API Key: {api_key[text]}')


if __name__ == "__main__":
   app = QApplication(sys.argv)
   window = MyMainWindow()
   window.show()
   sys.exit(app.exec())