import sys
import assemblyai as aai
from deepgram import DeepgramClient, DeepgramApiKeyError
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QScrollArea, QWidget, QInputDialog
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
      self.add_model_combobox_part()
      self.add_language_combobox_part()
      self.add_buttons()
      self.add_transcript_output()
      
   def add_model_combobox_part(self):
      main_layout = self.centralWidget().layout()
      layout = QHBoxLayout()
      layout.setAlignment(Qt.AlignmentFlag.AlignRight)
      label = QLabel('Model')
      combo = ModelComboBox()
      layout.addWidget(label)
      layout.addWidget(combo)
      main_layout.addLayout(layout)

   def add_language_combobox_part(self):
      main_layout = self.centralWidget().layout()
      layout = QHBoxLayout()
      layout.setAlignment(Qt.AlignmentFlag.AlignRight)
      label = QLabel('Language')
      combo = LanguageComboBox()
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

   def add_transcript_output(self):
      main_layout = self.centralWidget().layout()
      scroll = QScrollArea()
      widget = QWidget()
      layout = QVBoxLayout()
      widget.setLayout(layout)
      
      for i in range(100):
         label = QLabel(f'Transcript {i}' + 'A' * 100)
         layout.addWidget(label)
      
      scroll.setWidget(widget)
      scroll.setWidgetResizable(True)
      scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
      scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
      main_layout.addWidget(scroll)
      

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
            if self.checkAPIKey(text):
               break

         dialog.setLabelText('Please enter a valid API Key')

   def loadModel(self, text):
      print(f'Loading model: {text}')

   def loadAPI(self, text, api_key):
      print(f'Loading API: {text} with API Key: {api_key[text]}')

   def checkAPIKey(self, text):
      print(f'Checking API Key: {text}')
      if text == 'DeepGram':
         try:
            # Check if the API key is valid
            DeepgramClient(api_key=self.api_keys[text])
         except DeepgramApiKeyError:
            return False
         except Exception as e:
            print('Unexpected error:', e)
            return False
         else:
            return True
      elif text == 'AssemblyAI':
         try:
            aai.settings.api_key = self.api_keys[text]
            aai.Transcriber()
         except aai.AssemblyAIError:
            return False
         except Exception as e:
            print('Unexpected error:', e)
            return False
         else:
            return True

class LanguageComboBox(QComboBox):
   def __init__(self, parent = None):
      super().__init__(parent)
      self.addItem('English')
      self.addItem('Vietnamese')
      self.addItem('Auto')
      self.currentTextChanged.connect(self.textChanged)

   def textChanged(self, text):
      print(f'Language changed to: {text}')


if __name__ == "__main__":
   app = QApplication(sys.argv)
   window = MyMainWindow()
   window.show()
   sys.exit(app.exec())