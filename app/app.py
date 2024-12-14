import sys
import time
import assemblyai as aai
import faster_whisper
from deepgram import DeepgramClient, DeepgramApiKeyError
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QScrollArea, QWidget, QInputDialog
from PyQt6.QtCore import Qt, QRunnable, QThreadPool, pyqtSignal, QObject
from utility import check_assemblyai_api_key, check_deepgram_api_key, load_wav2vec, load_whisper

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
      self.last_index = 0
      self.threadpool = QThreadPool()
      self.model = None
      self.addItem('OpenAI Whisper: tiny')
      self.addItem('OpenAI Whisper: base')
      self.addItem('OpenAI Whisper: medium')
      self.addItem('OpenAI Whisper: large')
      self.addItem('OpenAI Whisper: turbo')
      self.addItem('Facebook Wav2Vec: base')
      self.addItem('Facebook Wav2Vec: large')
      self.addItem('DeepGram')
      self.addItem('AssemblyAI')
      self.activated.connect(self.textChanged)
   
   # Overwrite
   def textChanged(self, int):
      text = self.currentText()
      api_needed = ['DeepGram', 'AssemblyAI']
      if text in api_needed:
         if self.showAPIKeyInput(text):
            self.loadAPI(text, self.api_keys)
         else:
            self.setCurrentIndex(self.last_index)
      else:
         self.loadModel(text)

      self.last_index = int
   
   def showAPIKeyInput(self, text):
      if self.api_keys.get(text):
         return True
      
      dialog = QInputDialog(parent=self)
      dialog.setInputMode(QInputDialog.InputMode.TextInput)
      dialog.setWindowTitle('API Key?')
      dialog.setLabelText('Please enter API Key')
      while True:
         dialog.exec()

         if dialog.result() == 1 and dialog.textValue():
            api_key = dialog.textValue()
            if self.checkAPIKey(text, api_key):
               self.api_keys[text] = api_key
               return True
         elif dialog.result() == 0:
            return False

         dialog.setLabelText('Please enter a valid API Key')

   def loadModel(self, text):
      worker = ModelLoadThread(text)
      self.threadpool.start(worker)
      worker.signals.result.connect(self._loadModel)

   def _loadModel(self, model):
      self.model = model
      print(f'Model loaded: {model}')
      
   def loadAPI(self, text, api_key):
      print(f'Loading API: {text} with API Key:')

   def checkAPIKey(self, text, api_key):
      print(f'Checking API Key: {text}')
      if text == 'DeepGram':
         return check_deepgram_api_key(api_key)
      elif text == 'AssemblyAI':
         return check_assemblyai_api_key(api_key)

class LanguageComboBox(QComboBox):
   def __init__(self, parent = None):
      super().__init__(parent)
      self.addItem('English')
      self.addItem('Vietnamese')
      self.addItem('Auto')
      self.currentTextChanged.connect(self.textChanged)

   def textChanged(self, text):
      print(f'Language changed to: {text}')

class ModelLoadThread(QRunnable):
   def __init__(self, text):
      super().__init__()
      self.text = text
      self.signals = ModelLoadSignal()

   def run(self):
      time.sleep(2)
      if self.text.startswith('OpenAI Whisper'):
         model = load_whisper(self.text.split(':')[-1].strip())
         self.signals.result.emit(model)
         self.signals.finished.emit()
      elif self.text.startswith('Facebook Wav2Vec'):
         model = load_wav2vec(self.text.split(':')[-1].strip())
         self.signals.result.emit(model)
         self.signals.finished.emit()

class ModelLoadSignal(QObject):
   finished = pyqtSignal()
   result = pyqtSignal(object)

if __name__ == "__main__":
   app = QApplication(sys.argv)
   window = MyMainWindow()
   window.show()
   sys.exit(app.exec())