import os
from PyQt6.QtWidgets import QComboBox, QInputDialog
from PyQt6.QtCore import QThreadPool
from utility import check_assemblyai_api_key, check_deepgram_api_key, ModelLoadThread


class ModelComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.api_keys = {}
        self.last_index = 0
        self.threadpool = QThreadPool()
        self.parent = parent
        self.model = None
        self.addItem("OpenAI Whisper: tiny")
        self.addItem("OpenAI Whisper: base")
        self.addItem("OpenAI Whisper: medium")
        self.addItem("OpenAI Whisper: large")
        self.addItem("OpenAI Whisper: turbo")
        self.addItem("Facebook Wav2Vec: vietnamese")
        self.addItem("Facebook Wav2Vec: english")
        self.addItem("DeepGram")
        self.addItem("AssemblyAI")
        self.activated.connect(self.textChanged)

    # Overwrite
    def textChanged(self, int):
        text = self.currentText()
        api_needed = ["DeepGram", "AssemblyAI"]
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
        dialog.setWindowTitle("API Key?")
        dialog.setLabelText("Please enter API Key")
        while True:
            dialog.exec()

            if dialog.result() == 1 and dialog.textValue():
                api_key = dialog.textValue()
            if self.checkAPIKey(text, api_key):
                self.api_keys[text] = api_key
                return True
            elif dialog.result() == 0:
                return False

            dialog.setLabelText("Please enter a valid API Key")

    def loadModel(self, text):
        worker = ModelLoadThread(text)
        self.threadpool.start(worker)
        worker.signals.result.connect(self._loadModel)

    def _loadModel(self, model):
        self.parent.setModel(model)
        print(f"Model loaded: {model}")

    def loadAPI(self, text, api_key):
        print(f"Loading API: {text} with API Key:")

    def checkAPIKey(self, text, api_key):
        print(f"Checking API Key: {text}")
        if text == "DeepGram":
            return check_deepgram_api_key(api_key)
        elif text == "AssemblyAI":
            return check_assemblyai_api_key(api_key)
