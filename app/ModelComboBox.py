import os
from PyQt6.QtWidgets import QComboBox, QInputDialog, QErrorMessage, QLineEdit
from PyQt6.QtCore import QThreadPool
from utility import (
    check_assemblyai_api_key,
    check_deepgram_api_key,
    ModelLoadThread,
    APIError,
)
from deepgram import DeepgramClient
import assemblyai as aai


class ModelComboBox(QComboBox):
    def __init__(self, setting, parent=None):
        super().__init__(parent)
        self.setting = setting
        self.BASE_DIR = self.setting.value("BASE_DIR")
        self.last_index = 0
        self.parent = parent
        if self.parent:
            self.threadpool = parent.threadpool
        else:
            self.threadpool = QThreadPool()
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
        if text in api_needed and not self.parent.clients[text]:
            if self.showAPIKeyInput(text):
                self.loadAPI(text, self.setting.value(text + "/api_key"))
            else:
                self.setCurrentIndex(self.last_index)
                return
        elif text not in api_needed:
            self.loadModel(text)

        self.last_index = int

    def showAPIKeyInput(self, text):
        self.setting.beginGroup(text)
        api_key = self.setting.value("api_key")
        self.setting.endGroup()
        if api_key:
            return True

        dialog = QInputDialog(parent=self)
        dialog.setInputMode(QInputDialog.InputMode.TextInput)
        dialog.setTextEchoMode(QLineEdit.EchoMode.Password)
        dialog.setWindowTitle("API Key?")
        dialog.setLabelText("Please enter API Key")

        error_dialog = QErrorMessage(parent=self)
        error_dialog.setWindowTitle("Error")
        error_dialog.setModal(True)
        while True:
            dialog.exec()

            if dialog.result() == 1:
                api_key = dialog.textValue()
                try:
                    self.checkAPIKey(text, api_key)
                except APIError as e:
                    if e.code == 401:
                        dialog.setLabelText("Please enter a valid API Key")
                    else:
                        error_dialog.showMessage(f"Error: {e}")
                        return False
                    continue
                self.setting.beginGroup(text)
                self.setting.setValue("api_key", api_key)
                self.setting.endGroup()
                return True
            else:
                return False

    def loadModel(self, text):
        button = self.parent.transcribe_button
        button.setEnabled(False)
        button.setText("Loading Model...")
        worker = ModelLoadThread(text)
        worker.signals.result.connect(self._loadModel)
        worker.signals.finished.connect(self._loadFinished)
        self.threadpool.start(worker)

    def _loadModel(self, model):
        self.parent.setModel(model)
        print(f"Model loaded: {model}")

    def _loadFinished(self):
        button = self.parent.transcribe_button
        button.setEnabled(True)
        button.setText("Start")

    def loadAPI(self, text, api_key):
        if text == "DeepGram":
            self.parent.clients[text] = DeepgramClient(api_key)
        elif text == "AssemblyAI":
            aai.settings.api_key = api_key
            self.parent.clients[text] = aai.Transcriber()

    def checkAPIKey(self, text, api_key):
        print(f"Checking API Key: {text}")
        if text == "DeepGram":
            check_deepgram_api_key(api_key)
        elif text == "AssemblyAI":
            check_assemblyai_api_key(api_key)
