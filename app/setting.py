from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLineEdit,
    QLabel,
    QCheckBox,
)
from PyQt6.QtCore import Qt
from utility import check_assemblyai_api_key, check_deepgram_api_key
from deepgram import DeepgramClient
import assemblyai as aai


class SettingDialog(QDialog):
    def __init__(self, setting, parent=None):
        super().__init__(parent)
        self.setting = setting

        self.setWindowTitle("Setting")

        # The main scene
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Add widgets

        # Whisper settings
        whisper_widget_label = QLabel("Whisper Settings")
        main_layout.addWidget(whisper_widget_label)
        whisper_vad_widget = QWidget()
        whisper_vad_layout = QHBoxLayout()
        whisper_vad_widget.setLayout(whisper_vad_layout)
        whisper_vad_label = QLabel("VAD:")
        self.whisper_vad = QCheckBox()
        self.whisper_vad.checkStateChanged.connect(self.change_whisper_vad_setting)
        whisper_vad_layout.addWidget(whisper_vad_label)
        whisper_vad_layout.addWidget(self.whisper_vad)
        main_layout.addWidget(whisper_vad_widget)

        # Wav2Vec settings
        wav2vec_widget_label = QLabel("Wav2Vec Settings")
        main_layout.addWidget(wav2vec_widget_label)
        wav2vec_vad_widget = QWidget()
        wav2vec_vad_layout = QHBoxLayout()
        wav2vec_vad_widget.setLayout(wav2vec_vad_layout)
        wav2vec_vad_label = QLabel("VAD:")
        self.wav2vec_vad = QCheckBox()
        self.wav2vec_vad.checkStateChanged.connect(self.change_w2v_vad_setting)
        wav2vec_vad_layout.addWidget(wav2vec_vad_label)
        wav2vec_vad_layout.addWidget(self.wav2vec_vad)
        main_layout.addWidget(wav2vec_vad_widget)

        # Deepgram settings
        deepgram_widget_label = QLabel("DeepGram Settings")
        main_layout.addWidget(deepgram_widget_label)
        deepgram_widget = QWidget()
        deepgram_layout = QHBoxLayout()
        deepgram_widget.setLayout(deepgram_layout)
        deepgram_label = QLabel("DeepGram API Key:")
        self.deepgram_api = QLineEdit()
        deepgram_layout.addWidget(deepgram_label)
        deepgram_layout.addWidget(self.deepgram_api)
        self.deepgram_api.setEchoMode(QLineEdit.EchoMode.Password)
        self.deepgram_api.returnPressed.connect(self._check_deepgram_api_key)
        main_layout.addWidget(deepgram_widget)

        # AssemblyAI settings
        assemblyai_widget_label = QLabel("AssemblyAI Settings")
        main_layout.addWidget(assemblyai_widget_label)
        assemblyai_widget = QWidget()
        assemblyai_layout = QHBoxLayout()
        assemblyai_widget.setLayout(assemblyai_layout)
        assemblyai_label = QLabel("AssemblyAI API Key:")
        self.assemblyai_api = QLineEdit()
        assemblyai_layout.addWidget(assemblyai_label)
        assemblyai_layout.addWidget(self.assemblyai_api)
        self.assemblyai_api.setEchoMode(QLineEdit.EchoMode.Password)
        self.assemblyai_api.returnPressed.connect(self._check_assemblyai_api_key)
        main_layout.addWidget(assemblyai_widget)

        self.load_settings()

    def load_settings(self):
        self.setting.beginGroup("Whisper")
        whisper_vad = self.setting.value("vad")
        if whisper_vad:
            self.whisper_vad.setChecked(whisper_vad)
        self.setting.endGroup()

        self.setting.beginGroup("Wav2Vec")
        w2v_vad = self.setting.value("vad")
        if w2v_vad:
            self.wav2vec_vad.setChecked(w2v_vad)
        self.setting.endGroup()

        self.setting.beginGroup("DeepGram")
        deepgram_api_key = self.setting.value("api_key")
        if deepgram_api_key:
            self.deepgram_api.setText(deepgram_api_key)
        self.setting.endGroup()

        self.setting.beginGroup("AssemblyAI")
        assemblyai_api_key = self.setting.value("api_key")
        if assemblyai_api_key:
            self.assemblyai_api.setText(assemblyai_api_key)
        self.setting.endGroup()

    def change_whisper_vad_setting(self, state):
        state = True if state == Qt.CheckState.Checked else False
        self.setting.setValue("Whisper/vad", state)

    def change_w2v_vad_setting(self, state):
        state = True if state == Qt.CheckState.Checked else False
        self.setting.setValue("Wav2Vec/vad", state)

    def _check_deepgram_api_key(self):
        api_key = self.deepgram_api.text()
        if check_deepgram_api_key(api_key):
            self.setting.setValue("DeepGram/api_key", api_key)
            client = DeepgramClient(api_key)
            self.parent().clients["DeepGram"] = client
            return True
        self.deepgram_api.clear()
        return False

    def _check_assemblyai_api_key(self):
        api_key = self.assemblyai_api.text()
        if check_assemblyai_api_key(api_key):
            self.setting.setValue("AssemblyAI/api_key", api_key)
            aai.settings.api_key = api_key
            self.parent().clients["AssemblyAI"] = aai.Transcriber()
            return True
        self.assemblyai_api.clear()
        return False
