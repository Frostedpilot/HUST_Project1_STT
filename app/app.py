import sys
import os
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QWidget,
    QTextEdit,
    QSizePolicy,
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt, QThreadPool, QSize, QSettings
from ModelComboBox import ModelComboBox
from LanguageComboBox import LanguageComboBox
from utility import TranscribeThread
from FileChooseWidget import FileChooseWidget
from multiprocessing import freeze_support
from setting import SettingDialog
from deepgram import DeepgramClient
import assemblyai as aai


class MyMainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("App")
        self.load_style("dark")

        # Initialize variables and objects needed for the app
        self.language_dict = {"English": "en", "Vietnamese": "vi", "Auto": None}
        self.model = None
        self.clients = {"DeepGram": None, "AssemblyAI": None}
        self.threadpool = QThreadPool()
        self.settings = QSettings("Frostedpilot", "STT_app")

        # The main scene
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Add widgets
        self.add_model_combobox_part()
        self.add_language_combobox_part()
        self.add_input_selection()
        self.add_buttons()
        self.add_transcript_output()
        self.setup_menu()

        # Size policy for the main window
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        self.load_settings()

    def load_settings(self):
        self.settings.beginGroup("DeepGram")
        deepgram_api_key = self.settings.value("api_key")
        if deepgram_api_key:
            self.clients["DeepGram"] = DeepgramClient(deepgram_api_key)
        self.settings.endGroup()

        self.settings.beginGroup("AssemblyAI")
        assemblyai_api_key = self.settings.value("api_key")
        if assemblyai_api_key:
            aai.settings.api_key = assemblyai_api_key
            self.clients["AssemblyAI"] = aai.Transcriber()
        self.settings.endGroup()

    def setup_menu(self):
        # Create a "Theme" menu
        self.theme_menu = self.menuBar().addMenu("&Theme")

        # Dark theme action
        self.dark_theme_action = QAction("&Dark Theme", self)
        self.dark_theme_action.triggered.connect(lambda: self.load_style("dark"))
        self.theme_menu.addAction(self.dark_theme_action)

        # Light theme action
        self.light_theme_action = QAction("&Light Theme", self)
        self.light_theme_action.triggered.connect(lambda: self.load_style("light"))
        self.theme_menu.addAction(self.light_theme_action)

        # Create a "Setting" menu
        self.setting_menu = self.menuBar().addMenu("&Setting")
        self.setting_action = QAction("&Setting", self)
        self.setting_action.triggered.connect(self.open_setting)
        self.setting_menu.addAction(self.setting_action)

    def open_setting(self):
        setting_dialog = SettingDialog(setting=self.settings, parent=self)
        setting_dialog.exec()

    def load_style(self, theme_name):
        # Load common styles
        with open("styles/common.qss", "r") as f:
            common_style = f.read()

        # Load theme-specific styles
        try:
            with open(f"styles/{theme_name}_theme.qss", "r") as f:
                theme_style = f.read()
        except FileNotFoundError:
            print(f"Warning: Theme file not found: styles/{theme_name}_theme.qss")
            theme_style = ""

        # Apply the combined styles
        self.setStyleSheet(common_style + theme_style)

    def add_model_combobox_part(self):
        main_layout = self.centralWidget().layout()
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignRight)
        label = QLabel("Model")
        label.setMinimumSize(QSize(60, 30))
        self.model_combobox = ModelComboBox(setting=self.settings, parent=self)
        self.model_combobox.setMinimumSize(QSize(200, 30))
        self.model_combobox.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(label)
        layout.addWidget(self.model_combobox)
        main_layout.addLayout(layout)

    def add_language_combobox_part(self):
        main_layout = self.centralWidget().layout()
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignRight)
        label = QLabel("Language")
        label.setMinimumSize(QSize(60, 30))
        self.language_combobox = LanguageComboBox()
        self.language_combobox.setMinimumSize(QSize(200, 30))
        self.language_combobox.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(label)
        layout.addWidget(self.language_combobox)
        main_layout.addLayout(layout)

    def add_input_selection(self):
        main_layout = self.centralWidget().layout()
        self.file_chooser_widget = FileChooseWidget()
        main_layout.addWidget(self.file_chooser_widget)

    def add_buttons(self):
        main_layout = self.centralWidget().layout()
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.transcribe_button = QPushButton("Start")
        self.transcribe_button.setMinimumSize(QSize(100, 30))
        self.transcribe_button.clicked.connect(self.transcribe)
        layout.addWidget(self.transcribe_button)
        main_layout.addLayout(layout)

    def add_transcript_output(self):
        main_layout = self.centralWidget().layout()
        self.transcript_text_edit = QTextEdit()
        self.transcript_text_edit.setReadOnly(True)  # Make it read-only initially
        self.transcript_text_edit.setLineWrapMode(
            QTextEdit.LineWrapMode.WidgetWidth
        )  # Enable word wrap
        self.transcript_text_edit.setMinimumSize(QSize(400, 300))

        self.transcript_text_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        main_layout.addWidget(self.transcript_text_edit)

    def transcribe(self):
        self.transcript_text_edit.setText("")

        file_path = self.file_chooser_widget.file
        language = self.language_dict[self.language_combobox.currentText()]
        if not file_path:
            self.transcript_text_edit.setPlainText("Please choose a file first")
            return
        model_name = self.model_combobox.currentText()
        if model_name in ["DeepGram", "AssemblyAI"]:
            api_key = self.model_combobox.api_keys.get(model_name)
            if not api_key:
                self.transcript_text_edit.setPlainText("Please enter API Key first")
                return
        elif not self.model:
            self.transcript_text_edit.setPlainText("Please choose a model first")
            return
        if not language:
            self.transcript_text_edit.setPlainText("Please choose a language first")
            return
        worker = TranscribeThread(
            model_name=model_name,
            model=self.model,
            audio_path=file_path,
            language=language,
            clients=self.clients,
            whisper_vad=self.settings.value("Whisper/vad"),
            w2v_vad=self.settings.value("Wav2Vec/vad"),
        )
        worker.signals.result.connect(self.transcript_result)
        worker.signals.finished.connect(self.reenable_button)
        worker.signals.segment_added.connect(self.append_transcribe)
        self.threadpool.start(worker)

        # Disable the start button
        self.transcribe_button.setText("Transcribing...")
        self.transcribe_button.setEnabled(False)

    def transcript_result(self, result):
        self.transcript_text_edit.setPlainText(result)

    def reenable_button(self):
        self.transcribe_button.setText("Start")
        self.transcribe_button.setEnabled(True)

    def append_transcribe(self, text):
        self.transcript_text_edit.append(text + " ")

    def setModel(self, model):
        self.model = model


if __name__ == "__main__":
    # Bug fix for multiprocessing on Windows when using PyInstaller
    freeze_support()

    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    try:
        code = app.exec()
    except Exception as e:
        print(e)

    # Clean up all created folders during the app execution
    import shutil

    shutil.rmtree("res", ignore_errors=True)
    shutil.rmtree("downloads", ignore_errors=True)
    shutil.rmtree("chunks", ignore_errors=True)
    if os.path.exists("speech.wav"):
        os.remove("speech.wav")
    sys.exit(code)
