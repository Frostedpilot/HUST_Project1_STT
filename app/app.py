import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QWidget,
    QTextEdit,
    QButtonGroup,
    QRadioButton,
    QFileDialog,
)
from PyQt6.QtCore import Qt
from ModelComboBox import ModelComboBox
from LanguageComboBox import LanguageComboBox
from utility import (
    transcribe_whisper,
    transcribe_wav2vec,
    transcribe_deepgram,
    transcribe_assemblyai,
)
from deepgram import DeepgramClient
import assemblyai as aai


class MyMainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("App")
        self.model = None
        self.language_dict = {"English": "en", "Vietnamese": "vi", "Auto": None}
        self.clients = {"DeepGram": None, "AssemblyAI": None}

        # The main scene
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Add widgets
        self.add_model_combobox_part()
        self.add_language_combobox_part()
        self.add_input_selection()
        self.add_file_chooser()
        self.add_buttons()
        self.add_transcript_output()

    def add_model_combobox_part(self):
        main_layout = self.centralWidget().layout()
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignRight)
        label = QLabel("Model")
        self.model_combobox = ModelComboBox(parent=self)
        layout.addWidget(label)
        layout.addWidget(self.model_combobox)
        main_layout.addLayout(layout)

    def add_language_combobox_part(self):
        main_layout = self.centralWidget().layout()
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignRight)
        label = QLabel("Language")
        self.language_combobox = LanguageComboBox()
        layout.addWidget(label)
        layout.addWidget(self.language_combobox)
        main_layout.addLayout(layout)

    def add_input_selection(self):
        main_layout = self.centralWidget().layout()
        layout = QHBoxLayout()
        self.source_group = QButtonGroup()
        # self.youtube_radio = QRadioButton("YouTube Link")
        self.local_file_radio = QRadioButton("Local File")
        # self.source_group.addButton(self.youtube_radio)
        self.source_group.addButton(self.local_file_radio)
        self.local_file_radio.setChecked(True)  # Default selection
        # layout.addWidget(self.youtube_radio)
        layout.addWidget(self.local_file_radio)
        main_layout.addLayout(layout)

    def add_file_chooser(self):
        main_layout = self.centralWidget().layout()
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.file_chooser_button = QPushButton("Choose File")
        self.file_chooser_button.clicked.connect(self.choose_file)
        self.file_chooser_label = QLabel("No file chosen")
        layout.addWidget(self.file_chooser_button)
        layout.addWidget(self.file_chooser_label)
        main_layout.addLayout(layout)

    def add_buttons(self):
        main_layout = self.centralWidget().layout()
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignRight)
        button = QPushButton("Start")
        button.clicked.connect(self.transcribe)
        layout.addWidget(button)
        main_layout.addLayout(layout)

    def add_transcript_output(self):
        main_layout = self.centralWidget().layout()
        self.transcript_text_edit = QTextEdit()
        self.transcript_text_edit.setReadOnly(True)  # Make it read-only initially
        self.transcript_text_edit.setLineWrapMode(
            QTextEdit.LineWrapMode.WidgetWidth
        )  # Enable word wrap
        main_layout.addWidget(self.transcript_text_edit)

    def transcribe(self):
        file_path = self.file_chooser_label.text()
        if not file_path:
            self.transcript_text_edit.setPlainText("Please choose a file")
            return
        model_name = self.model_combobox.currentText()
        if model_name == "DeepGram":
            if not self.clients["DeepGram"]:
                self.clients["DeepGram"] = DeepgramClient(
                    api_key=self.model_combobox.api_keys["DeepGram"]
                )
            result = transcribe_deepgram(self.clients["DeepGram"], file_path)
            self.transcript_text_edit.setPlainText(result)
            return
        elif model_name == "AssemblyAI":
            if not self.clients["AssemblyAI"]:
                aai.settings.api_key = self.model_combobox.api_keys["AssemblyAI"]
                self.clients["AssemblyAI"] = aai.Transcriber()
            result = transcribe_assemblyai(self.clients["AssemblyAI"], file_path)
            self.transcript_text_edit.setPlainText(result)
            return
        language = self.language_dict[self.language_combobox.currentText()]
        if not self.model:
            self.transcript_text_edit.setPlainText("Please choose a model")
            return
        if model_name.split(":")[0] == "OpenAI Whisper":
            result = transcribe_whisper(self.model, file_path, language)
        elif model_name.split(":")[0] == "Facebook Wav2Vec":
            result = transcribe_wav2vec(self.model, file_path)
        self.transcript_text_edit.setPlainText(result)

    def choose_file(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilters(
            ["Audio files (*.wav *.mp3)", "Video files (*.mp4 *.mkv *.avi)"]
        )
        dialog.setViewMode(QFileDialog.ViewMode.List)
        dialog.setOption(QFileDialog.Option.ReadOnly)
        dialog.setWindowTitle("Choose a file")

        if dialog.exec():
            selected_files = dialog.selectedFiles()
            if selected_files:
                self.file_chooser_label.setText(selected_files[0])
                return selected_files[0]
        return None

    def setModel(self, model):
        self.model = model


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec())
