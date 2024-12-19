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
from PyQt6.QtCore import Qt, QThreadPool
from ModelComboBox import ModelComboBox
from LanguageComboBox import LanguageComboBox
from utility import TranscribeThread


class MyMainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("App")
        self.model = None
        self.language_dict = {"English": "en", "Vietnamese": "vi", "Auto": None}
        self.clients = {"DeepGram": None, "AssemblyAI": None}
        self.threadpool = QThreadPool()

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
        self.transcribe_button = QPushButton("Start")
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
        main_layout.addWidget(self.transcript_text_edit)

    def transcribe(self):
        self.transcript_text_edit.setText("")

        file_path = self.file_chooser_label.text()
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
