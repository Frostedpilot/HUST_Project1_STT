import os
from PyQt6.QtWidgets import (
    QStackedWidget,
    QRadioButton,
    QButtonGroup,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QWidget,
    QFileDialog,
    QLineEdit,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QSize
from yt_dlp import YoutubeDL
from yt_dlp.extractor import list_extractors
from utility import YoutubeDLThread


class FileChooseWidget(QWidget):
    def __init__(self, setting, parent=None):
        super().__init__(parent)
        self.file = None
        self.setting = setting
        if parent:
            self.threadpool = parent.threadpool
        self.BASE_DIR = self.setting.value("BASE_DIR")
        print("FileChooseWidget BASE_DIR:", self.BASE_DIR)

        main_layout = QVBoxLayout()

        self.upper_widget = QWidget()

        layout = QHBoxLayout()
        self.source_group = QButtonGroup()
        self.youtube_radio = QRadioButton("YouTube Link")
        self.local_file_radio = QRadioButton("Local File")
        self.source_group.addButton(self.youtube_radio)
        self.source_group.addButton(self.local_file_radio)
        self.local_file_radio.setChecked(True)  # Default selection
        layout.addWidget(self.youtube_radio)
        layout.addWidget(self.local_file_radio)

        self.upper_widget.setLayout(layout)

        self.lower_widget = QStackedWidget()

        local_file_widget = QWidget()
        layout_local = QHBoxLayout()
        layout_local.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.file_chooser_button = QPushButton("Choose File")
        self.file_chooser_button.setMinimumSize(100, 30)
        self.file_chooser_button.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum
        )
        self.file_chooser_button.clicked.connect(self.choose_file)
        self.file_chooser_label = QLabel("No file chosen")
        self.file_chooser_label.setMinimumSize(200, 30)
        self.file_chooser_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.file_chooser_label.setWordWrap(True)
        layout_local.addWidget(self.file_chooser_button)
        layout_local.addWidget(self.file_chooser_label)
        local_file_widget.setLayout(layout_local)

        youtube_widget = QWidget()
        layout_youtube = QVBoxLayout()
        layout_youtube.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.youtube_link_input = QLineEdit()
        self.youtube_link_input.setPlaceholderText("Enter a YouTube link")
        self.youtube_link_input.setMinimumSize(200, 30)
        self.youtube_link_input.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum
        )
        self.youtube_link_label = QLabel("No link entered")
        self.youtube_link_label.setMinimumSize(200, 30)
        self.youtube_link_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.youtube_link_label.setWordWrap(True)
        self.youtube_link_input.returnPressed.connect(self.test_yt_link)
        layout_youtube.addWidget(self.youtube_link_input)
        layout_youtube.addWidget(self.youtube_link_label)
        youtube_widget.setLayout(layout_youtube)

        self.lower_widget.addWidget(local_file_widget)
        self.lower_widget.addWidget(youtube_widget)

        self.youtube_radio.toggled.connect(self.flip_youtube_widget)
        self.local_file_radio.toggled.connect(self.flip_local_widget)

        main_layout.addWidget(self.upper_widget)
        main_layout.addWidget(self.lower_widget)
        self.setLayout(main_layout)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

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
                self.file = selected_files[0]
                return selected_files[0]
        return None

    def test_yt_link(self):
        url = self.youtube_link_input.text()
        ies = list_extractors()
        extractor = next(
            (
                ie.ie_key()
                for ie in ies
                if ie.suitable(url) and ie.ie_key() != "Generic"
            ),
            None,
        )
        if extractor:
            self.download_yt_link(url)
            self.youtube_link_input.setText("")
            self.youtube_link_input.setPlaceholderText("Enter a YouTube link")
            return
        self.youtube_link_input.setText("")
        self.youtube_link_input.setPlaceholderText("Invalid YouTube link")
        return False

    def download_yt_link(self, url):
        self.youtube_link_input.setPlaceholderText("Downloading...")
        self.youtube_link_label.setText("Downloading...")
        self.youtube_link_input.setDisabled(True)
        worker = YoutubeDLThread(url)
        worker.signals.result.connect(self._yt_download_finished)
        worker.signals.error.connect(self._yt_download_failed)
        worker.signals.finished.connect(
            lambda: self.youtube_link_input.setDisabled(False)
        )
        self.threadpool.start(worker)

    def _yt_download_finished(self, file, url):
        self.file = file
        self.youtube_link_label.setText("<b>Chosen YouTube link</b>: " + url)

    def _yt_download_failed(self, error):
        self.youtube_link_input.setText("")
        self.youtube_link_input.setPlaceholderText("Error: " + error)
        self.youtube_link_label.setText("No link entered")

    def flip_local_widget(self):
        self.lower_widget.setCurrentIndex(0)
        self.file = None
        self.file_chooser_label.setText("No file chosen")

    def flip_youtube_widget(self):
        self.lower_widget.setCurrentIndex(1)
        self.file = None
        self.youtube_link_label.setText("No link entered")
