from PyQt6.QtWidgets import QComboBox


class LanguageComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.addItem("English")
        self.addItem("Vietnamese")
        self.addItem("Auto")
        self.currentTextChanged.connect(self.textChanged)

    def textChanged(self, text):
        print(f"Language changed to: {text}")
