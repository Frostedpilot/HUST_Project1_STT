# THis file is used to clear the settings of the application for testing purposes.
from PyQt6.QtCore import QSettings

setting = QSettings("Frostedpilot", "STT_app")
setting.clear()
