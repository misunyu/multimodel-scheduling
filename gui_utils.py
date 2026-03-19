import os
import yaml
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileSystemModel, QDialog, QVBoxLayout, QListWidget, QListWidgetItem, QPushButton, QHBoxLayout

class ChangeDeployDialog(QDialog):
    """Modeless dialog to show and select generated combinations from model_schedules.yaml."""

    def __init__(self, parent, schedule_path):
        super().__init__(parent)
        self.setWindowTitle("Change Deployment")
        self.setModal(False)  # Modeless
        self.schedule_path = schedule_path
        self.parent_app = parent

        layout = QVBoxLayout(self)
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayout()
        self.execute_button = QPushButton("Execute Selected")
        self.execute_button.clicked.connect(self.on_execute_clicked)
        btn_layout.addWidget(self.execute_button)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        btn_layout.addWidget(self.close_button)

        layout.addLayout(btn_layout)
        self.load_combinations()

    def load_combinations(self):
        self.list_widget.clear()
        if not os.path.exists(self.schedule_path):
            self.list_widget.addItem("No schedule file found.")
            return

        try:
            with open(self.schedule_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            if not data:
                self.list_widget.addItem("Empty schedule file.")
                return

            for combo_name in sorted(data.keys()):
                # Format: combination_1 (model1_cpu, model2_gpu)
                models_info = []
                for mid, info in data[combo_name].items():
                    models_info.append(f"{info.get('model')}:{info.get('execution')}")
                display_text = f"{combo_name} ({', '.join(models_info)})"
                item = QListWidgetItem(display_text)
                item.setData(Qt.UserRole, combo_name)
                self.list_widget.addItem(item)
        except Exception as e:
            self.list_widget.addItem(f"Error loading: {e}")

    def on_execute_clicked(self):
        selected_item = self.list_widget.currentItem()
        if selected_item:
            combo_name = selected_item.data(Qt.UserRole)
            if combo_name and combo_name.startswith('combination_'):
                self.parent_app.log(f"[Action] Manually selected {combo_name} for execution.")
                self.parent_app._kill_existing_executor()
                self.parent_app._launch_executor_direct(self.schedule_path, combo_name=combo_name, duration=60)


class CheckableFileSystemModel(QFileSystemModel):
    """A QFileSystemModel that adds a checkbox to its items."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._check_states = {}  # {abs_path: Qt.CheckState}
        self._root_index = None

    def set_root_index(self, index):
        self._root_index = index

    def is_top_level_child(self, index):
        """Returns True if the index is an immediate child of our root index."""
        if not self._root_index or not self._root_index.isValid():
            return False
        return index.parent() == self._root_index

    def flags(self, index):
        base_flags = super().flags(index)
        if index.isValid() and self.is_top_level_child(index):
            return base_flags | Qt.ItemIsUserCheckable
        return base_flags

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.CheckStateRole and self.is_top_level_child(index):
            path = self.filePath(index)
            return self._check_states.get(path, Qt.Unchecked)
        return super().data(index, role)

    def setData(self, index, value, role=Qt.EditRole):
        if role == Qt.CheckStateRole and self.is_top_level_child(index):
            path = self.filePath(index)
            self._check_states[path] = Qt.Checked if value == Qt.Checked else Qt.Unchecked
            self.dataChanged.emit(index, index, [Qt.CheckStateRole])
            return True
        return super().setData(index, value, role)

    def get_checked_top_level_dirs(self):
        """Return absolute paths of checked immediate child directories under the current root."""
        return [path for path, state in self._check_states.items() if state == Qt.Checked]
