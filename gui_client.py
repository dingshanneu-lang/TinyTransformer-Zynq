#!/usr/bin/env python3
"""
TinyTransformer-Zynq GUI 上位机
图形界面：上传图片 -> 预处理 -> 发送给 FPGA -> 接收结果 -> 显示分类
"""

import sys
import os
import numpy as np
from PIL import Image, ImageQt
from io import BytesIO

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tiny_transformer_client import (
    TinyTransformerClient, preprocess_image, classify_output,
    PROTOCOL_MAGIC, PKT_TYPE_HELLO, PKT_TYPE_CONFIG, PKT_TYPE_WEIGHTS,
    PKT_TYPE_INFERENCE, PKT_TYPE_RESULT, PKT_TYPE_ERROR,
    MAX_SEQ_LEN, MAX_EMBED_DIM, DEFAULT_PORT,
    crc32_calc, EMBED_MODELS_AVAILABLE
)

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QFileDialog, QTextEdit, QComboBox,
        QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QProgressBar,
        QGroupBox, QFormLayout, QMessageBox, QSplitter, QFrame
    )
    from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize
    from PySide6.QtGui import QPixmap, QImage, QFont, QIcon
    PYSIDE_VERSION = 6
except ImportError:
    try:
        from PyQt5.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QLabel, QPushButton, QFileDialog, QTextEdit, QComboBox,
            QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QProgressBar,
            QGroupBox, QFormLayout, QMessageBox, QSplitter, QFrame
        )
        from PyQt5.QtCore import Qt, QThread, Signal, QTimer, QSize
        from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
        PYSIDE_VERSION = 5
    except ImportError:
        print("请安装 PySide6 或 PyQt5: pip install PySide6")
        sys.exit(1)


class InferenceThread(QThread):
    """后台推理线程，避免阻塞 UI"""
    
    progress = Signal(str)      # 进度文本
    finished = Signal(object)   # 结果: (label, cat_prob, dog_prob, confidence, output_array)
    error = Signal(str)         # 错误信息
    
    def __init__(self, client_params, image_path, device='cpu'):
        super().__init__()
        self.client_params = client_params
        self.image_path = image_path
        self.device = device
        self._cancelled = False
    
    def cancel(self):
        self._cancelled = True
    
    def run(self):
        try:
            self.progress.emit("🔄 正在提取特征 (MobileNetV2 + proj)...")
            input_tensor = preprocess_image(self.image_path, device=self.device)
            
            if self._cancelled:
                return
            
            self.progress.emit("🔌 正在连接 FPGA 服务器...")
            client = TinyTransformerClient(
                self.client_params['host'],
                self.client_params['port'],
                self.client_params['timeout']
            )
            
            if not client.connect():
                self.error.emit("连接失败，请检查 IP 和端口")
                return
            
            if self._cancelled:
                client.close()
                return
            
            self.progress.emit("🤝 正在握手...")
            if not client.hello():
                self.error.emit("握手失败")
                client.close()
                return
            
            if self._cancelled:
                client.close()
                return
            
            if not self.client_params['skip_config']:
                self.progress.emit("⚙️ 发送模型配置...")
                if not client.send_config():
                    self.error.emit("配置发送失败")
                    client.close()
                    return
            
            if self._cancelled:
                client.close()
                return
            
            if not self.client_params['skip_weights']:
                self.progress.emit("📦 发送权重文件...")
                if not client.send_weights(self.client_params['weights_path']):
                    self.progress.emit("⚠️ 权重发送失败，继续尝试推理...")
            
            if self._cancelled:
                client.close()
                return
            
            self.progress.emit("🧠 正在推理...")
            output = client.inference(input_tensor)
            client.close()
            
            if output is None:
                self.error.emit("推理失败")
                return
            
            label, cat_prob, dog_prob, confidence = classify_output(output)
            
            self.finished.emit((label, cat_prob, dog_prob, confidence, output))
            
        except Exception as e:
            self.error.emit(f"发生错误: {str(e)}")


class ImageLabel(QLabel):
    """可点击的图片显示标签"""
    clicked = Signal()
    
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 300)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 8px;
                background-color: #f5f5f5;
                color: #888;
            }
            QLabel:hover {
                border-color: #4a90d9;
                background-color: #eef6fc;
            }
        """)
        self.setText("点击或拖拽上传图片\n支持 JPG/PNG/BMP")
        self.setAcceptDrops(True)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
    
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                QLabel {
                    border: 2px solid #4a90d9;
                    border-radius: 8px;
                    background-color: #d6eaff;
                    color: #4a90d9;
                }
            """)
    
    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 8px;
                background-color: #f5f5f5;
                color: #888;
            }
            QLabel:hover {
                border-color: #4a90d9;
                background-color: #eef6fc;
            }
        """)
    
    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.clicked.emit()
            # 通过父窗口处理文件
            window = self.window()
            if hasattr(window, 'load_image'):
                window.load_image(file_path)
        self.dragLeaveEvent(event)
    
    def set_image(self, pixmap: QPixmap):
        scaled = pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #4a90d9;
                border-radius: 8px;
                background-color: #fff;
            }
        """)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TinyTransformer-Zynq 猫狗分类上位机")
        self.resize(1000, 700)
        
        self.current_image_path = None
        self.inference_thread = None
        self.embed_model = None
        
        self.init_ui()
        self.load_default_weights_path()
    
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # 左侧：图片预览 + 控制
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_widget.setMaximumWidth(450)
        
        # 图片预览区
        self.image_label = ImageLabel()
        self.image_label.clicked.connect(self.select_image)
        left_layout.addWidget(self.image_label)
        
        # 图片信息
        self.info_label = QLabel("未加载图片")
        self.info_label.setStyleSheet("color: #666; font-size: 12px;")
        self.info_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.info_label)
        
        # 控制面板
        control_group = QGroupBox("控制面板")
        control_layout = QVBoxLayout(control_group)
        
        # 服务器设置
        server_group = QGroupBox("服务器连接")
        server_layout = QFormLayout(server_group)
        
        # 从环境变量读取 (main.py 传递)
        import os
        default_host = os.environ.get("FPGA_HOST", "192.168.0.102")
        default_port = int(os.environ.get("FPGA_PORT", str(DEFAULT_PORT)))
        
        self.host_edit = QLineEdit(default_host)
        self.host_edit.setPlaceholderText("Zynq PS IP 地址")
        server_layout.addRow("IP 地址:", self.host_edit)
        
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(default_port)
        server_layout.addRow("端口:", self.port_spin)
        
        self.timeout_spin = QDoubleSpinBox()
        self.timeout_spin.setRange(1, 60)
        self.timeout_spin.setValue(5.0)
        self.timeout_spin.setSuffix(" 秒")
        server_layout.addRow("超时:", self.timeout_spin)
        
        control_layout.addWidget(server_group)
        
        # 模型设置
        model_group = QGroupBox("模型设置")
        model_layout = QFormLayout(model_group)
        
        self.weights_edit = QLineEdit()
        self.weights_edit.setPlaceholderText("权重文件路径")
        self.weights_edit.setReadOnly(True)
        weights_btn = QPushButton("浏览...")
        weights_btn.clicked.connect(self.select_weights)
        weights_layout = QHBoxLayout()
        weights_layout.addWidget(self.weights_edit)
        weights_layout.addWidget(weights_btn)
        model_layout.addRow("权重文件:", weights_layout)
        
        self.skip_weights_check = QCheckBox("跳过发送权重 (FPGA 已预加载)")
        self.skip_weights_check.setChecked(False)
        model_layout.addRow("", self.skip_weights_check)
        
        self.skip_config_check = QCheckBox("跳过发送配置 (使用默认配置)")
        self.skip_config_check.setChecked(False)
        model_layout.addRow("", self.skip_config_check)
        
        # 特征提取 (与训练一致: MobileNetV2 features + proj + scale -> Q12)
        self.embed_device_combo = QComboBox()
        self.embed_device_combo.addItems(["cpu", "cuda"])
        model_layout.addRow("特征设备:", self.embed_device_combo)

        hint = QLabel("特征链: MobileNetV2→proj→×scale→Q12\n权重: xform_weights.bin (522 int16)")
        hint.setStyleSheet("color: #888; font-size: 11px;")
        hint.setWordWrap(True)
        model_layout.addRow("", hint)

        control_layout.addWidget(model_group)
        
        # 推理按钮
        self.infer_btn = QPushButton("🚀 开始推理")
        self.infer_btn.setMinimumHeight(45)
        self.infer_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a90d9;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #357abd; }
            QPushButton:pressed { background-color: #2a65a0; }
            QPushButton:disabled { background-color: #ccc; color: #888; }
        """)
        self.infer_btn.clicked.connect(self.start_inference)
        control_layout.addWidget(self.infer_btn)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 无限进度条
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_bar)
        
        # 状态文本
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #666; font-size: 12px;")
        self.status_label.setWordWrap(True)
        control_layout.addWidget(self.status_label)
        
        left_layout.addWidget(control_group)
        left_layout.addStretch()
        
        # 右侧：结果显示 + 日志
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 结果显示
        result_group = QGroupBox("分类结果")
        result_layout = QVBoxLayout(result_group)
        
        self.result_label = QLabel("等待推理...")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: #333;
                padding: 20px;
            }
        """)
        result_layout.addWidget(self.result_label)
        
        # 概率条
        prob_widget = QWidget()
        prob_layout = QVBoxLayout(prob_widget)
        prob_layout.setSpacing(10)
        
        self.cat_bar = QProgressBar()
        self.cat_bar.setRange(0, 100)
        self.cat_bar.setFormat("猫: %p%")
        self.cat_bar.setStyleSheet("""
            QProgressBar { border: 1px solid #ccc; border-radius: 4px; text-align: center; height: 25px; }
            QProgressBar::chunk { background-color: #4caf50; border-radius: 3px; }
        """)
        prob_layout.addWidget(self.cat_bar)
        
        self.dog_bar = QProgressBar()
        self.dog_bar.setRange(0, 100)
        self.dog_bar.setFormat("狗: %p%")
        self.dog_bar.setStyleSheet("""
            QProgressBar { border: 1px solid #ccc; border-radius: 4px; text-align: center; height: 25px; }
            QProgressBar::chunk { background-color: #ff9800; border-radius: 3px; }
        """)
        prob_layout.addWidget(self.dog_bar)
        
        result_layout.addWidget(prob_widget)
        
        # 详细输出
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setMaximumHeight(150)
        self.detail_text.setFont(QFont("Consolas", 10))
        result_layout.addWidget(QLabel("FPGA 原始输出 (int16):"))
        result_layout.addWidget(self.detail_text)
        
        right_layout.addWidget(result_group)
        
        # 日志区域
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4;")
        log_layout.addWidget(self.log_text)
        
        # 清空日志按钮
        clear_log_btn = QPushButton("清空日志")
        clear_log_btn.clicked.connect(self.log_text.clear)
        log_layout.addWidget(clear_log_btn)
        
        right_layout.addWidget(log_group)
        
        # 分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 600])
        main_layout.addWidget(splitter)
    
    def load_default_weights_path(self):
        """尝试加载默认权重路径"""
        default_paths = [
            os.path.join(os.path.dirname(__file__), "weights", "xform_weights.bin"),
            os.path.join(os.path.dirname(__file__), "xform_weights.bin"),
            os.path.join(os.path.dirname(__file__), "weights", "weights.bin"),
        ]
        for p in default_paths:
            if os.path.exists(p):
                self.weights_edit.setText(p)
                size = os.path.getsize(p)
                self.log(f"[+] 自动加载权重: {p} ({size} bytes)")
                if size != 1044:
                    self.log(f"[!] 注意: 权重应为 1044 bytes (522 int16), 当前 {size}")
                break
    
    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp *.tiff)"
        )
        if file_path:
            self.load_image(file_path)
    
    def load_image(self, file_path):
        try:
            pixmap = QPixmap(file_path)
            if pixmap.isNull():
                QMessageBox.warning(self, "错误", "无法加载图片")
                return
            
            self.current_image_path = file_path
            self.image_label.set_image(pixmap)
            
            # 显示图片信息
            img = Image.open(file_path)
            self.info_label.setText(
                f"{os.path.basename(file_path)} | {img.size[0]}×{img.size[1]} | {img.mode} | "
                f"{os.path.getsize(file_path)/1024:.1f} KB"
            )
            
            self.log(f"[+] 已加载图片: {file_path}")
            self.result_label.setText("等待推理...")
            self.cat_bar.setValue(0)
            self.dog_bar.setValue(0)
            self.detail_text.clear()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图片失败: {e}")
    
    def select_weights(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择权重文件", "", "二进制文件 (*.bin);;所有文件 (*)"
        )
        if file_path:
            self.weights_edit.setText(file_path)
            self.log(f"[+] 选择权重文件: {file_path}")
    
    def log(self, message):
        """添加日志"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def start_inference(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "提示", "请先选择图片")
            return
        
        weights_path = self.weights_edit.text().strip()
        if not self.skip_weights_check.isChecked() and not weights_path:
            QMessageBox.warning(self, "提示", "请选择权重文件或勾选'跳过发送权重'")
            return
        
        if not self.skip_weights_check.isChecked() and not os.path.exists(weights_path):
            QMessageBox.warning(self, "提示", "权重文件不存在")
            return
        
        # 禁用按钮，显示进度
        self.infer_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_label.setText("正在推理...")
        
        # 特征提取设备
        device = self.embed_device_combo.currentText()
        
        # 启动后台线程
        client_params = {
            'host': self.host_edit.text().strip(),
            'port': self.port_spin.value(),
            'timeout': self.timeout_spin.value(),
            'weights_path': weights_path,
            'skip_weights': self.skip_weights_check.isChecked(),
            'skip_config': self.skip_config_check.isChecked(),
        }
        
        self.inference_thread = InferenceThread(client_params, self.current_image_path, device)
        self.inference_thread.progress.connect(self.on_progress)
        self.inference_thread.finished.connect(self.on_finished)
        self.inference_thread.error.connect(self.on_error)
        self.inference_thread.start()
    
    def on_progress(self, message):
        self.log(message)
        self.status_label.setText(message)
    
    def on_finished(self, result):
        label, cat_prob, dog_prob, confidence, output = result
        
        # 更新结果显示
        color = "#4caf50" if label == "猫" else "#ff9800"
        self.result_label.setText(f"{label}")
        self.result_label.setStyleSheet(f"""
            QLabel {{
                font-size: 36px;
                font-weight: bold;
                color: {color};
                padding: 20px;
            }}
        """)
        
        self.cat_bar.setValue(int(cat_prob * 100))
        self.dog_bar.setValue(int(dog_prob * 100))
        
        # 显示原始输出
        output_str = np.array2string(output, separator=', ', prefix='  ')
        self.detail_text.setPlainText(output_str)
        
        self.log(f"[✓] 推理完成: {label} (置信度: {confidence*100:.1f}%)")
        self.log(f"    猫: {cat_prob*100:.1f}% | 狗: {dog_prob*100:.1f}%")
        
        self.infer_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("就绪")
    
    def on_error(self, message):
        self.log(f"[✗] {message}")
        QMessageBox.critical(self, "推理错误", message)
        self.infer_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("错误")
    
    def closeEvent(self, event):
        if self.inference_thread and self.inference_thread.isRunning():
            self.inference_thread.cancel()
            self.inference_thread.wait(2000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # 设置应用字体
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()