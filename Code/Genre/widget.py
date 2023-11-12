#Python lib
import sys
import os
import random
import phonon
import cv2
import numpy as np
import dlib
import matplotlib

#QT lib
import PyQt6.QtCore
import PyQt6.QtGui
import PyQt6.QtWidgets
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtMultimedia import *
from PyQt6.QtMultimediaWidgets import *

#Our lib
from Swap import *
from CNN import *
from Morph import *
from Analyse import *

_Swap = Swap()
_CNN = CNN()
_Morph = Morph()
_Analyse = Analyse()

class MainApplication(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Processing App")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QStackedWidget(self)
        self.setCentralWidget(self.central_widget)

        self.image_files = []
        self.current_image_index = -1

        self.init_tabs()

    def init_tabs(self):
        self.tabs = QTabWidget(self)

        self.swap_tab = QWidget(self)
        self.morph_tab = QWidget(self)
        self.cnn_tab = QWidget(self)
        self.analyze_tab = QWidget(self)
        self.help_tab = QTextBrowser(self)

        self.tabs.addTab(self.swap_tab, "Swap")
        self.tabs.addTab(self.morph_tab, "Morph")
        self.tabs.addTab(self.cnn_tab, "CNN")
        self.tabs.addTab(self.analyze_tab, "Analyze")
        self.tabs.addTab(self.help_tab, "Help")

        self.central_widget.addWidget(self.tabs)

        #Swap onglet
        ##################################################

        self.swap_gauche_open_button = QPushButton("Open Source", self.swap_tab)
        self.swap_droite_open_button = QPushButton("Open Cible", self.swap_tab)
        self.swap_save_button = QPushButton("Save", self.swap_tab)
        self.swap_swap_button = QPushButton("Swap", self.swap_tab)
        self.swap_mp4_button = QPushButton("MP4", self.swap_tab)
        self.swap_gif_button = QPushButton("Gif", self.swap_tab)

        fps_layout = QVBoxLayout()
        frames_layout = QVBoxLayout()
        fps_layout.setSpacing(0)
        frames_layout.setSpacing(0)

        swap_fps_label = QLabel("FPS : ")
        swap_frames_label = QLabel("Frames")

        self.swap_fps_spinbox = QSpinBox()
        self.swap_fps_spinbox.setRange(0, 120)
        self.swap_fps_spinbox.setValue(15)
        self.swap_fps_spinbox.setSingleStep(1)

        self.swap_frames_spinbox = QSpinBox()
        self.swap_frames_spinbox.setRange(0, 150)
        self.swap_frames_spinbox.setValue(100)
        self.swap_frames_spinbox.setSingleStep(1)

        fps_layout.addWidget(swap_fps_label)
        frames_layout.addWidget(swap_frames_label)
        fps_layout.addWidget(self.swap_fps_spinbox)
        frames_layout.addWidget(self.swap_frames_spinbox)

        fps_layout.setSpacing(0)
        frames_layout.setSpacing(0)

        self.swap_placeholder = QLabel(self.swap_tab)
        self.swap_placeholder.setAutoFillBackground(True)
        self.swap_pal = self.swap_placeholder.palette()
        self.swap_pal.setColor(QPalette.ColorRole.Window, QColor(0, 255, 0))
        self.swap_placeholder.setPalette(self.swap_pal)

        self.swap_placeholder.setAcceptDrops(True)
        self.swap_placeholder.dragEnterEvent = lambda event: self.dragEnterEvent_swap_gauche(event)
        self.swap_placeholder.dropEvent = lambda event: self.dropEvent_swap_gauche(event)

        self.inter_swap_placeholder = QLabel(self.swap_tab)
        self.inter_swap_placeholder.setAutoFillBackground(True)
        self.inter_swap_pal = self.inter_swap_placeholder.palette()
        self.inter_swap_pal.setColor(QPalette.ColorRole.Window, QColor(0, 255, 0))
        self.inter_swap_placeholder.setPalette(self.inter_swap_pal)

        self.inter_swap_placeholder.setAcceptDrops(True)
        self.inter_swap_placeholder.dragEnterEvent = lambda event: self.dragEnterEvent_swap_doite(event)
        self.inter_swap_placeholder.dropEvent = lambda event: self.dropEvent_swap_droite(event)

        self.res_swap_placeholder = QLabel(self.swap_tab)
        self.res_swap_placeholder.setAutoFillBackground(True)
        self.res_swap_pal = self.res_swap_placeholder.palette()
        self.res_swap_pal.setColor(QPalette.ColorRole.Window, QColor(125, 255, 125))
        self.res_swap_placeholder.setPalette(self.res_swap_pal)

        placeholders_layout = QHBoxLayout()
        placeholders_layout.addWidget(self.swap_placeholder)
        placeholders_layout.addWidget(self.inter_swap_placeholder)
        placeholders_layout.addWidget(self.res_swap_placeholder)

        buttons_layout = QVBoxLayout()
        buttons_layout.addWidget(self.swap_gauche_open_button)
        buttons_layout.addWidget(self.swap_droite_open_button)
        buttons_layout.addWidget(self.swap_save_button)
        buttons_layout.addWidget(self.swap_swap_button)
        buttons_layout.addWidget(self.swap_mp4_button)
        buttons_layout.addWidget(self.swap_gif_button)
        buttons_layout.addLayout(fps_layout)
        buttons_layout.addLayout(frames_layout)

        num_placeholders = 3
        button_width = int(self.swap_tab.width() * 3 / num_placeholders)
        self.swap_gauche_open_button.setFixedSize(button_width, self.swap_gauche_open_button.height())
        self.swap_droite_open_button.setFixedSize(button_width, self.swap_droite_open_button.height())
        self.swap_save_button.setFixedSize(button_width, self.swap_save_button.height())
        self.swap_swap_button.setFixedSize(button_width, self.swap_swap_button.height())
        self.swap_mp4_button.setFixedSize(button_width, self.swap_mp4_button.height())
        self.swap_gif_button.setFixedSize(button_width, self.swap_gif_button.height())
        self.swap_fps_spinbox.setFixedSize(button_width, self.swap_save_button.height())
        self.swap_frames_spinbox.setFixedSize(button_width, self.swap_save_button.height())

        self.swap_layout = QHBoxLayout(self.swap_tab)
        self.swap_layout.addLayout(placeholders_layout)
        self.swap_layout.addLayout(buttons_layout)

        ##########################################################################

        #Morph Onglet
        #-------------------------------------------------------------------------

        self.morph_gauche_open_button = QPushButton("Open Source", self.morph_tab)
        self.morph_droite_open_button = QPushButton("Open Cible", self.morph_tab)
        self.morph_save_button = QPushButton("Save", self.morph_tab)
        self.morph_morph_button = QPushButton("Median Morph", self.morph_tab)

        self.morph_slider = QSlider(Qt.Orientation.Horizontal)
        self.morph_slider.setRange(0, 100)
        self.morph_slider.setFixedWidth(int(self.morph_tab.width()))
        self.morph_slider.setFixedHeight(int(self.morph_tab.height()))
        self.morph_slider.valueChanged.connect(self.slider_value_changed)

        self.morph_label = QLabel("Morph Value: 0")
        layout = QVBoxLayout()
        layout.addWidget(self.morph_slider)
        layout.addWidget(self.morph_label)

        self.morph_placeholder = QLabel(self.morph_tab)
        self.morph_placeholder.setAutoFillBackground(True)
        self.morph_pal = self.morph_placeholder.palette()
        self.morph_pal.setColor(QPalette.ColorRole.Window, QColor(255, 0, 0))
        self.morph_placeholder.setPalette(self.morph_pal)

        self.morph_placeholder.setAcceptDrops(True)
        self.morph_placeholder.dragEnterEvent = lambda event: self.dragEnterEvent_morph_gauche(event)
        self.morph_placeholder.dropEvent = lambda event: self.dropEvent_morph_gauche(event)

        self.inter_morph_placeholder = QLabel(self.morph_tab)
        self.inter_morph_placeholder.setAutoFillBackground(True)
        self.inter_morph_pal = self.inter_morph_placeholder.palette()
        self.inter_morph_pal.setColor(QPalette.ColorRole.Window, QColor(255, 0, 0))
        self.inter_morph_placeholder.setPalette(self.inter_morph_pal)

        self.inter_morph_placeholder.setAcceptDrops(True)
        self.inter_morph_placeholder.dragEnterEvent = lambda event: self.dragEnterEvent_morph_doite(event)
        self.inter_morph_placeholder.dropEvent = lambda event: self.dropEvent_morph_droite(event)

        self.res_morph_placeholder = QLabel(self.morph_tab)
        self.res_morph_placeholder.setAutoFillBackground(True)
        self.res_morph_pal = self.res_morph_placeholder.palette()
        self.res_morph_pal.setColor(QPalette.ColorRole.Window, QColor(255, 125, 125))
        self.res_morph_placeholder.setPalette(self.res_morph_pal)

        placeholders_layout = QHBoxLayout()
        placeholders_layout.addWidget(self.morph_placeholder)
        placeholders_layout.addWidget(self.inter_morph_placeholder)
        placeholders_layout.addWidget(self.res_morph_placeholder)

        buttons_layout = QVBoxLayout()
        buttons_layout.addWidget(self.morph_gauche_open_button)
        buttons_layout.addWidget(self.morph_droite_open_button)
        buttons_layout.addWidget(self.morph_save_button)
        buttons_layout.addWidget(self.morph_morph_button)
        buttons_layout.addLayout(layout)

        num_placeholders = 3
        button_width = int(self.morph_tab.width() * 3 / num_placeholders)
        self.morph_gauche_open_button.setFixedSize(button_width, self.morph_gauche_open_button.height())
        self.morph_droite_open_button.setFixedSize(button_width, self.morph_droite_open_button.height())
        self.morph_save_button.setFixedSize(button_width, self.morph_save_button.height())
        self.morph_morph_button.setFixedSize(button_width, self.morph_morph_button.height())

        self.morph_layout = QHBoxLayout(self.morph_tab)
        self.morph_layout.addLayout(placeholders_layout)
        self.morph_layout.addLayout(buttons_layout)


        #--------------------------------------------------------------------------------

        #CNN onglet
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        self.cnn_open_button = QPushButton("Open", self.cnn_tab)
        self.cnn_save_button = QPushButton("Save", self.cnn_tab)
        self.cnn_cnn_button = QPushButton("CNN", self.cnn_tab)

        self.cnn_placeholder = QLabel(self.cnn_tab)
        self.cnn_placeholder.setAutoFillBackground(True)
        self.cnn_pal = self.cnn_placeholder.palette()
        self.cnn_pal.setColor(QPalette.ColorRole.Window, QColor(0, 0, 255))
        self.cnn_placeholder.setPalette(self.cnn_pal)

        self.cnn_placeholder.setAcceptDrops(True)
        self.cnn_placeholder.dragEnterEvent = lambda event: self.dragEnterEvent_cnn(event)
        self.cnn_placeholder.dropEvent = lambda event: self.dropEvent_cnn(event)

        self.res_cnn_placeholder = QLabel(self.cnn_tab)
        self.res_cnn_placeholder.setAutoFillBackground(True)
        self.res_cnn_pal = self.res_cnn_placeholder.palette()
        self.res_cnn_pal.setColor(QPalette.ColorRole.Window, QColor(125, 125, 255))
        self.res_cnn_placeholder.setPalette(self.res_cnn_pal)

        placeholders_layout = QHBoxLayout()
        placeholders_layout.addWidget(self.cnn_placeholder)
        placeholders_layout.addWidget(self.res_cnn_placeholder)

        buttons_layout = QVBoxLayout()
        buttons_layout.addWidget(self.cnn_open_button)
        buttons_layout.addWidget(self.cnn_save_button)
        buttons_layout.addWidget(self.cnn_cnn_button)

        num_placeholders = 2
        button_width = int(self.cnn_tab.width() * 2 / num_placeholders)
        self.cnn_open_button.setFixedSize(button_width, self.cnn_open_button.height())
        self.cnn_save_button.setFixedSize(button_width, self.cnn_save_button.height())
        self.cnn_cnn_button.setFixedSize(button_width, self.cnn_cnn_button.height())

        self.cnn_layout = QHBoxLayout(self.cnn_tab)
        self.cnn_layout.addLayout(placeholders_layout)
        self.cnn_layout.addLayout(buttons_layout)


        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        # Analyse onglet
        #{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{

        self.analyze_homme_button = QPushButton("Homme", self.analyze_tab)
        self.analyze_femme_button = QPushButton("Femme", self.analyze_tab)
        self.analyze_next_button = QPushButton("Next", self.analyze_tab)

        self.analyze_placeholder = QLabel(self.analyze_tab)
        self.analyze_placeholder.setAutoFillBackground(True)
        self.analyze_pal = self.analyze_placeholder.palette()
        self.analyze_pal.setColor(QPalette.ColorRole.Window, QColor(0, 0, 255))
        self.analyze_placeholder.setPalette(self.analyze_pal)

        placeholders_layout = QHBoxLayout()
        placeholders_layout.addWidget(self.analyze_placeholder)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.analyze_homme_button)
        buttons_layout.addWidget(self.analyze_next_button)
        buttons_layout.addWidget(self.analyze_femme_button)

        button_width = int(self.analyze_tab.width() * 3)
        self.analyze_homme_button.setFixedSize(button_width, self.analyze_homme_button.height())
        self.analyze_next_button.setFixedSize(button_width, self.analyze_next_button.height())
        self.analyze_femme_button.setFixedSize(button_width, self.analyze_femme_button.height())

        self.analyze_layout = QVBoxLayout(self.analyze_tab)
        self.analyze_layout.addLayout(placeholders_layout)
        self.analyze_layout.addLayout(buttons_layout)

        self.load_image_files("./")

        #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        self.placeholder_dict = {}
        self.placeholder_dict[self.swap_tab] = self.swap_placeholder
        self.placeholder_dict[self.morph_tab] = self.morph_placeholder
        self.placeholder_dict[self.cnn_tab] = self.cnn_placeholder
        self.res_placeholder_dict = {}
        self.res_placeholder_dict[self.swap_tab] = self.res_swap_placeholder
        self.res_placeholder_dict[self.morph_tab] = self.res_morph_placeholder
        self.res_placeholder_dict[self.cnn_tab] = self.res_cnn_placeholder

        #[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
        #Connecter boutton avec fonction

        self.swap_gauche_open_button.clicked.connect(lambda: self.open_image(option=3))
        self.swap_droite_open_button.clicked.connect(lambda: self.open_image(option=4))
        self.swap_save_button.clicked.connect(lambda: self.save_image(option=3))
        self.swap_swap_button.clicked.connect(self.Operation_swap)
        self.swap_mp4_button.clicked.connect(lambda: self.on_swap_anim_button_clicked(option=1))
        self.swap_gif_button.clicked.connect(lambda: self.on_swap_anim_button_clicked(option=2))


        #////////////////////////////////////////////////////////////////////////////////
        self.morph_gauche_open_button.clicked.connect(lambda: self.open_image(option=1))
        self.morph_droite_open_button.clicked.connect(lambda: self.open_image(option=2))
        self.morph_save_button.clicked.connect(lambda: self.save_image(option=1))
        self.morph_morph_button.clicked.connect(self.Operator_linear_morph)

        #////////////////////////////////////////////////////////////////////////////////
        self.cnn_open_button.clicked.connect(lambda: self.open_image(option=5))
        self.cnn_save_button.clicked.connect(lambda: self.save_image(option=2))
        self.cnn_cnn_button.clicked.connect(self.Operation_cnn)

        #////////////////////////////////////////////////////////////////////////////////
        self.analyze_next_button.clicked.connect(self.load_next_image)

        #////////////////////////////////////////////////////////////////////////////////

        #Help dans texte.txt
        #Parcourt du fichier et on écrit directement dans la fenetre
        self.tabs.currentChanged.connect(lambda i: self.load_help_text(self.help_tab) if i == 4 else "")

    def dragEnterEvent_swap_gauche(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent_swap_gauche(self, event):
        for url in event.mimeData().urls():
            file_name = url.toLocalFile()
            self.open_image(3,file_name)

    def dragEnterEvent_swap_doite(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent_swap_droite(self, event):
        for url in event.mimeData().urls():
            file_name = url.toLocalFile()
            self.open_image(4,file_name)

    def dragEnterEvent_morph_gauche(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent_morph_gauche(self, event):
        for url in event.mimeData().urls():
            file_name = url.toLocalFile()
            self.open_image(1,file_name)

    def dragEnterEvent_morph_doite(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent_morph_droite(self, event):
        for url in event.mimeData().urls():
            file_name = url.toLocalFile()
            self.open_image(2,file_name)

    def dragEnterEvent_morph(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent_morph(self, event):
        for url in event.mimeData().urls():
            file_name = url.toLocalFile()
            self.open_image(1,file_name)

    def dragEnterEvent_cnn(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent_cnn(self, event):
        for url in event.mimeData().urls():
            file_name = url.toLocalFile()
            self.open_image(2,file_name)

    def slider_value_changed(self):
        morph_value = self.morph_slider.value() / 100.0
        self.morph_label.setText(f"Morph Value: {morph_value}")
        _Morph.set_power(float(morph_value))
        self.Operator_linear_morph()

    def load_help_text(self, help_tab):
        try:
            with open("texte.txt", "r", encoding="utf-8") as file:
                content = file.read()
            help_tab.setPlainText(content)
        except FileNotFoundError:
            help_tab.setPlainText("File not found")
        except Exception as e:
            help_tab.setPlainText(f"An error occurred: {str(e)}")

    def cv2_to_qimage(self, cv_image):
        height, width, channel = cv_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
        return q_image

    def load_image_files(self, directory):
            self.image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in self.image_extensions):
                        self.image_files.append(os.path.join(root, file))

    def load_image(self):
        if 0 <= self.current_image_index < len(self.image_files):
            file_name = self.image_files[self.current_image_index]
            pixmap = QPixmap(file_name)
            if not pixmap.isNull():
                placeholder_size = self.analyze_placeholder.size()
                placeholder_rect = self.analyze_placeholder.rect()

                scaled_pixmap = pixmap.scaled(placeholder_size, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio, transformMode=Qt.TransformationMode.SmoothTransformation)
                scaled_rect = scaled_pixmap.rect()

                center_x = (placeholder_rect.width() - scaled_rect.width()) / 2
                center_y = (placeholder_rect.height() - scaled_rect.height()) / 2

                self.analyze_placeholder.setPixmap(scaled_pixmap)
                self.analyze_placeholder.setGeometry(int(center_x - 1), int(center_y - 1), int(scaled_rect.width() - 1), int(scaled_rect.height() - 1))


    def load_next_image(self):
        if 0 <= self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_image()
        else :
            self.current_image_index = 0
            self.load_image()

    def open_image(self, option=1, name=None):
        if name is not None: file_name = name
        else: file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)")
        if file_name:
            print("Open file " + file_name + "\n")
            image = cv2.imread(str(file_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image is not None:
                if option == 1:
                    placeholder = self.morph_placeholder
                    _Morph.set_source(str(file_name))
                    print("Set Source" + str(file_name))
                elif option == 2:
                    placeholder = self.inter_morph_placeholder
                    _Morph.set_target(str(file_name))
                    print("Set Target" + str(file_name))
                elif option == 3:
                    placeholder = self.swap_placeholder
                    _Swap.set_body(str(file_name))
                    print("Set Body" + str(file_name))
                elif option == 4:
                    placeholder = self.inter_swap_placeholder
                    _Swap.set_face(str(file_name))
                    print("Set Face" + str(file_name))
                elif option == 5:
                    placeholder = self.cnn_placeholder
                    print("Set Portrait" + str(file_name))
                if placeholder:
                    empty_pixmap = QPixmap(placeholder.size())
                    empty_pixmap.fill(Qt.GlobalColor.white)
                    placeholder.setPixmap(empty_pixmap)
                    size = placeholder.size()
                    image_height, image_width, _ = image.shape
                    ratio = min(size.height() / image_height, size.width() / image_width)
                    new_height = int(image_height * ratio)
                    new_width = int(image_width * ratio)
                    resized_image = cv2.resize(image, (new_width, new_height))
                    q_image = QImage(resized_image.data, new_width, new_height, new_width * 3, QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_image)
                    placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    placeholder.setPixmap(pixmap)
                else:
                    print("No placeholder found in the active tab")
            else:
                print("Failed to open the image")

    def save_image(self, option=1):
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Image File", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)")
        if file_name:
            active_tab = self.central_widget.currentWidget()
            if active_tab:
                if option == 1:
                    placeholder = self.res_morph_placeholder
                elif option == 2:
                    placeholder = self.res_cnn_placeholder
                elif option == 3:
                    placeholder = self.res_swap_placeholder
                if placeholder:
                    pixmap = placeholder.pixmap()
                    if not pixmap.isNull():
                        q_image = pixmap.toImage()
                        if q_image.save(file_name):
                            print("Image saved successfully.")
                        else:
                            print("Failed to save the image.")
                    else:
                        print("No image to save")
                else:
                    print("No placeholder found in the active tab")
            else:
                print("No active tab found")


    def on_swap_anim_button_clicked(self,option=1):
        options = 0

        file_dialog = QFileDialog()
        folder_path = file_dialog.getExistingDirectory(None, "Sélectionnez un dossier de destination")

        if folder_path:
            file_name, ok = QInputDialog.getText(self, "Nom du fichier", "Entrez le nom du fichier : ")

            if ok and file_name:
                output_filename = os.path.join(folder_path, file_name)
                print(output_filename)
                _Swap.set_animation(output_filename)
                _Swap.set_fps(int(self.swap_fps_spinbox.value()))
                _Swap.set_frames(int(self.swap_frames_spinbox.value()))
                _Swap.get_animation() if option==1 else _Swap.get_gif()

    def Operation_swap(self):
        _Swap.swap()
        res = _Swap.get_result()
        res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

        q_image = QImage(res.data, res.shape[1], res.shape[0], res.shape[1] * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        placeholder = self.res_swap_placeholder
        if placeholder:
            empty_pixmap = QPixmap(placeholder.size())
            empty_pixmap.fill(Qt.GlobalColor.white)
            placeholder.setPixmap(empty_pixmap)
            pixmap = pixmap.scaled(placeholder.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setPixmap(pixmap)
        else:
            print("No placeholder found for the swap operation")

    def Operator_linear_morph(self):
        _Morph.linear_morph2()
        self.Operation_morph()

    def Operator_landmark_morph(self):
        _Morph.landmark_morph()
        self.Operation_morph()

    def Operation_morph(self):
        res = _Morph.get_result()
        res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

        q_image = QImage(res.data, res.shape[1], res.shape[0], res.shape[1] * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        placeholder = self.res_morph_placeholder
        if placeholder:
            empty_pixmap = QPixmap(placeholder.size())
            empty_pixmap.fill(Qt.GlobalColor.white)
            placeholder.setPixmap(empty_pixmap)
            pixmap = pixmap.scaled(placeholder.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setPixmap(pixmap)
        else:
            print("No placeholder found for the morph operation")

    def Operation_cnn(self):
        output_directory = "results/f2m/"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        output_directory = "results/m2f/"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        train()
        cnn_test()

def main():
    app = QApplication(sys.argv)
    window = MainApplication()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
