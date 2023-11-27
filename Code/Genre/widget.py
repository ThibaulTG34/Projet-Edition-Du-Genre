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
from Analyse import *
from classificateur import *

_Swap = Swap()
_CNN = CNN()
_Analyse = Analyse()

class OptionsDialog(QDialog):
    def __init__(self, parent=None):
        super(OptionsDialog, self).__init__(parent)
        self.setWindowTitle("Plotting options")

        ecran_pc = QGuiApplication.primaryScreen().availableGeometry()
        self.x = ecran_pc.width() / 6
        self.y = ecran_pc.height() / 4
        self.z = ecran_pc.width() / 4
        self.t = ecran_pc.height() / 4

        self.setGeometry(int(self.x), int(self.y), int(self.z), int(self.t))
        self.setMinimumSize(int(self.x/2), int(self.y/2))
        self.setMaximumSize(int(self.z/2), int(self.t/2))

        self.checkbox_confusion = QCheckBox("Confusion Matrix", self)
        self.checkbox_score = QCheckBox("Score Plot", self)

        layout = QVBoxLayout(self)
        layout.addWidget(self.checkbox_confusion)
        layout.addWidget(self.checkbox_score)

        button_ok = QPushButton("OK", self)
        button_ok.clicked.connect(self.accept)
        layout.addWidget(button_ok)

class MainApplication(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Sirin magical Morph")

        ecran_pc = QGuiApplication.primaryScreen().availableGeometry()
        self.x = int(ecran_pc.width() / 7.5)
        self.y = int(ecran_pc.height() / 5.2)
        self.z = int(ecran_pc.width() * 1.6)
        self.t = int(ecran_pc.height() * 1.5)

        self.setGeometry(self.x, self.y, self.z, self.t)
        self.setMinimumSize(int(self.x/2), int(self.y/2))
        self.setMaximumSize(int(self.z/2), int(self.t/2))

        self.male_directory = str("./")
        self.female_directory = str("./")
        self.analyse_directory = str("./")
        self.json_directory = str("./")
        self.keras_directory = str("./")
        self.gan_output_directory = str("./")

        self.central_widget = QStackedWidget(self)
        self.setCentralWidget(self.central_widget)

        self.image_files = []
        self.current_image_index = -1

        self.last_gender = "Male"

        self.gan_parameters = [
            ('Epoch', 'epoch', 0),
            ('Number of Epochs', 'n_epochs', 10),
            ('Batch Size', 'batchSize', 1),
            ('Learning Rate', 'lr', 0.0002),
            ('Decay Epoch', 'decay_epoch', 5),
            ('Size', 'size', 256),
            ('Input Channels', 'input_nc', 3),
            ('Output Channels', 'output_nc', 3),
            ('Use GPU', 'cuda', False),
            ('Number of CPU Threads', 'n_cpu', 8),
        ]

        self.init_tabs()

    def init_tabs(self):
        self.tabs = QTabWidget(self)

        self.swap_tab = QWidget(self)
        self.morph_tab = QWidget(self)
        self.param_tab = QWidget(self)
        self.param_tab.setStyleSheet("background-color: lightblue;")
        self.analyze_tab = QWidget(self)
        self.help_tab = QTextBrowser(self)

        self.tabs.addTab(self.swap_tab, "Swap")
        self.tabs.addTab(self.morph_tab, "Morph")
        self.tabs.addTab(self.analyze_tab, "Analyze")
        self.tabs.addTab(self.param_tab, "Param")
        self.tabs.addTab(self.help_tab, "Help")

        self.central_widget.addWidget(self.tabs)

        #Swap onglet
        ##################################################

        self.swap_gauche_open_button = QPushButton("Open Source", self.swap_tab)
        self.swap_droite_open_button = QPushButton("Open Cible", self.swap_tab)
        self.swap_save_button = QPushButton("Save", self.swap_tab)
        self.swap_mtf_button = QPushButton("mtf", self.swap_tab)
        self.swap_ftm_button = QPushButton("ftm", self.swap_tab)
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
        buttons_layout.addWidget(self.swap_mtf_button)
        buttons_layout.addWidget(self.swap_ftm_button)
        buttons_layout.addWidget(self.swap_mp4_button)
        buttons_layout.addWidget(self.swap_gif_button)
        buttons_layout.addLayout(fps_layout)
        buttons_layout.addLayout(frames_layout)

        num_placeholders = 3
        button_width = int(self.swap_tab.width() * 3 / num_placeholders)
        self.swap_gauche_open_button.setFixedSize(button_width, self.swap_gauche_open_button.height())
        self.swap_droite_open_button.setFixedSize(button_width, self.swap_droite_open_button.height())
        self.swap_save_button.setFixedSize(button_width, self.swap_save_button.height())
        self.swap_mtf_button.setFixedSize(button_width, self.swap_mtf_button.height())
        self.swap_ftm_button.setFixedSize(button_width, self.swap_ftm_button.height())
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
        self.morph_save_button = QPushButton("Save", self.morph_tab)
        self.morph_mtf_button = QPushButton("Morph mtf", self.morph_tab)
        self.morph_ftm_button = QPushButton("Morph ftm", self.morph_tab)
        self.morph_mp4_button = QPushButton("MP4", self.morph_tab)
        self.morph_gif_button = QPushButton("GIF", self.morph_tab)

        fps_layout = QVBoxLayout()
        frames_layout = QVBoxLayout()
        fps_layout.setSpacing(0)
        frames_layout.setSpacing(0)

        morph_fps_label = QLabel("FPS : ")
        morph_frames_label = QLabel("Frames")

        self.morph_fps_spinbox = QSpinBox()
        self.morph_fps_spinbox.setRange(0, 120)
        self.morph_fps_spinbox.setValue(15)
        self.morph_fps_spinbox.setSingleStep(1)

        self.morph_frames_spinbox = QSpinBox()
        self.morph_frames_spinbox.setRange(0, 150)
        self.morph_frames_spinbox.setValue(100)
        self.morph_frames_spinbox.setSingleStep(1)

        fps_layout.addWidget(morph_fps_label)
        frames_layout.addWidget(morph_frames_label)
        fps_layout.addWidget(self.morph_fps_spinbox)
        frames_layout.addWidget(self.morph_frames_spinbox)

        fps_layout.setSpacing(0)
        frames_layout.setSpacing(0)

        self.morph_placeholder = QLabel(self.morph_tab)
        self.morph_placeholder.setAutoFillBackground(True)
        self.morph_pal = self.morph_placeholder.palette()
        self.morph_pal.setColor(QPalette.ColorRole.Window, QColor(255, 0, 0))
        self.morph_placeholder.setPalette(self.morph_pal)

        self.morph_placeholder.setAcceptDrops(True)
        self.morph_placeholder.dragEnterEvent = lambda event: self.dragEnterEvent_morph_gauche(event)
        self.morph_placeholder.dropEvent = lambda event: self.dropEvent_morph_gauche(event)

        self.res_morph_placeholder = QLabel(self.morph_tab)
        self.res_morph_placeholder.setAutoFillBackground(True)
        self.res_morph_pal = self.res_morph_placeholder.palette()
        self.res_morph_pal.setColor(QPalette.ColorRole.Window, QColor(255, 125, 125))
        self.res_morph_placeholder.setPalette(self.res_morph_pal)

        placeholders_layout = QHBoxLayout()
        placeholders_layout.addWidget(self.morph_placeholder)
        placeholders_layout.addWidget(self.res_morph_placeholder)

        buttons_layout = QVBoxLayout()
        buttons_layout.addWidget(self.morph_gauche_open_button)
        buttons_layout.addWidget(self.morph_save_button)
        buttons_layout.addWidget(self.morph_mtf_button)
        buttons_layout.addWidget(self.morph_ftm_button)
        buttons_layout.addWidget(self.morph_mp4_button)
        buttons_layout.addWidget(self.morph_gif_button)
        buttons_layout.addLayout(fps_layout)
        buttons_layout.addLayout(frames_layout)

        num_placeholders = 3
        button_width = int(self.morph_tab.width() * 3 / num_placeholders)
        self.morph_gauche_open_button.setFixedSize(button_width, self.morph_gauche_open_button.height())
        self.morph_save_button.setFixedSize(button_width, self.morph_save_button.height())
        self.morph_mtf_button.setFixedSize(button_width, self.morph_mtf_button.height())
        self.morph_ftm_button.setFixedSize(button_width, self.morph_ftm_button.height())
        self.morph_mp4_button.setFixedSize(button_width, self.morph_mp4_button.height())
        self.morph_gif_button.setFixedSize(button_width, self.morph_gif_button.height())
        self.morph_fps_spinbox.setFixedSize(button_width, self.morph_save_button.height())
        self.morph_frames_spinbox.setFixedSize(button_width, self.morph_save_button.height())

        self.morph_layout = QHBoxLayout(self.morph_tab)
        self.morph_layout.addLayout(placeholders_layout)
        self.morph_layout.addLayout(buttons_layout)

        #--------------------------------------------------------------------------------

        #param onglet
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        self.param_layout = QVBoxLayout(self.param_tab)
        self.param_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Floating square
        self.param_floating_square = QFrame(self.param_tab)
        a = int(int(self.x)/4)
        b = int(int(self.x)/2)
        self.param_floating_square.setGeometry(a,a,b,b)
        self.param_floating_square.setStyleSheet("background-color: gray;")
        self.param_floating_square_layout = QVBoxLayout(self.param_floating_square)

        # Buttons
        self.param_mdir = QPushButton("Set Female Img directory -> ")
        self.param_fdir = QPushButton("Set Male Img directory -> ")
        self.param_adir = QPushButton("Set Analyze directory -> ")
        self.param_json = QPushButton("Set JSON directory -> ")
        self.param_kdir = QPushButton("Set Keras directory -> ")
        self.param_outdir = QPushButton("Set Gan Output directory -> ")
        self.param_save = QPushButton("Save Gan settings")
        self.param_gan = QPushButton("Active Train")

        hbox_layout = QHBoxLayout()
        hbox_layout.addWidget(self.param_mdir)
        self.male_text_label = QLabel(self.male_directory)
        hbox_layout.addWidget(self.male_text_label)
        self.param_floating_square_layout.addLayout(hbox_layout)

        hbox_layout = QHBoxLayout()
        hbox_layout.addWidget(self.param_fdir)
        self.female_text_label = QLabel(self.female_directory)
        hbox_layout.addWidget(self.female_text_label)
        self.param_floating_square_layout.addLayout(hbox_layout)

        hbox_layout = QHBoxLayout()
        hbox_layout.addWidget(self.param_adir)
        self.analyze_text_label = QLabel(self.analyse_directory)
        hbox_layout.addWidget(self.analyze_text_label)
        self.param_floating_square_layout.addLayout(hbox_layout)

        hbox_layout = QHBoxLayout()
        hbox_layout.addWidget(self.param_json)
        self.json_text_label = QLabel(self.json_directory)
        hbox_layout.addWidget(self.json_text_label)
        self.param_floating_square_layout.addLayout(hbox_layout)

        hbox_layout = QHBoxLayout()
        hbox_layout.addWidget(self.param_kdir)
        self.keras_text_label = QLabel(self.keras_directory)
        hbox_layout.addWidget(self.keras_text_label)
        self.param_floating_square_layout.addLayout(hbox_layout)

        hbox_layout = QHBoxLayout()
        hbox_layout.addWidget(self.param_outdir)
        self.gan_text_label = QLabel(self.gan_output_directory)
        hbox_layout.addWidget(self.gan_text_label)
        self.param_floating_square_layout.addLayout(hbox_layout)

        # SpinBox
        self.param_gan_layout = QVBoxLayout()

        row_layout_epoch = QHBoxLayout()
        row_layout_epoch.addWidget(QLabel(self.gan_parameters[0][0]))
        self.param_epoch_spinbox = QSpinBox()
        self.param_epoch_spinbox.setRange(0, 100000000)
        self.param_epoch_spinbox.setValue(int(self.gan_parameters[0][2]))
        row_layout_epoch.addWidget(self.param_epoch_spinbox)
        self.param_gan_layout.addLayout(row_layout_epoch)

        row_layout_nepoch = QHBoxLayout()
        row_layout_nepoch.addWidget(QLabel(self.gan_parameters[1][0]))
        self.param_nepoch_spinbox = QSpinBox()
        self.param_nepoch_spinbox.setRange(0, 100000000)
        self.param_nepoch_spinbox.setValue(int(self.gan_parameters[1][2]))
        row_layout_nepoch.addWidget(self.param_nepoch_spinbox)
        self.param_gan_layout.addLayout(row_layout_nepoch)

        row_layout_batchsize = QHBoxLayout()
        row_layout_batchsize.addWidget(QLabel(self.gan_parameters[2][0]))
        self.param_batchsize_spinbox = QSpinBox()
        self.param_batchsize_spinbox.setRange(0, 100000000)
        self.param_batchsize_spinbox.setValue(int(self.gan_parameters[2][2]))
        row_layout_batchsize.addWidget(self.param_batchsize_spinbox)
        self.param_gan_layout.addLayout(row_layout_batchsize)

        row_layout_learning_rate = QHBoxLayout()
        row_layout_learning_rate.addWidget(QLabel(self.gan_parameters[3][0]))
        self.param_learning_rate_spinbox = QDoubleSpinBox()
        self.param_learning_rate_spinbox.setRange(0.0, 100000000.0)
        self.param_learning_rate_spinbox.setValue(float(self.gan_parameters[3][2]))
        row_layout_learning_rate.addWidget(self.param_learning_rate_spinbox)
        self.param_gan_layout.addLayout(row_layout_learning_rate)

        row_layout_depoch = QHBoxLayout()
        row_layout_depoch.addWidget(QLabel(self.gan_parameters[4][0]))
        self.param_depoch_spinbox = QSpinBox()
        self.param_depoch_spinbox.setRange(0, 100000000)
        self.param_depoch_spinbox.setValue(int(self.gan_parameters[4][2]))
        row_layout_depoch.addWidget(self.param_depoch_spinbox)
        self.param_gan_layout.addLayout(row_layout_depoch)

        row_layout_size = QHBoxLayout()
        row_layout_size.addWidget(QLabel(self.gan_parameters[5][0]))
        self.param_size_spinbox = QSpinBox()
        self.param_size_spinbox.setRange(0, 100000000)
        self.param_size_spinbox.setValue(int(self.gan_parameters[5][2]))
        row_layout_size.addWidget(self.param_size_spinbox)
        self.param_gan_layout.addLayout(row_layout_size)

        row_layout_inchannel = QHBoxLayout()
        row_layout_inchannel.addWidget(QLabel(self.gan_parameters[6][0]))
        self.param_inchannel_spinbox = QSpinBox()
        self.param_inchannel_spinbox.setRange(0, 4)
        self.param_inchannel_spinbox.setValue(int(self.gan_parameters[6][2]))
        row_layout_inchannel.addWidget(self.param_inchannel_spinbox)
        self.param_gan_layout.addLayout(row_layout_inchannel)

        row_layout_outchannel = QHBoxLayout()
        row_layout_outchannel.addWidget(QLabel(self.gan_parameters[7][0]))
        self.param_outchannel_spinbox = QSpinBox()
        self.param_outchannel_spinbox.setRange(0, 4)
        self.param_outchannel_spinbox.setValue(int(self.gan_parameters[7][2]))
        row_layout_outchannel.addWidget(self.param_outchannel_spinbox)
        self.param_gan_layout.addLayout(row_layout_outchannel)

        row_layout_gpu = QHBoxLayout()
        row_layout_gpu.addWidget(QLabel(self.gan_parameters[8][0]))
        self.param_gpu_checkbox = QCheckBox()
        self.param_gpu_checkbox.setChecked(bool(self.gan_parameters[8][2]))
        row_layout_gpu.addWidget(self.param_gpu_checkbox)
        self.param_gan_layout.addLayout(row_layout_gpu)

        row_layout_cpu = QHBoxLayout()
        row_layout_cpu.addWidget(QLabel(self.gan_parameters[9][0]))
        self.param_cpu_spinbox = QSpinBox()
        self.param_cpu_spinbox.setRange(0, 1000)
        self.param_cpu_spinbox.setValue(int(self.gan_parameters[9][2]))
        row_layout_cpu.addWidget(self.param_cpu_spinbox)
        self.param_gan_layout.addLayout(row_layout_cpu)

        self.param_floating_square_layout.addLayout(self.param_gan_layout)
        self.param_layout.addWidget(self.param_floating_square)
        self.param_layout.addWidget(self.param_save)
        self.param_layout.addWidget(self.param_gan)

        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        # Analyse onglet
        #{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{

        self.analyze_homme_button = QPushButton("Homme", self.analyze_tab)
        self.analyze_femme_button = QPushButton("Femme", self.analyze_tab)
        self.analyze_next_button = QPushButton("Next", self.analyze_tab)
        self.analyze_prec_button = QPushButton("Prec", self.analyze_tab)

        self.analyze_gnuplot_swap_rc_button = QPushButton("Swap_Real_Class.dat", self.analyze_tab)
        self.analyze_gnuplot_swap_uc_button = QPushButton("Swap_Real_User.dat", self.analyze_tab)
        self.analyze_gnuplot_swap_ur_button = QPushButton("Swap_User_Class.dat", self.analyze_tab)
        self.analyze_gnuplot_gan_rc_button = QPushButton("Gan_Real_Class.dat", self.analyze_tab)
        self.analyze_gnuplot_gan_uc_button = QPushButton("Gan_Real_User.dat", self.analyze_tab)
        self.analyze_gnuplot_gan_ur_button = QPushButton("Gan_User_Class.dat", self.analyze_tab)

        self.analyze_plot_swap_rc_button = QPushButton("Plot Swap_Real_Class", self.analyze_tab)
        self.analyze_plot_swap_uc_button = QPushButton("Plot Swap_Real_User", self.analyze_tab)
        self.analyze_plot_swap_ur_button = QPushButton("Plot Swap_User_Class", self.analyze_tab)
        self.analyze_plot_gan_rc_button = QPushButton("Plot Gan_Real_Class", self.analyze_tab)
        self.analyze_plot_gan_uc_button = QPushButton("Plot Gan_Real_User", self.analyze_tab)
        self.analyze_plot_gan_ur_button = QPushButton("Plot Gan_User_Class", self.analyze_tab)

        self.analyze_placeholder = QLabel(self.analyze_tab)
        self.analyze_placeholder.setAutoFillBackground(True)
        self.analyze_pal = self.analyze_placeholder.palette()
        self.analyze_pal.setColor(QPalette.ColorRole.Window, QColor(0, 0, 255))
        self.analyze_placeholder.setPalette(self.analyze_pal)

        placeholders_layout = QHBoxLayout()
        placeholders_layout.addWidget(self.analyze_placeholder)

        buttons_layout = QVBoxLayout()
        buttons_layout_haut = QHBoxLayout()
        buttons_layout_bas = QHBoxLayout()
        buttons_layout_footer = QHBoxLayout()

        buttons_layout_haut.addWidget(self.analyze_homme_button)
        buttons_layout_haut.addWidget(self.analyze_prec_button)
        buttons_layout_haut.addWidget(self.analyze_next_button)
        buttons_layout_haut.addWidget(self.analyze_femme_button)

        buttons_layout_bas.addWidget(self.analyze_gnuplot_swap_rc_button)
        buttons_layout_bas.addWidget(self.analyze_gnuplot_swap_ur_button)
        buttons_layout_bas.addWidget(self.analyze_gnuplot_swap_uc_button)
        buttons_layout_bas.addWidget(self.analyze_gnuplot_gan_rc_button)
        buttons_layout_bas.addWidget(self.analyze_gnuplot_gan_ur_button)
        buttons_layout_bas.addWidget(self.analyze_gnuplot_gan_uc_button)

        buttons_layout_footer.addWidget(self.analyze_plot_swap_rc_button)
        buttons_layout_footer.addWidget(self.analyze_plot_swap_ur_button)
        buttons_layout_footer.addWidget(self.analyze_plot_swap_uc_button)
        buttons_layout_footer.addWidget(self.analyze_plot_gan_rc_button)
        buttons_layout_footer.addWidget(self.analyze_plot_gan_ur_button)
        buttons_layout_footer.addWidget(self.analyze_plot_gan_uc_button)

        buttons_layout.addLayout(buttons_layout_haut)
        buttons_layout.addLayout(buttons_layout_bas)
        buttons_layout.addLayout(buttons_layout_footer)

        button_width = int(self.analyze_tab.width() * 3)
        self.analyze_homme_button.setFixedSize(button_width, self.analyze_homme_button.height())
        self.analyze_prec_button.setFixedSize(button_width, self.analyze_prec_button.height())
        self.analyze_next_button.setFixedSize(button_width, self.analyze_next_button.height())
        self.analyze_femme_button.setFixedSize(button_width, self.analyze_femme_button.height())

        self.analyze_gnuplot_swap_rc_button.setFixedSize(int(button_width/2), self.analyze_gnuplot_swap_rc_button.height())
        self.analyze_gnuplot_swap_ur_button.setFixedSize(int(button_width/2), self.analyze_gnuplot_swap_ur_button.height())
        self.analyze_gnuplot_swap_uc_button.setFixedSize(int(button_width/2), self.analyze_gnuplot_swap_uc_button.height())
        self.analyze_gnuplot_gan_rc_button.setFixedSize(int(button_width/2), self.analyze_gnuplot_gan_rc_button.height())
        self.analyze_gnuplot_gan_ur_button.setFixedSize(int(button_width/2), self.analyze_gnuplot_gan_ur_button.height())
        self.analyze_gnuplot_gan_uc_button.setFixedSize(int(button_width/2), self.analyze_gnuplot_gan_uc_button.height())


        self.analyze_plot_swap_ur_button.setFixedSize(int(button_width/2), self.analyze_plot_swap_ur_button.height())
        self.analyze_plot_swap_rc_button.setFixedSize(int(button_width/2), self.analyze_plot_swap_rc_button.height())
        self.analyze_plot_swap_uc_button.setFixedSize(int(button_width/2), self.analyze_plot_swap_uc_button.height())
        self.analyze_plot_gan_ur_button.setFixedSize(int(button_width/2), self.analyze_plot_gan_ur_button.height())
        self.analyze_plot_gan_rc_button.setFixedSize(int(button_width/2), self.analyze_plot_gan_rc_button.height())
        self.analyze_plot_gan_uc_button.setFixedSize(int(button_width/2), self.analyze_plot_gan_uc_button.height())

        self.analyze_layout = QVBoxLayout(self.analyze_tab)
        self.analyze_layout.addLayout(placeholders_layout)
        self.analyze_layout.addLayout(buttons_layout)

        #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        self.placeholder_dict = {}
        self.placeholder_dict[self.swap_tab] = self.swap_placeholder
        self.placeholder_dict[self.morph_tab] = self.morph_placeholder
        self.res_placeholder_dict = {}
        self.res_placeholder_dict[self.swap_tab] = self.res_swap_placeholder
        self.res_placeholder_dict[self.morph_tab] = self.res_morph_placeholder

        #[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
        #Connecter boutton avec fonction

        self.swap_gauche_open_button.clicked.connect(lambda: self.open_image(option=3))
        self.swap_droite_open_button.clicked.connect(lambda: self.open_image(option=4))
        self.swap_save_button.clicked.connect(lambda: self.save_image(option=3, option_model=1))
        self.swap_mtf_button.clicked.connect(lambda: self.Operation_swap(option=1))
        self.swap_ftm_button.clicked.connect(lambda: self.Operation_swap(option=2))
        self.swap_mp4_button.clicked.connect(lambda: self.on_swap_anim_button_clicked(option=1))
        self.swap_gif_button.clicked.connect(lambda: self.on_swap_anim_button_clicked(option=2))


        #////////////////////////////////////////////////////////////////////////////////
        self.morph_gauche_open_button.clicked.connect(lambda: self.open_image(option=1))
        self.morph_save_button.clicked.connect(lambda: self.save_image(option=1, option_model=2))
        self.morph_mtf_button.clicked.connect(lambda: self.Operator_morph(option=1))
        self.morph_ftm_button.clicked.connect(lambda: self.Operator_morph(option=2))
        self.morph_mp4_button.clicked.connect(lambda: self.on_morph_anim_button_clicked(option=1))
        self.morph_gif_button.clicked.connect(lambda: self.on_morph_anim_button_clicked(option=2))

        #////////////////////////////////////////////////////////////////////////////////

        # Buttons
        self.param_mdir.clicked.connect(self.set_male_dir)
        self.param_fdir.clicked.connect(self.set_female_dir)
        self.param_adir.clicked.connect(self.set_analyze_dir)
        self.param_json.clicked.connect(self.set_json_dir)
        self.param_kdir.clicked.connect(self.set_keras_dir)
        self.param_outdir.clicked.connect(self.set_gan_dir)
        self.param_save.clicked.connect(self.set_gan_data)
        self.param_gan.clicked.connect(_CNN.train)

        #////////////////////////////////////////////////////////////////////////////////
        self.analyze_next_button.clicked.connect(self.load_next_image)
        self.analyze_prec_button.clicked.connect(self.load_previous_image)
        self.analyze_homme_button.clicked.connect(lambda : self.analyze_upload(option=1))
        self.analyze_femme_button.clicked.connect(lambda : self.analyze_upload(option=2))

        self.analyze_gnuplot_swap_rc_button.clicked.connect(lambda : self.analyse_gnuplot(option=1, model=str("Swap")))
        self.analyze_gnuplot_swap_uc_button.clicked.connect(lambda : self.analyse_gnuplot(option=2, model=str("Swap")))
        self.analyze_gnuplot_swap_ur_button.clicked.connect(lambda : self.analyse_gnuplot(option=0, model=str("Swap")))
        self.analyze_gnuplot_gan_rc_button.clicked.connect(lambda : self.analyse_gnuplot(option=1, model=str("GAN")))
        self.analyze_gnuplot_gan_uc_button.clicked.connect(lambda : self.analyse_gnuplot(option=2, model=str("GAN")))
        self.analyze_gnuplot_gan_ur_button.clicked.connect(lambda : self.analyse_gnuplot(option=0, model=str("GAN")))

        self.analyze_plot_swap_rc_button.clicked.connect(lambda : self.analyse_plot(fight_option=1, model_option=1, option=2))
        self.analyze_plot_swap_uc_button.clicked.connect(lambda : self.analyse_plot(fight_option=3, model_option=1, option=2))
        self.analyze_plot_swap_ur_button.clicked.connect(lambda : self.analyse_plot(fight_option=2, model_option=1, option=2))
        self.analyze_plot_gan_rc_button.clicked.connect(lambda : self.analyse_plot(fight_option=1, model_option=2, option=2))
        self.analyze_plot_gan_uc_button.clicked.connect(lambda : self.analyse_plot(fight_option=3, model_option=2, option=2))
        self.analyze_plot_gan_ur_button.clicked.connect(lambda : self.analyse_plot(fight_option=2, model_option=2, option=2))

        #////////////////////////////////////////////////////////////////////////////////

        #Help dans texte.txt
        #Parcourt du fichier et on écrit directement dans la fenetre
        self.tabs.currentChanged.connect(lambda i: self.load_help_text(self.help_tab) if i == 4 else "")

###################################################################################################################
###################################################################################################################
###################################################################################################################
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

    def dragEnterEvent_morph(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent_morph(self, event):
        for url in event.mimeData().urls():
            file_name = url.toLocalFile()
            self.open_image(1,file_name)

    def dragEnterEvent_param(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent_param(self, event):
        for url in event.mimeData().urls():
            file_name = url.toLocalFile()
            self.open_image(2,file_name)

    def slider_value_changed(self):
        morph_value = self.morph_slider.value() / 100.0
        self.morph_label.setText(f"Morph Value: {morph_value}")
        _Morph.set_power(float(morph_value))
        self.Operator_linear_morph()

    def set_male_dir(self):
        file_dialog = QFileDialog()
        s = file_dialog.getExistingDirectory(None, "Sélectionnez un dossier de destination")
        self.male_directory = str(s)
        self.male_text_label.setText(str(s))

    def set_female_dir(self):
        file_dialog = QFileDialog()
        s = file_dialog.getExistingDirectory(None, "Sélectionnez un dossier de destination")
        self.female_directory = str(s)
        self.female_text_label.setText(str(s))

    def set_analyze_dir(self):
        file_dialog = QFileDialog()
        s = file_dialog.getExistingDirectory(None, "Sélectionnez un dossier de destination")
        self.analyse_directory = str(s)
        self.analyze_text_label.setText(str(s))
        self.load_image_files(str(s))

    def set_json_dir(self):
        s, _ = QFileDialog.getSaveFileName(self, "Save JSON File", "", "JSON Files (*.json);;All Files (*)")
        if s:
            s = os.path.normpath(s)
            if not s.lower().endswith(('.json',)):
                s += ".json"
        self.json_directory = str(s)
        self.json_text_label.setText(str(s))
        _Analyse.set_path(s)

    def set_keras_dir(self):
        file_dialog = QFileDialog()
        s = file_dialog.getExistingDirectory(None, "Sélectionnez un dossier de destination")
        self.keras_directory = str(s)
        _CNN.set_keras_dir(str(s))
        self.keras_text_label.setText(str(s))

    def set_gan_dir(self):
        file_dialog = QFileDialog()
        s = file_dialog.getExistingDirectory(None, "Sélectionnez un dossier de destination")
        self.gan_output_directory = str(s)
        _CNN.set_tensor_dir(str(s))
        self.gan_text_label.setText(str(s))

    def get_gpu(self):
        return bool(self.param_gpu_checkbox.isChecked())

    def get_epoch(self):
        return int(self.param_epoch_spinbox.value())

    def get_nepoch(self):
        return int(self.param_nepoch_spinbox.value())

    def get_depoch(self):
        return int(self.param_depoch_spinbox.value())

    def get_learning_rate(self):
        return float(int(self.param_learning_rate_spinbox.value())/100)

    def get_batch_size(self):
        return int(self.param_batchsize_spinbox.value())

    def get_size(self):
        return int(self.param_size_spinbox.value())

    def get_in_channel(self):
        return int(self.param_inchannel_spinbox.value())

    def get_out_channel(self):
        return int(self.param_outchannel_spinbox.value())

    def get_cpu(self):
        return int(self.param_cpu_spinbox.value())

    def set_gan_data(self):
        _CNN.set(0,self.get_epoch())
        _CNN.set(1,self.get_nepoch())
        _CNN.set(2,self.get_depoch())
        _CNN.set(3,self.get_learning_rate())
        _CNN.set(4,self.get_size())
        _CNN.set(5,self.get_batch_size())
        _CNN.set(6,self.get_in_channel())
        _CNN.set(7,self.get_out_channel())
        _CNN.set(8,self.get_cpu())
        _CNN.set(9,self.get_gpu())
        #_CNN.print_parameters()
        _CNN.to_dict()

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

    def load_previous_image(self):
        if self.current_image_index >= 0:
            self.current_image_index -= 1
            self.load_image()
        else:
            self.current_image_index = 0
            self.load_image()

    def analyze_upload(self, option=1):
        file_name = self.image_files[self.current_image_index]
        real_form = ("Male" if option==1 else "Female")
        _Analyse.update_or_create_entry(str(file_name), None, str(real_form), None, None, 1)

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
                    _CNN.set_source(str(file_name))
                    print("Set Source" + str(file_name))
                elif option == 3:
                    placeholder = self.swap_placeholder
                    _Swap.set_body(str(file_name))
                    print("Set Body" + str(file_name))
                elif option == 4:
                    placeholder = self.inter_swap_placeholder
                    _Swap.set_face(str(file_name))
                    print("Set Face" + str(file_name))
                elif option == 5:
                    placeholder = self.param_placeholder
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

    def save_image(self, option = 1, option_model = 1):
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Image File", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)")
        if file_name:
            active_tab = self.central_widget.currentWidget()
            if active_tab:
                if option == 1:
                    placeholder = self.res_morph_placeholder
                elif option == 2:
                    placeholder = self.res_param_placeholder
                elif option == 3:
                    placeholder = self.res_swap_placeholder
                if placeholder:
                    pixmap = placeholder.pixmap()
                    if not pixmap.isNull():
                        q_image = pixmap.toImage()
                        if q_image.save(file_name):
                            print("Image saved successfully.")
                            _Analyse.update_or_create_entry(str(file_name), str(self.last_gender), None, ("Swap" if option_model == 1 else "GAN"), (None if option_model == 1 else _CNN.to_dict()),1)
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

    def on_morph_anim_button_clicked(self,option=1):
        options = 0

        file_dialog = QFileDialog()
        folder_path = file_dialog.getExistingDirectory(None, "Sélectionnez un dossier de destination")

        if folder_path:
            file_name, ok = QInputDialog.getText(self, "Nom du fichier", "Entrez le nom du fichier : ")

            if ok and file_name:
                output_filename = os.path.join(folder_path, file_name)
                print(output_filename)
                _CNN.set_animation(output_filename)
                _CNN.set_fps(int(self.morph_fps_spinbox.value()))
                _CNN.set_frames(int(self.morph_frames_spinbox.value()))
                _CNN.get_animation() if option==1 else _CNN.get_gif()

    def analyse_gnuplot(self, option = 1, model = "Swap"):
        file_name_in, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Files (*.json);;All Files (*)")
        file_name_out, _ = QFileDialog.getSaveFileName(self, "Save File", "", "Files (*.dat);;All Files (*)")
        if file_name_in and file_name_out:
            _Analyse.metrique(str(file_name_in), option, True, str(model), str(file_name_out))

    def analyse_plot(self, fight_option = 1, model_option = 1, option = 1):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Files (*.dat);;All Files (*)")
        if file_name:
            options_dialog = OptionsDialog(self)
            result = options_dialog.exec()
            if options_dialog.checkbox_confusion.isChecked():
                _Analyse.plot_results(file_name, fight_option, model_option, 1)
            if options_dialog.checkbox_score.isChecked():
                _Analyse.plot_results(file_name, fight_option, model_option, 2)

    def Operation_swap(self, option=1):
        _Swap.set_directory(self.male_directory) if option == 1 else _Swap.set_directory(self.female_directory)
        self.last_gender = ("Female" if option==1 else "Male")
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

    def Operator_landmark_morph(self):
        _Morph.landmark_morph()
        self.Operation_morph()

    def Operator_morph(self, option=1):
        _CNN.set_mode(1) if option == 1 else _CNN.set_mode(2)
        self.last_gender = ("Female" if option == 1 else "Male" )
        _CNN.morphing(option)
        res = _CNN.get_result()
        if res is None : return
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

    def Operation_param(self):
        output_directory = "results/f2m/"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        output_directory = "results/m2f/"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        _CNN.train()
        param_test()

def main():
    app = QApplication(sys.argv)
    window = MainApplication()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
