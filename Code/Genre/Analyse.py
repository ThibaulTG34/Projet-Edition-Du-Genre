# This Python file uses the following encoding: utf-8
import json
import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from google.auth import exceptions
from google.oauth2.service_account import Credentials
from classificateur import *

class Analyse:
    def __init__(self):
        super().__init__()
        self.json_file = str("./analyze.json")
        self.directory = str("./")
        self.path_file = str("./analyze.json")

    def set_file_name(self, s):
        self.json_file = str(s)

    def set_directory(self, s):
        self.directory = str(s)

    def set_file(self):
        s, _ = QFileDialog.getOpenFileName(self, "Open JSON File", "", "Image Files (*.json);;All Files (*)")
        self.json_file = str(s)

    def set_directory(self):
        file_dialog = QFileDialog()
        s = file_dialog.getExistingDirectory(None, "Sélectionnez un dossier de destination")
        self.directory = str(s)

    def set_path(self, s):
        self.path_file = str(s)

    def union_path(self):
        self.path_file = os.path.join(self.directory, self.json_file)

    def calculate_scores(self, image_path):
        img = cv.imread(image_path)
        if img is None:
            print(f"Impossible de lire l'image à partir du chemin : {image_path}")
            return

        grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blurScore = cv.Laplacian(grey, cv.CV_64F).var()
        score = cv.quality.QualityBRISQUE_compute(img, "./classifieurs/brisque_model_live.yml", "./classifieurs/brisque_range_live.yml")

        #print(f' >> Blur Score: {blurScore}')
        #print(f' >> BRISQUE Score: {score}')

        return blurScore, score

    def update_or_create_entry(self, s, s2, s3 = str("Unknow"), s4 = str("Swap"), gan_parameters = None, option = 1):

        try:
            with open(self.path_file, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            data = {}

        _class = classified(s)

        if not option==0 :
            s = os.path.basename(s)

        #print("Auto classified as " + str(_class))
        if s in data:
            if s2 is not None:
                data[s]["real"] = s2

            data[s]["classificateur"] = _class

            if s3 is not None:
                data[s]["user"] = s3

            if s4 is not None:
                data[s]["model"] = s4

            if gan_parameters is not None:
                data[s]["gan_epochs"] = str(gan_parameters["nepochs"])
                data[s]["gan_decay"] = str(gan_parameters["decay_epochs"])
                data[s]["gan_learning_rate"] = str(gan_parameters["learning_decay"])
                data[s]["gan_batch_size"] = str(gan_parameters["batch_size"])
                data[s]["gan_size"] = str(gan_parameters["size"])


        else:
            if s2 is None: s2 = str("Unknow")
            if s3 is None: s3 = str("Unknow")
            if s4 is None: s4 = str("Unknow")
            if gan_parameters is None:
                data[s] = {
                            "real": s2,
                            "classificateur": _class,
                            "user": s3,
                            "model": s4,
                            "gan_epochs": str("Unknow"),
                            "gan_decay": str("Unknow"),
                            "gan_learning_rate": str("Unknow"),
                            "gan_batch_size": str("Unknow"),
                            "gan_size": str("Unknow")
                        }
            else:
                data[s] = {
                            "real": s2,
                            "classificateur": _class,
                            "user": s3,
                            "model": s4,
                            "gan_epochs": gan_parameters["nepochs"],
                            "gan_decay": gan_parameters["decay_epochs"],
                            "gan_learning_rate": gan_parameters["learning_decay"],
                            "gan_batch_size": gan_parameters["batch_size"],
                            "gan_size": gan_parameters["size"]
                        }


        with open(self.path_file, 'w') as file:
            json.dump(data, file, indent=2)

    #Vrais Positifs (VP): Nombre d'images correctement classées comme "Male".
    #Vrais Négatifs (VN): Nombre d'images correctement classées comme "Female".
    #Faux Positifs (FP): Nombre d'images incorrectement classées comme "Male" alors qu'elles sont réellement "Female".
    #Faux Négatifs (FN): Nombre d'images incorrectement classées comme "Female" alors qu'elles sont réellement "Male".

    def metrique(self, path, option = 1, gnuplot = False, m_model = str("Swap"), output_file_path = str("metrics_data.dat")):

        with open(path, 'r') as f:
            data = json.load(f)

        vp_list, vn_list, fp_list, fn_list = [], [], [], []

        for filename, info in data.items():
            real = info['real']
            classificateur = info['classificateur']
            user = info['user']
            model = info['model']
            gan_epochs = info["gan_epochs"]
            gan_decay = info["gan_decay"]
            gan_learning = info["gan_learning_rate"]
            gan_batch = info["gan_batch_size"]
            gan_size = info["gan_size"]

            player = str("user")
            adversaire = str("machine")
            if option == 1 :
                player = str("real")
                adversaire = str("machine")
            elif option == 2:
                player = str("real")
                adversaire = str("user")

            if str(model) == str(m_model):

                if option == 1 :
                    if real == 'Male' and classificateur == 'Male':
                        vp_list.append(filename)
                    elif real == 'Female' and classificateur == 'Female':
                        vn_list.append(filename)
                    elif real == 'Female' and classificateur == 'Male':
                        fp_list.append(filename)
                    elif real == 'Male' and classificateur == 'Female':
                        fn_list.append(filename)

                elif option == 2 :
                    if real == 'Male' and user == 'Male':
                        vp_list.append(filename)
                    elif real == 'Female' and user == 'Female':
                        vn_list.append(filename)
                    elif real == 'Female' and user == 'Male':
                        fp_list.append(filename)
                    elif real == 'Male' and user == 'Female':
                        fn_list.append(filename)

                else:
                    if user == 'Male' and classificateur == 'Male':
                        vp_list.append(filename)
                    elif user == 'Female' and classificateur == 'Female':
                        vn_list.append(filename)
                    elif user == 'Female' and classificateur == 'Male':
                        fp_list.append(filename)
                    elif user == 'Male' and classificateur == 'Female':
                        fn_list.append(filename)

        m_precision = len(vp_list) / ((len(vp_list) + len(fp_list)) if (len(vp_list) + len(fp_list)) != 0 else 1)
        m_recall = len(vp_list) / ((len(vp_list) + len(fn_list)) if (len(vp_list) + len(fn_list)) != 0 else 1)
        m_f1 = 2 * (m_precision * m_recall) / ((m_precision + m_recall) if (m_precision + m_recall) != 0 else 1)

        f_precision = len(vn_list) / ((len(vn_list) + len(fn_list)) if (len(vn_list) + len(fn_list)) != 0 else 1)
        f_recall = len(vn_list) / ((len(vn_list) + len(fp_list)) if (len(vn_list) + len(fp_list)) != 0 else 1)
        f_f1 = 2 * (f_precision * f_recall) / ((f_precision + f_recall) if (f_precision + f_recall) != 0 else 1)

        accuracy = (len(vp_list) + len(vn_list)) / ((len(vp_list) + len(vn_list) + len(fp_list) + len(fn_list)) if (len(vp_list) + len(vn_list) + len(fp_list) + len(fn_list)) != 0 else 1)

        #print(f'VP: {len(vp_list)}')
        #print(f'VN: {len(vn_list)}')
        #print(f'FP: {len(fp_list)}')
        #print(f'FN: {len(fn_list)}')
        #print(f'Male Précision: {m_precision}')
        #print(f'Male Rappel: {m_recall}')
        #print(f'Male F1 Score: {m_f1}')
        #print(f'Female Précision: {f_precision}')
        #print(f'Female Rappel: {f_recall}')
        #print(f'Female F1 Score: {f_f1}')
        #print(f'Exactitude: {accuracy}')

        gan_epochs = info["gan_epochs"]
        gan_decay = info["gan_decay"]
        gan_learning = info["gan_learning_rate"]
        gan_batch = info["gan_batch_size"]
        gan_size = info["gan_size"]

        #Vrais Positifs (VP): Nombre d'images correctement classées comme "Male".
        #Vrais Négatifs (VN): Nombre d'images correctement classées comme "Female".
        #Faux Positifs (FP): Nombre d'images incorrectement classées comme "Male" alors qu'elles sont réellement "Female".
        #Faux Négatifs (FN): Nombre d'images incorrectement classées comme "Female" alors qu'elles sont réellement "Male".

        if gnuplot is True:
            with open(output_file_path, 'a') as file:
                file.write(f'model | {m_model} | player | {player} | adversaire | {adversaire} | m_true | {len(vp_list)} | f_true | {len(vn_list)} | m_false | {len(fp_list)} | f_false | {len(fn_list)} | male_precision | {m_precision} | male_recall | {m_recall} | male_f1 | {m_f1} | female_precision | {f_precision} | female_recall | {f_recall} | female_f1 | {f_f1} | accuracy | {accuracy} | gan_decay | {gan_decay} | gan_learning | {gan_learning} | gan_batch | {gan_batch} | gan_size | {gan_size} | gan_epochs | {gan_epochs}\n')
            print(f'Données ajoutées dans {output_file_path}')

    def plot_results(self, file_path, fight_option = 1, model_option = 1, option = 1, proj = False):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        vp_list = []
        vn_list = []
        fp_list = []
        fn_list = []
        m_precision_list = []
        m_recall_list = []
        m_f1_list = []
        f_precision_list = []
        f_recall_list = []
        f_f1_list = []
        accuracy_list = []

        model = str("Swap" if model_option == 1 else "GAN")
        player = str("real" if fight_option == 1 else ( "real" if fight_option == 2 else "user"))
        adversaire = str("machine" if fight_option == 1 else ( "user" if fight_option == 2 else "machine"))

        print("Model[" + model + "] for : " + player + " vs " + adversaire)
        # 1 model
        # 3 player
        # 5 adversaire
        # 7 m_true
        # 9 f_true
        # 11 m_false
        # 13 f_false
        # 15 male_precision
        # 17 male_recall
        # 19 male_f1
        # 21 female_precision
        # 23 female_recall
        # 25 female_f1
        # 27 accuracy
        # 29 gan_decay
        # 31 gan_learning
        # 33 gan_batch
        # 35 gan_size

        for line in lines:
            if not line.strip():
                continue
            columns = line.strip().split(' | ')
            if not (len(columns) > 1):
                continue

            if columns[1] == model and columns[3] == player and columns[5] == adversaire:
                vp = int(columns[7])
                vn = int(columns[9])
                fp = int(columns[11])
                fn = int(columns[13])
                m_precision = float(columns[15])
                m_recall = float(columns[17])
                m_f1 = float(columns[19])
                f_precision = float(columns[21])
                f_recall = float(columns[23])
                f_f1 = float(columns[25])
                accuracy = float(columns[27])

                vp_list.append(vp)
                vn_list.append(vn)
                fp_list.append(fp)
                fn_list.append(fn)
                m_precision_list.append(m_precision)
                m_recall_list.append(m_recall)
                m_f1_list.append(m_f1)
                f_precision_list.append(f_precision)
                f_recall_list.append(f_recall)
                f_f1_list.append(f_f1)
                accuracy_list.append(accuracy)

        #if len(vp_list)<=0 :
        #    vp_list.append()
        total = sum(vp_list) + sum(vn_list) + sum(fp_list) + sum(fn_list)

        vp_list = [vp / total for vp in vp_list]
        vn_list = [vp / total for vp in vn_list]
        fp_list = [vp / total for vp in fp_list]
        fn_list = [vp / total for vp in fn_list]

        if option == 1:
            if (len(vp_list) <= 0) or (len(vn_list) <= 0) or (len(fp_list) <= 0) or (len(fn_list) <= 0) : return
            if proj is True:
                labels = ['M_true', 'F_true', 'M_false', 'F_false']
                sizes = [sum(vp_list), sum(vn_list), sum(fp_list), sum(fn_list)]
                fig = go.Figure()
                fig.add_trace(go.Pie(labels=labels, values=sizes, hole=0.4))
                fig.update_layout(title='Confusion Matrix (3D Projection)', scene=dict(aspectmode='data'))
                fig.show()
            else:
                labels = ['M_true', 'F_true', 'M_false', 'F_false']
                sizes = [sum(vp_list), sum(vn_list), sum(fp_list), sum(fn_list)]
                max_index = sizes.index(max(sizes))
                explode = [0] * 4
                explode[max_index] = 0.1
                plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, explode=explode)
                plt.title('Confusion Matrix')
                plt.show()
        elif option == 2:
            if (len(m_precision_list) <= 0) or (len(m_recall_list) <= 0) or (len(m_f1_list) <= 0) or (len(f_precision_list) <= 0) or (len(f_recall_list) <= 0) or (len(f_f1_list) <= 0) or (len(accuracy_list) <= 0): return
            if proj is True :
                indices = list(range(len(m_precision_list)))
                metric_labels = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
                m_data = [m_precision_list, m_recall_list, m_f1_list, accuracy_list]
                f_data = [f_precision_list, f_recall_list, f_f1_list, accuracy_list]
                fig = go.Figure()

                # Male
                for i, metric_label in enumerate(metric_labels):
                    fig.add_trace(go.Scatter3d(
                        x=indices,
                        y=[metric_label] * len(indices),
                        z=m_data[i],
                        mode='lines+markers',
                        name=f'Male_{metric_label}',
                        line=dict(color='blue')
                    ))
                # Female
                for i, metric_label in enumerate(metric_labels):
                    fig.add_trace(go.Scatter3d(
                        x=indices,
                        y=[metric_label] * len(indices),
                        z=f_data[i],
                        mode='lines+markers',
                        name=f'Female_{metric_label}',
                        line=dict(color='green')
                    ))
                # Accuracy
                fig.add_trace(go.Scatter3d(
                    x=indices,
                    y=['Accuracy'] * len(indices),
                    z=accuracy_list,
                    mode='lines+markers',
                    name='Accuracy',
                    line=dict(color='purple', width=4),
                    marker=dict(size=8, symbol='x', color='purple')
                    ))

                fig.update_layout(scene=dict(aspectmode="cube", xaxis=dict(title_text='Sample'), yaxis=dict(title_text='Metric'), zaxis=dict(title_text='Score')),
                                  title='Precision, Recall, F1 Score, Accuracy (3D)')
                fig.show()

            else :
                indices = list(range(len(m_precision_list)))
                plt.figure(figsize=(10, 6))

                plt.plot(indices, m_precision_list, label='Male_Precision', color='blue')
                plt.plot(indices, m_recall_list, label='Male_Recall', color='green')
                plt.plot(indices, m_f1_list, label='Male_F1 Score', color='orange')

                plt.plot(indices, f_precision_list, label='Female_Precision', linestyle='dashed', color='blue')
                plt.plot(indices, f_recall_list, label='Female_Recall', linestyle='dashed', color='green')
                plt.plot(indices, f_f1_list, label='Female_F1 Score', linestyle='dashed', color='orange')

                plt.plot(indices, accuracy_list, label='Accuracy', color='purple', linewidth=2, linestyle='-', marker='o', markersize=8)

                plt.legend()
                plt.title('Precision, Recall, F1 Score, Accuracy')
                plt.xlabel('Sample')
                plt.ylabel('Score')

                plt.show()
        else:
            return

_Analyse = Analyse()
macro_oui_3d = True
macro_non_3d = False
macro_swap = 1
macro_gan = 2
macro_confusion = 1
macro_stats = 2
macro_real_machine = 1
macro_real_user = 2
macro_user_machine = 3

#_Analyse.plot_results(str("./metrics_data_1.dat"), macro_real_user, macro_swap, macro_confusion, macro_non_3d)
#_Analyse.plot_results(str("./metrics_data_1.dat"), macro_real_user, macro_swap, macro_confusion, macro_oui_3d)
#_Analyse.plot_results(str("./metrics_data_1.dat"), macro_real_user, macro_swap, macro_stats, macro_non_3d)
#_Analyse.plot_results(str("./metrics_data_1.dat"), macro_real_user, macro_swap, macro_stats, macro_oui_3d)

#_Analyse.plot_results(str("./metrics_data_1.dat"), macro_real_machine, macro_swap, macro_confusion, macro_non_3d)
#_Analyse.plot_results(str("./metrics_data_1.dat"), macro_real_machine, macro_swap, macro_confusion, macro_oui_3d)
#_Analyse.plot_results(str("./metrics_data_1.dat"), macro_real_machine, macro_swap, macro_stats, macro_non_3d)
#_Analyse.plot_results(str("./metrics_data_1.dat"), macro_real_machine, macro_swap, macro_stats, macro_oui_3d)

#_Analyse.plot_results(str("./metrics_data_1.dat"), macro_real_user, macro_gan, macro_confusion, macro_non_3d)
#_Analyse.plot_results(str("./metrics_data_1.dat"), macro_real_user, macro_gan, macro_confusion, macro_oui_3d)
#_Analyse.plot_results(str("./metrics_data_1.dat"), macro_real_user, macro_gan, macro_stats, macro_non_3d)
#_Analyse.plot_results(str("./metrics_data_1.dat"), macro_real_user, macro_gan, macro_stats, macro_oui_3d)

#_Analyse.plot_results(str("./metrics_data_1.dat"), macro_real_machine, macro_gan, macro_confusion, macro_non_3d)
#_Analyse.plot_results(str("./metrics_data_1.dat"), macro_real_machine, macro_gan, macro_confusion, macro_oui_3d)
#_Analyse.plot_results(str("./metrics_data_1.dat"), macro_real_machine, macro_gan, macro_stats, macro_non_3d)
#_Analyse.plot_results(str("./metrics_data_1.dat"), macro_real_machine, macro_gan, macro_stats, macro_oui_3d)
