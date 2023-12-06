# This Python file uses the following encoding: utf-8
import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
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

        precision = len(vp_list) / ((len(vp_list) + len(fp_list)) if (len(vp_list) + len(fp_list)) != 0 else 1)
        recall = len(vp_list) / ((len(vp_list) + len(fn_list)) if (len(vp_list) + len(fn_list)) != 0 else 1)
        f1 = 2 * (precision * recall) / ((precision + recall) if (precision + recall) != 0 else 1)
        accuracy = (len(vp_list) + len(vn_list)) / ((len(vp_list) + len(vn_list) + len(fp_list) + len(fn_list)) if (len(vp_list) + len(vn_list) + len(fp_list) + len(fn_list)) != 0 else 1)

        print(f'VP: {len(vp_list)}')
        print(f'VN: {len(vn_list)}')
        print(f'FP: {len(fp_list)}')
        print(f'FN: {len(fn_list)}')

        print(f'Précision: {precision}')
        print(f'Rappel: {recall}')
        print(f'F1 Score: {f1}')
        print(f'Exactitude: {accuracy}')

        gan_decay = info["gan_decay"]
        gan_learning = info["gan_learning_rate"]
        gan_batch = info["gan_batch_size"]
        gan_size = info["gan_size"]

        if gnuplot is True:
            with open(output_file_path, 'a') as file:
                file.write(f'model\t{model}\tplayer\t{player}\tadversaire\t{adversaire}\tvp\t{len(vp_list)}\tvn\t{len(vn_list)}\tfp\t{len(fp_list)}\tfn\t{len(fn_list)}\tprecision\t{precision}\trecall\t{recall}\tf1\t{f1}\taccuracy\t{accuracy}\tgan_decay\t{gan_decay}\tgan_learning\t{gan_learning}\tgan_batch\t{gan_batch}\tgan_size\t{gan_size}\n')
            print(f'Données ajoutées dans {output_file_path}')

    def plot_results(self, file_path, fight_option = 1, model_option = 1, option = 1, proj = False):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        vp_list = []
        vn_list = []
        fp_list = []
        fn_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        accuracy_list = []

        model = str("Swap" if model_option == 1 else "GAN")
        player = str("real" if fight_option == 1 else ( "real" if fight_option == 2 else "user"))
        adversaire = str("machine" if fight_option == 1 else ( "user" if fight_option == 2 else "machine"))

        #print("Model[" + model + "] for : " + player + " vs " + adversaire)
        for line in lines:
            columns = line.strip().split('\t')

            if columns[1] == model and columns[3] == player and columns[5] == adversaire:
                vp = int(columns[7])
                vn = int(columns[9])
                fp = int(columns[11])
                fn = int(columns[13])
                precision = float(columns[15])
                recall = float(columns[17])
                f1 = float(columns[19])
                accuracy = float(columns[21])

                vp_list.append(vp)
                vn_list.append(vn)
                fp_list.append(fp)
                fn_list.append(fn)
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)
                accuracy_list.append(accuracy)

        total = sum(vp_list) + sum(vn_list) + sum(fp_list) + sum(fn_list)

        vp_list = [vp / total for vp in vp_list]
        vn_list = [vp / total for vp in vn_list]
        fp_list = [vp / total for vp in fp_list]
        fn_list = [vp / total for vp in fn_list]
        if option == 1:
            if proj is True:
                labels = ['VP', 'VN', 'FP', 'FN']
                sizes = [sum(vp_list), sum(vn_list), sum(fp_list), sum(fn_list)]
                fig = go.Figure()
                fig.add_trace(go.Pie(labels=labels, values=sizes, hole=0.4))
                fig.update_layout(title='Confusion Matrix (3D Projection)', scene=dict(aspectmode='data'))
                fig.show()
            else:
                labels = ['VP', 'VN', 'FP', 'FN']
                sizes = [sum(vp_list), sum(vn_list), sum(fp_list), sum(fn_list)]
                max_index = sizes.index(max(sizes))
                explode = [0] * 4
                explode[max_index] = 0.1  
                plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, explode=explode)
                plt.title('Confusion Matrix')
                plt.show()
        elif option == 2:
            if proj is True : 
                metrics = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
                lines = [(precision_list, 'Precision'), (recall_list, 'Recall'), (f1_list, 'F1 Score'), (accuracy_list, 'Accuracy')]
                fig = go.Figure()
                for metric_data, metric_label in lines:
                    fig.add_trace(go.Scatter3d(
                        x=list(range(len(metric_data))),
                        y=[metric_label] * len(metric_data),
                        z=metric_data,
                        mode='lines+markers',
                        name=metric_label
                    ))
                fig.update_layout(scene=dict(aspectmode="cube", xaxis=dict(title_text='Sample'), yaxis=dict(title_text='Metric'), zaxis=dict(title_text='Score')),
                                title='Precision, Recall, F1 Score, Accuracy (3D)')
                fig.show()
            else :
                plt.plot(range(len(precision_list)), precision_list, label='Precision')
                plt.plot(range(len(recall_list)), recall_list, label='Recall')
                plt.plot(range(len(f1_list)), f1_list, label='F1 Score')
                plt.plot(range(len(accuracy_list)), accuracy_list, label='Accuracy')
                plt.legend()
                plt.title('Precision, Recall, F1 Score, Accuracy')
                plt.xlabel('Sample')
                plt.ylabel('Score')
                plt.show()
        else:
            return

_Analyse = Analyse()
#_Analyse.metrique("./analyze.json", 1, True, "Swap", str("./metrics_data_1.dat"))
#_Analyse.metrique("./analyze.json", 2, True, "Swap", str("./metrics_data_2.dat"))
#_Analyse.metrique("./analyze.json", 3, True, "Swap", str("./metrics_data_3.dat"))

#_Analyse.plot_results(str("./metrics_data_1.dat"), 1, 2, 1)
#_Analyse.plot_results(str("./metrics_data_1.dat"), 2, 1, 1)
# _Analyse.plot_results(str("./metrics_data_1.dat"), 3, 1, 1,False)
# _Analyse.plot_results(str("./metrics_data_1.dat"), 3, 1, 1,True)
# #_Analyse.plot_results(str("./metrics_data_1.dat"), 1, 1, 2)
# #_Analyse.plot_results(str("./metrics_data_1.dat"), 2, 1, 2)
# _Analyse.plot_results(str("./metrics_data_1.dat"), 3, 1, 2,False)
# _Analyse.plot_results(str("./metrics_data_1.dat"), 3, 1, 2,True)
