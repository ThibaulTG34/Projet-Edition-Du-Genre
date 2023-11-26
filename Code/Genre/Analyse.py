# This Python file uses the following encoding: utf-8
import gspread
import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
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

    def update_or_create_entry(self, s, s2, s3 = str("Unknow"), s4 = str("Swap"), option = 0):

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
        else:
            if s2 is None: s2 = str("Unknow")
            if s3 is None: s3 = str("Unknow")
            if s4 is None: s4 = str("Unknow")
            data[s] = {"real": s2, "classificateur" : _class, "user" : s3, "model" : s4}

        with open(self.path_file, 'w') as file:
            json.dump(data, file, indent=2)

    def init_sheet(self):
        try:
            credentials = Credentials.from_service_account_file('credentials.json', scopes=['url'])
            gc = gspread.authorize(credentials)
            self.sheet = gc.open('ProjetGenre')
        except exceptions.DefaultCredentialsError as e:
            raise ValueError("Impossible de charger les informations d'identification.")

        self.sheet = gc.open('ProjetGenre')

    def update_sheet(self, data):
        worksheet = self.sheet.worksheet("ProjectGenre")

        records = worksheet.get_all_records()

        record_exists = False
        for record in records:
            if record["Name"] == data[0]:
                record["Homme"] = data[1]
                record["Femme"] = data[2]
                record["Res"] = data[3]
                record_exists = True
                break

        if not record_exists:
            worksheet.append_table(data)


    #Vrais Positifs (VP): Nombre d'images correctement classées comme "Male".
    #Vrais Négatifs (VN): Nombre d'images correctement classées comme "Female".
    #Faux Positifs (FP): Nombre d'images incorrectement classées comme "Male" alors qu'elles sont réellement "Female".
    #Faux Négatifs (FN): Nombre d'images incorrectement classées comme "Female" alors qu'elles sont réellement "Male".

    def metrique(self, path, option = 1, gnuplot = False, ploting = False, output_file_path = str("metrics_data.dat")):

        with open(path, 'r') as f:
            data = json.load(f)

        vp_list, vn_list, fp_list, fn_list = [], [], [], []

        for filename, info in data.items():
            real = info['real']
            classificateur = info['classificateur']
            user = info['user']
            model = info['model']

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

        if gnuplot is True:
            with open(output_file_path, 'a') as file:
                file.write(f'{model}\t{len(vp_list)}\t{len(vn_list)}\t{len(fp_list)}\t{len(fn_list)}\t{precision}\t{recall}\t{f1}\t{accuracy}\n')
            print(f'Données ajoutées dans {output_file_path}')

        if ploting is True:
            confusion_matrix_names = ['VP', 'VN', 'FP', 'FN']
            confusion_matrix_values = [len(vp_list), len(vn_list), len(fp_list), len(fn_list)]
            metrics_curve_names = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
            metrics_curve_values = [precision, recall, f1, accuracy]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(confusion_matrix_names, confusion_matrix_values, color=['green', 'blue', 'red', 'purple'])
            ax.set_ylabel('Nombre d\'échantillons')
            ax.set_title('Confusion Matrix Metrics')
            plt.show()

            for metric_name, metric_value in zip(metrics_curve_names, metrics_curve_values):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot([metric_name], [metric_value], marker='o', linestyle='-', color='b')
                ax.set_ylim([0, 1])
                ax.set_ylabel('Valeur')
                ax.set_title(f'{metric_name} Curve')
                plt.show()
