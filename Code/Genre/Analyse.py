# This Python file uses the following encoding: utf-8
import gspread
import json
import os
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

    def update_or_create_entry(self, s, s2, option = 0):
        if not option==0 :
            s = os.path.basename(s)
        try:
            with open(self.path_file, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            data = {}

        _class = classified(s)
        if s in data:
            data[s]["real"] = s2
            data[s]["classificateur"] = _class
        else:
            data[s] = {"real": s2, "classificateur" : _class}

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
