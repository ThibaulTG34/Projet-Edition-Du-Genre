# This Python file uses the following encoding: utf-8
import gspread
from google.auth import exceptions
from google.oauth2.service_account import Credentials

class Analyse:
    def __init__(self):
        super().__init__()
        return
        try:
            credentials = Credentials.from_service_account_file('credentials.json', scopes=['url'])
            gc = gspread.authorize(credentials)
        except exceptions.DefaultCredentialsError as e:
            raise ValueError("Impossible de charger les informations d'identification.")

        self.sheet = gc.open('ProjetGenre')

    def update_data(self, data):
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

#["NomPersonne", 1, 2, "Résultat"]


