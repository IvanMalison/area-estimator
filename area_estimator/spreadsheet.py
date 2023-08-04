import gspread
from google.oauth2.service_account import Credentials

from . import area


class GoogleSheetUpdater:

    def __init__(
            self, project_area_estimator: area.ProjectAreaEstimator,
            google_api_key_path, spreadsheet_url,
    ):
        self.estimator = project_area_estimator
        self.google_api_key_path = google_api_key_path
        self.spreadsheet_url = spreadsheet_url
        self.scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]

        self.creds = Credentials.from_service_account_file(
            self.google_api_key_path, scopes=self.scope,
        )
        self.client = gspread.authorize(self.creds)

    def update_spreadsheet(self):
        # Open the spreadsheet
        spreadsheet = self.client.open_by_url(self.spreadsheet_url)

        # Select the first sheet
        worksheet = spreadsheet.get_worksheet(0)

        # Get the estimator data
        data = self.estimator.get_all()

        # Get the existing data
        existing_data = worksheet.get_all_values()

        # Extract existing sample ids
        id_row_map = {row[0]: i + 1 for i, row in enumerate(existing_data)}

        # Get the values in the first row (headers)
        headers = worksheet.row_values(1)[:2]

        # Define the headers
        desired_headers = ['Sample ID', 'Area']

        # Add headers only if they don't exist
        if headers != desired_headers:
            worksheet.insert_row(desired_headers, 1)

        # Append rows of data
        for sample_id, area_value in data.items():
            if sample_id in id_row_map:
                # This sample id already exists, update the row
                worksheet.update_cell(id_row_map[sample_id], 2, area_value)
            else:
                # This is a new sample id, append a new row
                worksheet.append_row([sample_id, area_value])
