import pandas as pd
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows

# Read the workbook
xl_path = r'C:\Makerslab\Projects\IndiaAI\Indus\spreadsheet.xlsx'
xl = pd.ExcelFile(xl_path)

print("=" * 80)
print("WORKBOOK ANALYSIS")
print("=" * 80)

# Read all sheets
sheets = {}
for sheet_name in xl.sheet_names:
    sheets[sheet_name] = pd.read_excel(xl, sheet_name=sheet_name)
    print(f"\n--- {sheet_name} ---")
    print(f"Shape: {sheets[sheet_name].shape}")
    print(f"Columns: {list(sheets[sheet_name].columns)}")
    print(f"\nFirst few rows:")
    print(sheets[sheet_name].head(10))
    print("\n")

# Analyze Core Evaluation Pipeline
print("\n" + "=" * 80)
print("CORE EVALUATION PIPELINE ANALYSIS")
print("=" * 80)
core = sheets['Core Evaluation Pipeline']
print(core)

# Check for detail sheets reference
print("\n" + "=" * 80)
print("DETAIL SHEETS ANALYSIS")
print("=" * 80)

for detail_sheet in ['Language Understanding Detail', 'Language Generation Detail',
                       'Domain Physics Coverage', 'Safety & Ethics', 'Pedagogical Quality']:
    if detail_sheet in sheets:
        print(f"\n--- {detail_sheet} ---")
        detail = sheets[detail_sheet]
        print(f"Shape: {detail.shape}")
        print(detail)
