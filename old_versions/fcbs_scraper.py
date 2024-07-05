import pandas as pd
import itertools
from openpyxl import load_workbook
import numpy as np
import os
import win32com.client as win32


# Load the Excel file
file_path = os.path.abspath('data/raw/preprocess/FCBS_Carbon_V0.8.4_BETA.xlsm')
output_file_path = os.path.abspath('/data/processed/all_building_variants.xlsx')

# Define the ranges for user inputs required
building_perimeter_range = np.arange(50, 1001, 50)
building_footprint_range = np.arange(100, 10001, 100)
building_width_range = np.arange(10, 201, 10)
floor_to_floor_height_range = np.arange(2.5, 6.5, 0.5)
no_storeys_ground_above_range = np.arange(1, 51)
no_storeys_below_ground_range = np.arange(0, 6)
glazing_ratio_range = np.arange(10, 91, 10)
gia_ranges = np.concatenate((np.arange(250, 2001, 250), np.arange(2000, 15001, 500)))

# Manually define dropdown options for each field
sector_options = ["Housing", "Office"]
sub_sector_options = {
    "Housing": ["Flat/maisonette", "Single family house", "Multi-family (< 6 storeys)", "Multi-family (6 - 15 storeys)", "Multi-family (> 15 storeys)"],
    "Office": ["Office"]
}

# Define all building elements and their corresponding materials
building_elements = [
    'Piles', 'Pile caps', 'Capping beams', 'Raft', 'Basement walls', 'Lowest floor slab', 
    'Ground insulation', 'Core structure', 'Columns', 'Beams', 'Secondary beams', 'Floor slab',
    'Joisted floors', 'Roof', 'Roof insulation', 'Roof finishes', 'Facade', 'Wall insulation',
    'Glazing', 'Window frames', 'Partitions', 'Ceilings', 'Floors', 'Services'
]
material_options = {
    'Piles': ["RC 32/40 (50kg/m3 reinforcement)", "RC 32/40 25% GGBS (50kg/m3 reinforcement)", "RC 32/40 50% GGBS (50kg/m3 reinforcement)", "RC 32/40 70% GGBS (50kg/m3 reinforcement)", "Steel", ""],
    'Pile caps': ["RC 32/40 (200kg/m3 reinforcement)", "RC 32/40 25% GGBS (200kg/m3 reinforcement)", "RC 32/40 50% GGBS (200kg/m3 reinforcement)", "RC 32/40 70% GGBS (200kg/m3 reinforcement)", ""],
    'Capping beams': ["RC 32/40 (200kg/m3 reinforcement)", "RC 32/40 25% GGBS (200kg/m3 reinforcement)", "RC 32/40 50% GGBS (200kg/m3 reinforcement)", "RC 32/40 70% GGBS (200kg/m3 reinforcement)", "Foamglass (domestic only)", ""],
    'Raft': ["RC 32/40 (150kg/m3 reinforcement)", "RC 32/40 25% GGBS (150kg/m3 reinforcement)", "RC 32/40 50% GGBS (150kg/m3 reinforcement)", "RC 32/40 70% GGBS (150kg/m3 reinforcement)", ""],
    'Basement walls': ["RC 32/40 (125kg/m3 reinforcement)", "RC 32/40 25% GGBS (125kg/m3 reinforcement)", "RC 32/40 50% GGBS (125kg/m3 reinforcement)", "RC 32/40 70% GGBS (125kg/m3 reinforcement)", ""],
    'Lowest floor slab': ["RC 32/40 (150kg/m3 reinforcement)", "RC 32/40 25% GGBS (150kg/m3 reinforcement)", "RC 32/40 50% GGBS (150kg/m3 reinforcement)", "RC 32/40 70% GGBS (150kg/m3 reinforcement)", "Beam and Block", ""],
    'Ground insulation': ["EPS", "XPS", "Glass mineral wool", ""],
    'Core structure': ["CLT", "Precast RC 32/40 (100kg/m3 reinforcement)", "RC 32/40 (100kg/m3 reinforcement)", "RC 32/40 25% GGBS (100kg/m3 reinforcement)", "RC 32/40 50% GGBS (100kg/m3 reinforcement)", "RC 32/40 70% GGBS (100kg/m3 reinforcement)", ""],
    'Columns': ["Glulam", "Iron (existing buildings)", "Precast RC 32/40 (300kg/m3 reinforcement)", "RC 32/40 (300kg/m3 reinforcement)", "RC 32/40 25% GGBS (300kg/m3 reinforcement)", "RC 32/40 50% GGBS (300kg/m3 reinforcement)", "RC 32/40 70% GGBS (300kg/m3 reinforcement)", "Steel", ""],
    'Beams': ["Glulam", "Iron (existing buildings)", "Precast RC 32/40 (250kg/m3 reinforcement)", "RC 32/40 (250kg/m3 reinforcement)", "RC 32/40 25% GGBS (250kg/m3 reinforcement)", "RC 32/40 50% GGBS (250kg/m3 reinforcement)", "RC 32/40 70% GGBS (250kg/m3 reinforcement)", "Steel", ""],
    'Secondary beams': ["Glulam", "Iron (existing buildings)", "Precast RC 32/40 (250kg/m3 reinforcement)", "RC 32/40 (250kg/m3 reinforcement)", "RC 32/40 25% GGBS (250kg/m3 reinforcement)", "RC 32/40 50% GGBS (250kg/m3 reinforcement)", "RC 32/40 70% GGBS (250kg/m3 reinforcement)", "Steel", ""],
    'Floor slab': ["CLT", "Precast RC 32/40 (100kg/m3 reinforcement)", "RC 32/40 (100kg/m3 reinforcement)", "RC 32/40 25% GGBS (100kg/m3 reinforcement)", "RC 32/40 50% GGBS (100kg/m3 reinforcement)", "RC 32/40 70% GGBS (100kg/m3 reinforcement)", "Steel Concrete Composite", ""],
    'Joisted floors': ["JJI Engineered Joists + OSB topper", "Timber Joists + OSB topper (Domestic)", "Timber Joists + OSB topper (Office)", ""],
    'Roof': ["CLT", "Metal Deck", "Precast RC 32/40 (100kg/m3 reinforcement)", "RC 32/40 (100kg/m3 reinforcement)", "RC 32/40 25% GGBS (100kg/m3 reinforcement)", "RC 32/40 50% GGBS (100kg/m3 reinforcement)", "RC 32/40 70% GGBS (100kg/m3 reinforcement)", "Steel Concrete Composite", "Timber Cassette", "Timber Pitch Roof", ""],
    'Roof insulation': ["Cellulose, loose fill", "EPS", "Expanded Perlite", "Expanded Vermiculite", "Glass mineral wool", "PIR", "Rockwool", "Sheeps wool", "Vacuum Insulation", "Woodfibre", "XPS", ""],
    'Roof finishes': ["Aluminium", "Asphalt (Mastic)", "Asphalt (Polymer modified)", "Bitumous Sheet", "Ceramic tile", "Fibre cement tile", "Green Roof", "Roofing membrane (PVC)", "Slate tile", "Zinc Standing Seam", ""],
    'Facade': ["Blockwork with Brick", "Blockwork with render", "Blockwork with Timber", "Curtain Walling", "Load Bearing Precast Concrete Panel", "Load Bearing Precast Concrete with Brick Slips", "Party Wall Blockwork", "Party Wall Brick", "Party Wall Timber Cassette",
               "SFS with Aluminium Cladding", "SFS with Brick", "SFS with Ceramic Tiles", "SFS with Granite", "SFS with Limestone", "SFS with Zinc Cladding", "Solid Brick, single leaf", "Timber Cassette Panel with brick", "Timber Cassette Panel with Cement Render",
               "Timber Cassette Panel with Larch Cladding ", "Timber Cassette Panel with Lime Render", "Timber SIPs with Brick", ""],
    'Wall insulation': ["Cellulose, loose fill", "EPS", "Expanded Perlite", "Expanded Vermiculite", "Glass mineral wool", "PIR", "Rockwool", "Sheeps wool", "Vacuum Insulation", "Woodfibre", "XPS", ""],
    'Glazing': ["Triple Glazing", "Double Glazing", "Single Glazing", ""],
    'Window frames': ["Al/Timber Composite", "Aluminium", "Steel (single glazed)", "Solid softwood timber frame", "uPVC", ""],
    'Partitions': ["CLT", "Plasterboard + Steel Studs", "Plasterboard + Timber Studs", "Plywood + Timber Studs", "Blockwork", ""],
    'Ceilings': ["Exposed Soffit", "Plasterboard", "Steel grid system", "Steel tile", "Steel tile with 18mm acoustic pad", "Suspended plasterboard", ""],
    'Floors': ["70mm screed", "Carpet", "Earthenware tile", "Raised floor", "Solid timber floorboards", "Stoneware tile", "Terrazzo", "Vinyl", ""],
    'Services': ["Low", "Medium", "High", ""]
}

# Generate all valid combinations
def generate_combinations(*args):
    return itertools.product(*args)

def force_recalculate(file_path):
    excel = win32.DispatchEx("Excel.Application")
    excel.Visible = False
    wb = excel.Workbooks.Open(file_path)
    wb.RefreshAll()
    excel.CalculateUntilAsyncQueriesDone()
    wb.Save()
    wb.Close()
    excel.Quit()

def set_input_values(excel, wb, sector, sub_sector, gia, perimeter, footprint, width, height, storeys_above, storeys_below, glazing_ratio, combination):
    ws_project = wb.Sheets['0. INPUT Project Details']
    ws_embodied = wb.Sheets['2. INPUT Embodied Carbon']
    
    ws_project.Cells(9, 2).Value = sector
    ws_project.Cells(10, 2).Value = sub_sector
    ws_project.Cells(11, 2).Value = gia
    ws_embodied.Cells(19, 2).Value = perimeter
    ws_embodied.Cells(20, 2).Value = footprint
    ws_embodied.Cells(21, 2).Value = width
    ws_embodied.Cells(22, 2).Value = height
    ws_embodied.Cells(23, 2).Value = storeys_above
    ws_embodied.Cells(24, 2).Value = storeys_below
    ws_embodied.Cells(25, 2).Value = glazing_ratio
    
    for idx, elem in enumerate(building_elements):
        material_cell = f'C{29 + idx}'  # Assuming materials start from row 29
        ws_embodied.Cells(29 + idx, 3).Value = combination[idx]

def extract_embodied_carbon(wb):
    ws_output = wb.Sheets['5. OUTPUT Machine']
    return ws_output.Cells(19, 2).Value

def extract_grid_size(wb):
    ws_project = wb.Sheets['0. INPUT Project Details']
    return ws_project.Cells(12, 2).Value


def save_partial_results(df, file_path):
    if not os.path.exists(file_path):
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
    else:
        with pd.ExcelWriter(file_path, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1', startrow=writer.sheets['Sheet1'].max_row, header=False)
    print("Partial data saved.")

# Master DataFrame to store all variants
master_df = pd.DataFrame()

# Excel application object
excel = win32.DispatchEx("Excel.Application")
excel.Visible = False

# Iterate through all combinations and populate the master DataFrame
batch_size = 1
batch_counter = 0
for sector in sector_options:
    for sub_sector in sub_sector_options[sector]:
        for gia in gia_ranges:
            for perimeter in building_perimeter_range:
                for footprint in building_footprint_range:
                    for width in building_width_range:
                        for height in floor_to_floor_height_range:
                            for storeys_above in no_storeys_ground_above_range:
                                for storeys_below in no_storeys_below_ground_range:
                                    for glazing_ratio in glazing_ratio_range:
                                        material_combinations = generate_combinations(*(material_options[elem] for elem in building_elements))
                                        for material_combination in material_combinations:
                                            # Load the Excel file for each combination
                                            wb = excel.Workbooks.Open(file_path)

                                            # Process the combination
                                            set_input_values(excel, wb, sector, sub_sector, gia, perimeter, footprint, width, height, storeys_above, storeys_below, glazing_ratio, material_combinations)

                                            # Force recalculation to update formulas
                                            wb.RefreshAll()
                                            excel.CalculateUntilAsyncQueriesDone()
                                            
                                            # Extract the final embodied carbon value
                                            embodied_carbon = extract_embodied_carbon(wb)
                                            
                                            # Extract dependent fields (grid size)
                                            grid_size = extract_grid_size(wb)

                                            # Create a dictionary for the current variant
                                            variant_dict = {
                                                'Sector': sector,
                                                'Sub-sector': sub_sector,
                                                'GIA (m2)': gia,
                                                'Building perimeter': perimeter,
                                                'Building footprint': footprint,
                                                'Building width': width,
                                                'Floor-to-floor height': height,
                                                'No. storeys ground & above': storeys_above,
                                                'No. storeys below ground': storeys_below,
                                                'Glazing ratio': glazing_ratio,
                                                'Grid size': grid_size,
                                                'Embodied Carbon (kgCO2e/m2)': embodied_carbon
                                            }

                                            # Add each building element's material to the dictionary
                                            for elem, material in zip(building_elements, material_combinations):
                                                variant_dict[f'{elem} Material'] = material

                                            # Append the variant to the master_df
                                            master_df = pd.concat([master_df, pd.DataFrame([variant_dict])], ignore_index=True)

                                            # Save the DataFrame to the Excel file after every batch
                                            batch_counter += 1
                                            if batch_counter % batch_size == 0:
                                                save_partial_results(master_df, output_file_path)
                                                master_df = pd.DataFrame()  # Clear the DataFrame to free memory
                                            wb.Close(SaveChanges=False)

print("Data processing complete.")
excel.Quit()
