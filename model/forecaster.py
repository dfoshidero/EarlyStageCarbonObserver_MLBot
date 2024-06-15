# Load the datasets
ASPECTS_ELEMENTS_MATERIALS_DF = pd.read_excel('./data/FCBS_Aspects-Elements-Materials_MachineReadable.xlsx')
BUILDUPS_DETAILS_DF = pd.read_excel('./data/FCBS_Build Ups-Details_MachineReadable.xlsx')
SECTORS_DF = pd.read_excel('./data/FCBS_Sectors-Subsectors_MachineReadable.xlsx')
ICE_DB_DF = pd.read_csv('./data/ICE DB_Cleaned.csv')
CLF_EMBODIED_CARBON_DF = pd.read_csv('./data/CLF Embodied Carbon_Cleaned.csv')
RIBA_TARGETS_DF = pd.read_excel('./data/RIBA 2030-Targets_MachineReadable.xlsx')

# Fill missing values
def fill_missing_values(df):
    return df.fillna(0)

ASPECTS_ELEMENTS_MATERIALS_DF = fill_missing_values(ASPECTS_ELEMENTS_MATERIALS_DF)
BUILDUPS_DETAILS_DF = fill_missing_values(BUILDUPS_DETAILS_DF)
SECTORS_DF = fill_missing_values(SECTORS_DF)
ICE_DB_DF = fill_missing_values(ICE_DB_DF)
CLF_EMBODIED_CARBON_DF = fill_missing_values(CLF_EMBODIED_CARBON_DF)
RIBA_TARGETS_DF = fill_missing_values(RIBA_TARGETS_DF)

# Extract unique materials, building types, and element buildups
all_materials = list(set(list(ICE_DB_DF['Material'].unique()) + list(ICE_DB_DF['Sub-material'].unique())))
all_materials = [str(material).strip().lower() for material in all_materials if pd.notna(material)]

all_building_types = list(set(list(CLF_EMBODIED_CARBON_DF['Building Type'].unique()) + 
                              list(CLF_EMBODIED_CARBON_DF['Building Use'].unique()) +
                              list(SECTORS_DF['Sector'].unique()) + 
                              list(SECTORS_DF['Building Typology'].unique()) + 
                              list(SECTORS_DF['Sub-sector'].unique())))
all_building_types = [str(building_type).strip().lower() for building_type in all_building_types if pd.notna(building_type)]

all_element_buildups = list(set(list(ASPECTS_ELEMENTS_MATERIALS_DF['Building Aspect'].unique()) + 
                                list(ASPECTS_ELEMENTS_MATERIALS_DF['Element'].unique()) + 
                                list(ASPECTS_ELEMENTS_MATERIALS_DF['Material'].unique())))
all_element_buildups = [str(build_up).strip().lower() for build_up in all_element_buildups if pd.notna(build_up)]