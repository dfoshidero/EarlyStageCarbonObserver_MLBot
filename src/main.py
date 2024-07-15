from model_predictor import combined_prediction, validate_user_input, unique_values

def get_user_input():
    """
    Get user input for the new columns.
    
    :return: dictionary with user input values for the new columns
    """
    user_input = {
            'Sector': ['Housing'],
            'Sub-Sector': ['High-Rise Apartments/Hotels'],
            'Gross Internal Area (m2)': [1800],
            'Building Perimeter (m)': [1000],
            'Building Footprint (m2)': [648],
            'Building Width (m)': [23],
            'Floor-to-Floor Height (m)': [3.8],
            'Storeys Above Ground': [38],
            'Storeys Below Ground': [4],
            'Glazing Ratio (%)': [48],
            'Piles Material': ['Reinforced Concrete'],
            'Pile Caps Material': ['Reinforced Concrete'],
            'Capping Beams Material': ['Other'],
            'Raft Foundation Material': ['Other'],
            'Basement Walls Material': ['Reinforced Concrete'],
            'Lowest Floor Slab Material': ['Beam and Block'],
            'Ground Insulation Material': ['Glass mineral wool'],
            'Core Structure Material': ['Other'],
            'Columns Material': ['Glulam'],
            'Beams Material': ['Precast Concrete'],
            'Secondary Beams Material': ['Reinforced Concrete'],
            'Floor Slab Material': ['Other'],
            'Joisted Floors Material': ['JJI Engineered Joists'],
            'Roof Material': ['Precast Concrete'],
            'Roof Insulation Material': ['Vacuum Insulation'],
            'Roof Finishes Material': ['Bitumous Sheet'],
            'Facade Material': ['SFS with Granite'],
            'Wall Insulation Material': ['Other'],
            'Glazing Material': ['Triple Glazing'],
            'Window Frames Material': ['uPVC'],
            'Partitions Material': ['Plywood // Timber Studs'],
            'Ceilings Material': ['Other'],
            'Floors Material': ['Solid timber floorboards'],
            'Services': ['High']
        }
    
    return user_input

def main():
    user_input = get_user_input()
    
    # Validate user input
    validate_user_input(user_input, unique_values)
    
    # Uncomment the line below to use natural language input
    # user_input = get_natural_language_input()
    prediction = combined_prediction(user_input)
    print("Final Prediction:", prediction)

if __name__ == "__main__":
    main()
