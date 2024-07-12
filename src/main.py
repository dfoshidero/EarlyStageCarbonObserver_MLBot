from model.model_predictor import combined_prediction, parse_input

def get_user_input():
    """
    Get user input for both becd and carbenmats models.
    
    :return: tuple of dictionaries (user_input_becd, user_input_carbenmats)
    """
    user_input_becd = {
        'Building Project Type': ['New built'],
        'Primary Foundation Type': ['Piles (pile caps)'],
        'Primary Ground Floor Type': ['Suspended'],
        'Primary Vertical Element Type': ['Steel'],
        'Primary Horizontal Element Type': ['Concrete'],
        'Primary Slab Type': ['Concrete'],
        'Primary Cladding Type': ['Masonry Only'],
        'Primary Heating Type': ['Air Source Heat Pump (ASHP) - Water distribution'],
        'Primary Cooling Type': ['Chiller - Water distribution'],
        'Primary Finishes Type': ['Fully finished - Cat A'],
        'Primary Ventilation Type': ['Natural'],
    }
    
    user_input_carbenmats = {
        'Building Use Type': ['Non-residential'],
        'Building Use Subtype': ['Multi-family house'],
        'Continent': ['Europe'],
        'Country': ['Denmark'],
        'Total Users': [20],
        'Floors Above Ground': [5],
        'Floors Below Ground': [1],
        'Structure Type': ['frame wood']
    }
    
    return user_input_becd, user_input_carbenmats

def get_natural_language_input():
    text = "I am designing a concrete building in Europe for 20 users with 5 floors above ground."
    features = parse_input(text)
    return features

def main():
    user_input_becd, user_input_carbenmats = get_user_input()
    # Uncomment the line below to use natural language input
    # user_input_becd = get_natural_language_input()
    prediction = combined_prediction(user_input_becd, user_input_carbenmats)
    print("Final Prediction:", prediction)

if __name__ == "__main__":
    main()
