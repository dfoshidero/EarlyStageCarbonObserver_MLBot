from model_predictor import load_resources, predict as model_predict
from feature_extractor import extract
import time
import numpy as np
import os
import psutil
import gc

# Load resources
model, features, label_encoders, unique_values = load_resources()


def predict(
    SECTOR,
    SUBSECTOR,
    GIA,
    PERIMETER,
    FOOTPRINT,
    WIDTH,
    HEIGHT,
    ABOVE_GROUND,
    BELOW_GROUND,
    GLAZING_RATIO,
    PILES,
    PILE_CAPS,
    CAPPING_BEAMS,
    RAFT,
    BASEMENT_WALLS,
    LOWEST_FLOOR_SLAB,
    GROUND_INSULATION,
    CORE_STRUCTURE,
    COLUMNS,
    BEAMS,
    SECONDARY_BEAMS,
    FLOOR_SLAB,
    JOISTED_FLOORS,
    ROOF,
    ROOF_INSULATION,
    ROOF_FINISHES,
    FACADE,
    WALL_INSULATION,
    GLAZING,
    WINDOW_FRAMES,
    PARTITIONS,
    CEILINGS,
    FLOORS,
    SERVICES,
):
    """
    Get user input for the new columns and make a prediction.

    :return: list with prediction values
    """
    user_input = {
        "Sector": [None if SECTOR == "None" else SECTOR],
        "Sub-Sector": [None if SUBSECTOR == "None" else SUBSECTOR],
        "Gross Internal Area (m2)": [None if GIA == "None" else GIA],
        "Building Perimeter (m)": [None if PERIMETER == "None" else PERIMETER],
        "Building Footprint (m2)": [None if FOOTPRINT == "None" else FOOTPRINT],
        "Building Width (m)": [None if WIDTH == "None" else WIDTH],
        "Floor-to-Floor Height (m)": [None if HEIGHT == "None" else HEIGHT],
        "Storeys Above Ground": [None if ABOVE_GROUND == "None" else ABOVE_GROUND],
        "Storeys Below Ground": [None if BELOW_GROUND == "None" else BELOW_GROUND],
        "Glazing Ratio (%)": [None if GLAZING_RATIO == "None" else GLAZING_RATIO],
        "Piles Material": [None if PILES == "None" else PILES],
        "Pile Caps Material": [None if PILE_CAPS == "None" else PILE_CAPS],
        "Capping Beams Material": [None if CAPPING_BEAMS == "None" else CAPPING_BEAMS],
        "Raft Foundation Material": [None if RAFT == "None" else RAFT],
        "Basement Walls Material": [
            None if BASEMENT_WALLS == "None" else BASEMENT_WALLS
        ],
        "Lowest Floor Slab Material": [
            None if LOWEST_FLOOR_SLAB == "None" else LOWEST_FLOOR_SLAB
        ],
        "Ground Insulation Material": [
            None if GROUND_INSULATION == "None" else GROUND_INSULATION
        ],
        "Core Structure Material": [
            None if CORE_STRUCTURE == "None" else CORE_STRUCTURE
        ],
        "Columns Material": [None if COLUMNS == "None" else COLUMNS],
        "Beams Material": [None if BEAMS == "None" else BEAMS],
        "Secondary Beams Material": [
            None if SECONDARY_BEAMS == "None" else SECONDARY_BEAMS
        ],
        "Floor Slab Material": [None if FLOOR_SLAB == "None" else FLOOR_SLAB],
        "Joisted Floors Material": [
            None if JOISTED_FLOORS == "None" else JOISTED_FLOORS
        ],
        "Roof Material": [None if ROOF == "None" else ROOF],
        "Roof Insulation Material": [
            None if ROOF_INSULATION == "None" else ROOF_INSULATION
        ],
        "Roof Finishes Material": [None if ROOF_FINISHES == "None" else ROOF_FINISHES],
        "Facade Material": [None if FACADE == "None" else FACADE],
        "Wall Insulation Material": [
            None if WALL_INSULATION == "None" else WALL_INSULATION
        ],
        "Glazing Material": [None if GLAZING == "None" else GLAZING],
        "Window Frames Material": [None if WINDOW_FRAMES == "None" else WINDOW_FRAMES],
        "Partitions Material": [None if PARTITIONS == "None" else PARTITIONS],
        "Ceilings Material": [None if CEILINGS == "None" else CEILINGS],
        "Floors Material": [None if FLOORS == "None" else FLOORS],
        "Services": [None if SERVICES == "None" else SERVICES],
    }

    prediction = model_predict(user_input, model, features, label_encoders)
    prediction_list = (
        prediction.tolist() if isinstance(prediction, np.ndarray) else prediction
    )

    log_memory_usage("During Prediction")

    return prediction_list


def log_memory_usage(phase):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    gc.collect()
    print(
        f"[{phase}] Memory Usage: RSS={memory_info.rss / (1024 * 1024):.2f} MB, VMS={memory_info.vms / (1024 * 1024):.2f} MB"
    )


def get_natural_language_input(text):
    """
    Extracts structured data from a given natural language input.

    :param text: The natural language input text.
    :return: tuple of extracted values
    """
    value_list = extract(text)
    SECTOR = value_list.get("Sector")
    SUBSECTOR = value_list.get("Sub-Sector")
    GIA = value_list.get("Gross Internal Area (m2)")
    PERIMETER = value_list.get("Building Perimeter (m)")
    FOOTPRINT = value_list.get("Building Footprint (m2)")
    WIDTH = value_list.get("Building Width (m)")
    HEIGHT = value_list.get("Floor-to-Floor Height (m)")
    ABOVE_GROUND = value_list.get("Storeys Above Ground")
    BELOW_GROUND = value_list.get("Storeys Below Ground")
    GLAZING_RATIO = value_list.get("Glazing Ratio (%)")
    PILES = value_list.get("Piles Material")
    PILE_CAPS = value_list.get("Pile Caps Material")
    CAPPING_BEAMS = value_list.get("Capping Beams Material")
    RAFT = value_list.get("Raft Foundation Material")
    BASEMENT_WALLS = value_list.get("Basement Walls Material")
    LOWEST_FLOOR_SLAB = value_list.get("Lowest Floor Slab Material")
    GROUND_INSULATION = value_list.get("Ground Insulation Material")
    CORE_STRUCTURE = value_list.get("Core Structure Material")
    COLUMNS = value_list.get("Columns Material")
    BEAMS = value_list.get("Beams Material")
    SECONDARY_BEAMS = value_list.get("Secondary Beams Material")
    FLOOR_SLAB = value_list.get("Floor Slab Material")
    JOISTED_FLOORS = value_list.get("Joisted Floors Material")
    ROOF = value_list.get("Roof Material")
    ROOF_INSULATION = value_list.get("Roof Insulation Material")
    ROOF_FINISHES = value_list.get("Roof Finishes Material")
    FACADE = value_list.get("Facade Material")
    WALL_INSULATION = value_list.get("Wall Insulation Material")
    GLAZING = value_list.get("Glazing Material")
    WINDOW_FRAMES = value_list.get("Window Frames Material")
    PARTITIONS = value_list.get("Partitions Material")
    CEILINGS = value_list.get("Ceilings Material")
    FLOORS = value_list.get("Floors Material")
    SERVICES = value_list.get("Services")

    return (
        SECTOR,
        SUBSECTOR,
        GIA,
        PERIMETER,
        FOOTPRINT,
        WIDTH,
        HEIGHT,
        ABOVE_GROUND,
        BELOW_GROUND,
        GLAZING_RATIO,
        PILES,
        PILE_CAPS,
        CAPPING_BEAMS,
        RAFT,
        BASEMENT_WALLS,
        LOWEST_FLOOR_SLAB,
        GROUND_INSULATION,
        CORE_STRUCTURE,
        COLUMNS,
        BEAMS,
        SECONDARY_BEAMS,
        FLOOR_SLAB,
        JOISTED_FLOORS,
        ROOF,
        ROOF_INSULATION,
        ROOF_FINISHES,
        FACADE,
        WALL_INSULATION,
        GLAZING,
        WINDOW_FRAMES,
        PARTITIONS,
        CEILINGS,
        FLOORS,
        SERVICES,
    )


def main():
    def time_it(description, func, *args, **kwargs):
        print(description)
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return result, elapsed_time

    text = (
        "A residential concrete building with raft, a basement and timber joists floors"
    )

    print(f"From text: {text}.")
    inputs, time_elapsed_extraction = time_it(
        "Extracting features \n", get_natural_language_input, text
    )
    prediction, time_elapsed_prediction = time_it("Making prediction", predict, *inputs)
    print("\n")
    print("Final Prediction:", prediction)

    time_elapsed = time_elapsed_extraction + time_elapsed_prediction
    print("\n")
    print(f"Time elapsed: {time_elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
