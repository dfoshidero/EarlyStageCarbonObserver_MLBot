from flask import Flask, request, jsonify
from model_predictor import predictor
from feature_extractor import extract

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json()
    prediction = predict(
        data.get("SECTOR"),
        data.get("SUBSECTOR"),
        data.get("GIA"),
        data.get("PERIMETER"),
        data.get("FOOTPRINT"),
        data.get("WIDTH"),
        data.get("HEIGHT"),
        data.get("ABOVE_GROUND"),
        data.get("BELOW_GROUND"),
        data.get("GLAZING_RATIO"),
        data.get("PILES"),
        data.get("PILE_CAPS"),
        data.get("CAPPING_BEAMS"),
        data.get("RAFT"),
        data.get("BASEMENT_WALLS"),
        data.get("LOWEST_FLOOR_SLAB"),
        data.get("GROUND_INSULATION"),
        data.get("CORE_STRUCTURE"),
        data.get("COLUMNS"),
        data.get("BEAMS"),
        data.get("SECONDARY_BEAMS"),
        data.get("FLOOR_SLAB"),
        data.get("JOISTED_FLOORS"),
        data.get("ROOF"),
        data.get("ROOF_INSULATION"),
        data.get("ROOF_FINISHES"),
        data.get("FACADE"),
        data.get("WALL_INSULATION"),
        data.get("GLAZING"),
        data.get("WINDOW_FRAMES"),
        data.get("PARTITIONS"),
        data.get("CEILINGS"),
        data.get("FLOORS"),
        data.get("SERVICES"),
    )
    return jsonify(prediction)


@app.route("/extract", methods=["POST"])
def extract_route():
    text = request.get_json().get("text")
    extracted_values = extract(text)
    return jsonify(extracted_values)


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

    prediction = predictor(user_input)

    return prediction


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=80)
