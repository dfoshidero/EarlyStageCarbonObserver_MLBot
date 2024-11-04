# ECO (Early-stage Carbon Observer)

## Overview
ECO (Early-stage Carbon Observer) is a machine learning-based tool designed to assist architects and designers in predicting the embodied carbon impact of building designs during the early stages of the design process. This tool is capable of translating textual design descriptions into accurate carbon footprint predictions, enabling more informed and sustainable design decisions from the outset.

## Features
- **Natural Language Processing (NLP)**: Converts unstructured textual design descriptions into structured data for carbon footprint regression analysis.
- **Machine Learning Predictions**: Uses HGB regression to estimate the embodied carbon of a building's lifecycle from early-stage design inputs.
- **Real-Time Feedback**: Provides immediate feedback on design choices, for rapid iterations and more sustainable design outcomes.

## How It Works
1. **Input**: Users can input textual descriptions of building designs or structured data related to the building's specifications.
2. **Feature Extraction**: The tool extracts key features from the input using NLP techniques, such as named entity recognition (NER) and semantic analysis.
3. **Prediction**: The extracted features are fed into the machine learning model to predict the embodied carbon footprint of the building design.
4. **Output**: The tool provides a numerical prediction of the carbon impact, expressed in kgCO2e/m², for the different lifecycle stages of the building.


## Installation

### Prerequisites
- Python 3.10 or higher
- Required packages as listed in `requirements_app.txt` and `requirements_trainer.txt`

### Setting Up the Environment
1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/ECO.git
    cd ECO
    ```

2. **Install dependencies**:
    For the application:
    ```bash
    pip install -r requirements_app.txt
    ```

    For the training environment:
    ```bash
    pip install -r requirements_trainer.txt
    ```
    
## Usage
Input a textual description of the building, and run the code. The tool will extract the relevant features and make a carbon prediction.

Data is not included in this repository. Though, theoretically , the regression model training should work for any regression data as long as they are cleaned correctly. The "prep" files are catered to my datasets.

This tool might be adapted to generate predictions from textual data in any field.

# Contributors
Contributions are welcome! Please fork the repository and submit a pull request with your enhancements or bug fixes.
![Contributors](https://contrib.rocks/image?repo=dfoshidero/ECO)

# Acknowledgements
This tool is inspired by the University of Bath's ZEBRA toolkit and Feilden Clegg Bradley Studio's CARBON tool, and incorporates insights from recent research on sustainable design and machine learning applications in architecture.
Thank you to Professors David Coley and Michael Tipping for their guidance, to the University of Bath for their support.
 
