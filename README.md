# Rules Personal Generator

A machine learning explainability system that generates personalized rule-based explanations for predictions using LORE (Local Rule-based Explanations). This project provides APIs and interfaces for two domains: COVID-19 risk assessment and vessel movement classification.

## Overview

This project implements an explainable AI system that generates interpretable rules to explain machine learning model predictions. It uses neighborhood generation techniques and decision tree surrogates to create human-readable explanations for black-box classifiers.

### Key Features

- **Dual Domain Support**: Works with both COVID-19 risk data and vessel movement data
- **Multiple Neighborhood Generators**: 
  - Random generation
  - Genetic algorithm-based generation
  - Custom constraint-based generators
  - LLM-inspired generators
- **REST API**: FastAPI-based backend for real-time explanations
- **Interactive Web Interface**: Vue.js frontend for visualization
- **UMAP Visualization**: Dimensionality reduction for visual exploration
- **Batch Processing**: Support for batch explanation generation

## Project Structure

```
Rules_personal_generator/
├── main.py                           # COVID-19 data processing and model training
├── main_vessels.py                   # Vessel data processing and model training
├── vessels_api.py                    # FastAPI application entry point
├── covid_router.py                   # COVID-19 API endpoints
├── vessels_router.py                 # Vessel movement API endpoints
├── vessels_utils.py                  # Vessel feature definitions
├── neighbourhoodGenerator.py         # Custom neighborhood generator
├── MovementVesselRandomForest.py     # Vessel classifier training script
├── umap_xtrain.py                    # UMAP dimensionality reduction utilities
├── covid_batch_explanations.py       # Batch COVID explanation processing
├── vessels_batch_explanations.py     # Batch vessel explanation processing
├── vessels_client.py                 # API client for testing
├── datasets/                         # Training data
│   ├── Final_data.csv               # COVID-19 dataset
│   └── final_df_addedfeat.csv       # Vessel movement dataset
├── models/                           # Trained model storage
├── cvd_vue/                          # Vue.js web interface
└── requirements.txt                  # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8+
- Node.js 14+ (for web interface)
- pip package manager

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/Elecapp/Rules_personal_generator.git
cd Rules_personal_generator
```

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install LORE-SA library** (if not already installed):
```bash
# Follow the LORE-SA installation instructions from the official repository
```

4. **Build the web interface** (optional):
```bash
cd cvd_vue
npm install
npm run build
cd ..
```

## Usage

### Starting the API Server

Run the FastAPI server:

```bash
uvicorn vessels_api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- Main API: `http://localhost:8000/api`
- Interactive docs: `http://localhost:8000/api/docs`
- Web interface: `http://localhost:8000/`

### API Endpoints

#### COVID-19 Risk Explanation

**POST** `/api/covid/explain`

Generate an explanation for a COVID-19 risk prediction.

Request body:
```json
{
  "Week6_Covid": "c1",
  "Week5_Covid": "c2",
  "Week4_Covid": "c1",
  "Week3_Covid": "c1",
  "Week2_Covid": "c2",
  "Week6_Mobility": "m2",
  "Week5_Mobility": "m1",
  "Week4_Mobility": "m2",
  "Week3_Mobility": "m1",
  "Week2_Mobility": "m2",
  "Week1_Mobility": "m1",
  "Days_passed": 42,
  "neighborhood_type": "genetic"
}
```

#### Vessel Movement Explanation

**POST** `/api/vessels/explain`

Generate an explanation for a vessel movement classification.

Request body:
```json
{
  "SpeedMinimum": 0.03,
  "SpeedQ1": 15.51,
  "SpeedMedian": 15.91,
  "SpeedQ3": 16.52,
  "Log10Curvature": 0.004,
  "DistStartTrendAngle": 0.28,
  "Log10DistStartTrendDevAmplitude": 0.98,
  "MaxDistPort": 29.28,
  "Log10MinDistPort": -1.86,
  "neighborhood_type": "genetic"
}
```

### Training Models

#### COVID-19 Model

The COVID-19 model is automatically trained on first run or can be trained manually:

```python
from main import load_data_from_csv, create_and_train_model
import joblib

res = load_data_from_csv()
model = create_and_train_model(res)
joblib.dump(model, 'models/model.pkl')
```

#### Vessel Movement Model

Train the vessel classifier:

```bash
python MovementVesselRandomForest.py
```

### Batch Processing

Generate explanations for multiple instances:

```bash
# COVID-19 batch processing
python covid_batch_explanations.py

# Vessel movement batch processing
python vessels_batch_explanations.py
```

## Data Formats

### COVID-19 Data

Features:
- `Week6_Covid` through `Week2_Covid`: COVID-19 severity levels (c0-c4)
- `Week6_Mobility` through `Week1_Mobility`: Mobility levels (m0-m4)
- `Days_passed`: Number of days since initial observation
- `Class_label`: Target classification

### Vessel Movement Data

Features:
- `SpeedMinimum`: Minimum speed in trajectory
- `SpeedQ1`: First quartile of speed
- `SpeedMedian`: Median speed
- `SpeedQ3`: Third quartile of speed
- `Log10Curvature`: Log-transformed curvature measure
- `DistStartTrendAngle`: Distance from start to trend angle
- `Log10DistStartTrendDevAmplitude`: Log-transformed distance deviation amplitude
- `MaxDistPort`: Maximum distance from port
- `Log10MinDistPort`: Log-transformed minimum distance from port
- `class N`: Target classification

## Neighborhood Generation Methods

The system supports multiple neighborhood generation strategies:

1. **Random**: Uniformly random sampling from feature space
2. **Genetic**: Evolutionary algorithm-based generation
3. **Custom**: Domain-specific constraint-based generation
4. **LLM-inspired**: Uses transition probabilities derived from data

Each method generates synthetic instances around the target instance to train a local interpretable surrogate model.

## Explanation Output

The API returns:

- **Prediction**: The model's prediction for the instance
- **Probability**: Confidence of the prediction
- **Rule**: Human-readable if-then rule explaining the prediction
- **Counterfactual**: Alternative instance that would change the prediction
- **Visualization**: UMAP projection of the neighborhood

## Dependencies

Main dependencies (see `requirements.txt`):
- `deap`: Genetic algorithm framework
- `scikit-learn`: Machine learning library
- `umap-learn`: Dimensionality reduction
- `fastapi`: Web framework
- `pandas`: Data manipulation
- `numpy`: Numerical computing

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes with appropriate documentation
4. Add tests if applicable
5. Submit a pull request

## License

Please refer to the repository for license information.

## References

This project builds upon the LORE (Local Rule-based Explanations) framework for generating interpretable explanations of black-box machine learning models.

## Support

For issues and questions:
- Open an issue on GitHub
- Check the API documentation at `/api/docs`

## Acknowledgments

This project uses the LORE-SA library for generating local rule-based explanations.
