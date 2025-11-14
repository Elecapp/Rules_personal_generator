# API Documentation

This document provides detailed API documentation for the Rules Personal Generator REST API.

## Base URL

```
http://localhost:8000/api
```

## Interactive Documentation

The API provides auto-generated interactive documentation:
- **Swagger UI**: `http://localhost:8000/api/docs`
- **ReDoc**: `http://localhost:8000/api/redoc`

## Endpoints

### COVID-19 Endpoints

#### POST /api/covid/explain

Generate a rule-based explanation for a COVID-19 risk prediction.

**Request Body:**

```json
{
  "event": {
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
    "Days_passed": 42
  },
  "neighborhood_type": "genetic",
  "num_samples": 2000
}
```

**Parameters:**

- `event` (CovidEvent): COVID-19 instance to explain
  - `Week6_Covid` through `Week2_Covid`: COVID severity levels (c0-c4 or NONE)
  - `Week6_Mobility` through `Week1_Mobility`: Mobility levels (m0-m4 or NONE)
  - `Days_passed`: Integer representing days since initial observation
- `neighborhood_type` (string, optional): Type of neighborhood generator
  - Options: `"train"`, `"random"`, `"custom"`, `"genetic"`, `"gpt"`
  - Default: `"genetic"`
- `num_samples` (integer, optional): Number of synthetic instances to generate
  - Default: 2000
  - Range: 100-10000

**Response:**

```json
{
  "prediction": 2,
  "probability": 0.85,
  "rule": {
    "premises": [
      {"attr": "Week6_Covid", "op": ">", "val": "c1"},
      {"attr": "Days_passed", "op": "<=", "val": 100}
    ],
    "consequence": {"attr": "Class_label", "val": 2}
  },
  "counterfactuals": [...],
  "visualization": {...},
  "neighborhood_stats": {
    "size": 2000,
    "type": "genetic"
  }
}
```

**Response Fields:**

- `prediction`: Predicted class label (integer)
- `probability`: Confidence of prediction (0-1)
- `rule`: Main rule explaining the prediction
  - `premises`: List of conditions (feature, operator, value)
  - `consequence`: Predicted class
- `counterfactuals`: List of alternative rules for different outcomes
- `visualization`: Vega-Lite specification for UMAP projection
- `neighborhood_stats`: Information about generated neighborhood

**Status Codes:**

- `200 OK`: Explanation generated successfully
- `400 Bad Request`: Invalid input parameters
- `500 Internal Server Error`: Error during explanation generation

#### POST /api/covid/explainBatch

Generate explanations for multiple COVID-19 instances.

**Request Body:**

```json
{
  "events": [
    {
      "Week6_Covid": "c1",
      "Week5_Covid": "c2",
      ...
    },
    {
      "Week6_Covid": "c2",
      "Week5_Covid": "c3",
      ...
    }
  ],
  "neighborhood_types": ["random", "genetic"],
  "num_samples": 2000
}
```

**Response:**

Array of explanation objects, one per input instance.

---

### Vessel Movement Endpoints

#### POST /api/vessels/explain

Generate a rule-based explanation for a vessel movement classification.

**Request Body:**

```json
{
  "vessel_event": {
    "SpeedMinimum": 0.03,
    "SpeedQ1": 15.51,
    "SpeedMedian": 15.91,
    "SpeedQ3": 16.52,
    "Log10Curvature": 0.004,
    "DistStartTrendAngle": 0.28,
    "Log10DistStartTrendDevAmplitude": 0.98,
    "MaxDistPort": 29.28,
    "Log10MinDistPort": -1.86
  },
  "neighborhood_type": "llm",
  "num_samples": 2000
}
```

**Parameters:**

- `vessel_event` (VesselEvent): Vessel trajectory instance to explain
  - `SpeedMinimum`: Minimum speed (float, >= 0)
  - `SpeedQ1`: First quartile speed (float, >= SpeedMinimum)
  - `SpeedMedian`: Median speed (float, >= SpeedQ1)
  - `SpeedQ3`: Third quartile speed (float, >= SpeedMedian)
  - `Log10Curvature`: Log10 of trajectory curvature (float)
  - `DistStartTrendAngle`: Angle from start to trend (float)
  - `Log10DistStartTrendDevAmplitude`: Log10 of deviation amplitude (float)
  - `MaxDistPort`: Maximum distance from port in km (float)
  - `Log10MinDistPort`: Log10 of minimum distance from port (float)
- `neighborhood_type` (string, optional): Type of neighborhood generator
  - Options: `"train"`, `"random"`, `"custom"`, `"genetic"`, `"custom_genetic"`, `"llm"`
  - Default: `"genetic"`
- `num_samples` (integer, optional): Number of synthetic instances
  - Default: 2000

**Response:**

```json
{
  "prediction": "3",
  "probability": 0.92,
  "rule": {
    "premises": [
      {"attr": "SpeedMedian", "op": ">", "val": 10.5},
      {"attr": "Log10Curvature", "op": "<=", "val": 0.5}
    ],
    "consequence": {"attr": "class N", "val": "3"}
  },
  "counterfactuals": [...],
  "visualization": {...},
  "neighborhood_stats": {
    "size": 2000,
    "type": "llm"
  }
}
```

**Vessel Classes:**

- `"1"`: Straight trajectory
- `"2"`: Curved trajectory
- `"3"`: Trawling pattern
- `"4"`: Port connected
- `"5"`: Near port
- `"6"`: Anchored

#### POST /api/vessels/explainBatch

Generate explanations for multiple vessel instances.

Similar to COVID batch endpoint but with vessel events.

---

## Neighborhood Generation Types

### Train
Uses nearest neighbors from training data. Fast but limited diversity.

### Random
Uniformly random sampling from feature space. Simple baseline.

### Custom
Feature importance-based generation using decision tree guidance. Respects domain constraints.

### Genetic
Evolutionary algorithm-based generation. Good diversity and quality.

### Custom Genetic
Combines genetic algorithm with feature importance. Best for complex patterns.

### GPT (COVID only)
LLM-inspired generator using temporal transition probabilities.

### LLM (Vessels only)
Constraint-based generator with comprehensive domain knowledge enforcement.

---

## Error Handling

All endpoints return errors in the following format:

```json
{
  "detail": "Error description",
  "error_type": "ValidationError",
  "errors": [...]
}
```

**Common Errors:**

- **Validation Error**: Invalid input format or values
- **Model Error**: Issue with ML model prediction
- **Generation Error**: Failure in neighborhood generation
- **Explanation Error**: LORE explanation generation failed

---

## Rate Limiting

Currently no rate limiting is implemented. For production deployment, consider:
- Rate limiting per IP address
- Request queue management
- Timeout configuration for long-running requests

---

## CORS Configuration

The API is configured to accept requests from any origin (`*`). For production:
- Restrict to specific frontend domains
- Configure appropriate CORS headers
- Implement authentication/authorization

---

## Data Formats

### COVID-19 Severity Levels

- `c0` or `NONE`: No data / baseline
- `c1`: Low severity
- `c2`: Moderate severity
- `c3`: High severity
- `c4`: Very high severity

### Mobility Levels

- `m0` or `NONE`: No data / baseline
- `m1`: Low mobility
- `m2`: Moderate mobility
- `m3`: High mobility
- `m4`: Very high mobility

### Vessel Features

All speed features are in km/h:
- Speed quartiles must maintain ordering: min ≤ Q1 ≤ median ≤ Q3

Log-transformed features use base 10 logarithm.

---

## Performance Considerations

**Typical Response Times:**

- COVID-19 explanation: 2-5 seconds
- Vessel explanation: 3-7 seconds
- Batch processing: 2-5 seconds per instance

**Factors Affecting Performance:**

- `num_samples`: More samples = slower but potentially better quality
- `neighborhood_type`: Genetic/custom are slower than random/train
- Hardware: CPU cores and RAM affect parallel processing

**Optimization Tips:**

- Use cached models (automatically handled)
- Start with smaller `num_samples` for testing
- Use `train` neighborhood type for fastest results
- Consider batch endpoint for multiple instances

---

## Example Usage

### Python Client Example

```python
import requests

# COVID-19 explanation
covid_event = {
    "Week6_Covid": "c2",
    "Week5_Covid": "c2",
    "Week4_Covid": "c1",
    "Week3_Covid": "c1",
    "Week2_Covid": "c2",
    "Week6_Mobility": "m2",
    "Week5_Mobility": "m2",
    "Week4_Mobility": "m1",
    "Week3_Mobility": "m1",
    "Week2_Mobility": "m2",
    "Week1_Mobility": "m1",
    "Days_passed": 35
}

response = requests.post(
    "http://localhost:8000/api/covid/explain",
    json={
        "event": covid_event,
        "neighborhood_type": "genetic",
        "num_samples": 2000
    }
)

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Rule: {result['rule']}")
```

### JavaScript Client Example

```javascript
// Vessel movement explanation
const vesselEvent = {
  SpeedMinimum: 0.03,
  SpeedQ1: 15.51,
  SpeedMedian: 15.91,
  SpeedQ3: 16.52,
  Log10Curvature: 0.004,
  DistStartTrendAngle: 0.28,
  Log10DistStartTrendDevAmplitude: 0.98,
  MaxDistPort: 29.28,
  Log10MinDistPort: -1.86
};

fetch('http://localhost:8000/api/vessels/explain', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    vessel_event: vesselEvent,
    neighborhood_type: 'llm',
    num_samples: 2000
  })
})
.then(response => response.json())
.then(result => {
  console.log('Prediction:', result.prediction);
  console.log('Rule:', result.rule);
});
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/api/covid/explain" \
  -H "Content-Type: application/json" \
  -d '{
    "event": {
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
      "Days_passed": 42
    },
    "neighborhood_type": "genetic"
  }'
```

---

## Troubleshooting

### Common Issues

**Problem**: Model file not found error
**Solution**: Ensure models are trained or downloaded to the `models/` directory

**Problem**: UMAP visualization not appearing
**Solution**: Check that the response includes the `visualization` field

**Problem**: Slow response times
**Solution**: Reduce `num_samples` or use faster neighborhood types

**Problem**: Invalid feature values
**Solution**: Check that speed quartiles maintain ordering and log values are valid

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will show detailed information about:
- Model loading
- Neighborhood generation progress
- Rule extraction process
- Performance metrics

---

## Support

For issues and questions:
- GitHub Issues: https://github.com/Elecapp/Rules_personal_generator/issues
- Check logs for detailed error messages
- Review the interactive API docs at `/api/docs`
