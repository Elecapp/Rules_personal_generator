"""
Vessel Feature Definitions Module

This module defines the standard feature set used for vessel movement classification.
All vessel-related code uses these feature definitions to ensure consistency across
data loading, preprocessing, and model training.

Features:
    SpeedMinimum: Minimum speed observed in trajectory
    SpeedQ1: First quartile (25th percentile) of speed
    SpeedMedian: Median (50th percentile) of speed
    SpeedQ3: Third quartile (75th percentile) of speed
    Log10Curvature: Log10-transformed trajectory curvature measure
    DistStartTrendAngle: Distance from start to trend angle
    Log10DistStartTrendDevAmplitude: Log10-transformed distance deviation amplitude
    MaxDistPort: Maximum distance from nearest port
    Log10MinDistPort: Log10-transformed minimum distance from nearest port

Note: The original features included non-log-transformed versions, but the current
implementation uses log10-transformed versions for better model performance.
"""


# vessels_features = ['SpeedMinimum', 'SpeedQ1', 'SpeedMedian', 'SpeedQ3', 'DistanceStartShapeCurvature',
#             'DistStartTrendAngle', 'DistStartTrendDevAmplitude', 'MaxDistPort', 'MinDistPort']

vessels_features = ['SpeedMinimum', 'SpeedQ1', 'SpeedMedian', 'SpeedQ3', 'Log10Curvature',
            'DistStartTrendAngle', 'Log10DistStartTrendDevAmplitude', 'MaxDistPort', 'Log10MinDistPort']