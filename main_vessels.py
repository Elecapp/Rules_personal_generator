"""
Vessel Movement Classification and Explanation Module

This module implements machine learning model training and explanation generation
for vessel movement pattern classification. It uses Random Forest classification
with various neighborhood generation strategies including custom constraint-based
and LLM-inspired generators.

Features:
    - Data loading and preprocessing for vessel trajectory features
    - Random Forest classifier with standard scaling
    - Custom neighborhood generators respecting domain constraints
    - Decision tree-based feature importance analysis
    - LLM-inspired generator with realistic constraint enforcement
    - UMAP dimensionality reduction for visualization

Classes:
    GenerateDecisionTrees: Creates binary decision trees for each target class
    VesselsGenerator: Custom neighborhood generator using feature importance
    GeneticVesselsGenerator: Genetic algorithm-based neighborhood generator
    VesselsLLMGenerator: LLM-inspired generator with constraint enforcement

Functions:
    load_data_from_csv: Load and preprocess vessel movement dataset
    create_and_train_model: Build and train Random Forest classifier
    generate_neighborhoods: Generate synthetic neighborhoods for explanation
    neighborhood_type_to_generators: Map neighborhood types to generator classes
"""

import time

import pandas as pd
import numpy as np

import os

from deap import creator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_selector as selector

import umap
import umap.umap_ as umap

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline


import random
import math

import vessels_utils
from lore_sa.dataset import TabularDataset, Dataset
from lore_sa.bbox import sklearn_classifier_bbox, AbstractBBox
from lore_sa.neighgen import RandomGenerator
from lore_sa.neighgen import GeneticGenerator
from lore_sa.neighgen.neighborhood_generator import NeighborhoodGenerator
from lore_sa.surrogate import DecisionTreeSurrogate
from lore_sa.encoder_decoder import EncDec, ColumnTransformerEnc
from lore_sa.util import neuclidean
from lore_sa.lore import TabularRandomGeneratorLore, Lore

from sklearn.metrics import pairwise_distances, euclidean_distances

import altair as alt
alt.data_transformers.enable('default', max_rows=None)



class GenerateDecisionTrees:
    """
    Generate binary decision tree classifiers for each target class.
    
    This class creates one-vs-all decision tree classifiers for multi-class
    classification problems. Each tree is trained to distinguish one class
    from all others, enabling feature importance analysis per class.
    
    Attributes:
        test_size: Fraction of data to use for testing (default 0.3)
        random_state: Random seed for reproducibility (default 42)
        classifiers: Dictionary mapping class labels to trained decision trees
    """
    
    def __init__(self, test_size=0.3, random_state=42):
        """
        Initialize the decision tree generator.
        
        Args:
            test_size: Proportion of dataset for testing (default 0.3)
            random_state: Random seed for reproducibility (default 42)
        """
        self.random_state = random_state
        self.test_size = test_size
        self.classifiers = {}

    def decision_trees(self, X_feat, y):
        """
        Train binary decision trees for each class.
        
        Creates one decision tree per unique class label using a one-vs-all
        approach. Each tree distinguishes instances of one class from all others.
        
        Args:
            X_feat: Feature matrix
            y: Target labels
            
        Returns:
            Dictionary mapping class labels to trained DecisionTreeClassifier objects
        """
        self.classifiers = {}
        for target_class in np.unique(y):
            binary_y = (y == target_class).astype(int)
            X_train, X_test, y_train, y_test = train_test_split(X_feat, binary_y, random_state=self.random_state, test_size=self.test_size)
            clf = DecisionTreeClassifier(random_state=self.random_state)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            # print(f"Classification Report for class {target_class}:\n")
            # print(classification_report(y_test, y_pred))
            self.classifiers[target_class] = clf
        return self.classifiers



class VesselsGenerator(NeighborhoodGenerator):
    """
    Custom neighborhood generator for vessel movement data.
    
    This generator creates synthetic instances by identifying features that
    influence the prediction (using decision tree paths) and perturbing those
    features while respecting domain constraints (e.g., speed quartile ordering).
    
    The generator:
    1. Uses decision trees to identify influential features for the predicted class
    2. Adds Gaussian noise to features based on their IQR
    3. Enforces speed quartile constraints (min <= Q1 <= median <= Q3)
    4. Applies special handling for log-transformed features
    
    Attributes:
        bbox: Black-box classifier to explain
        dataset: Dataset object with feature information
        encoder: Encoder/decoder for data transformation
        X_feat: Training feature matrix (DataFrame) with column names
        classifiers: Dictionary of decision trees for each class
        prob_of_mutation: Probability of mutating non-influential features
        ocr: Outlier check ratio
        neighborhood: Generated neighborhood instances
        preprocess: Preprocessing pipeline from the black-box model
    """
    
    def __init__(self, bbox: AbstractBBox, dataset: Dataset, encoder: EncDec, X_feat, classifiers: dict, prob_of_mutation: float, ocr=0.1):
        """
        Initialize the vessels neighborhood generator.
        
        Args:
            bbox: Black-box classifier model
            dataset: Dataset object with feature metadata
            encoder: Encoder/decoder for data transformation
            X_feat: Training features as DataFrame (needed for column names)
            classifiers: Dictionary of decision trees per class
            prob_of_mutation: Probability of mutating non-influential features
            ocr: Outlier check ratio for filtering (default 0.1)
        """
        super().__init__(bbox, dataset, encoder, ocr)
        self.neighborhood = None
        self.preprocess = bbox.bbox.named_steps.get('columntransformer')
        self.gen_data = None
        self.X_feat = X_feat
        self.classifiers = classifiers
        self.prob_of_mutation = prob_of_mutation


    def generate(self, x, num_instances: int = 10000, descriptor: dict = None, encoder=None, list=None):
        """
        Generate a neighborhood of synthetic vessel movement instances.
        
        This method creates new instances by:
        1. Predicting the class for the input instance
        2. Finding influential features from the decision tree path
        3. Perturbing features with Gaussian noise scaled by IQR
        4. Enforcing speed quartile ordering constraints
        5. Applying special transformation for log features
        
        Args:
            x: Instance to generate neighborhood around
            num_instances: Number of synthetic instances to generate (default 10000)
            descriptor: Dictionary with feature metadata (min, max, quartiles)
            encoder: Encoder/decoder for transformations
            list: Not used, for interface compatibility
            
        Returns:
            Array of generated neighborhood instances
        """
        perturbed_list = []
        perturbed_list.append(x.copy())

        for _ in range(num_instances):
            perturbed_x = self.perturbate(x)
            perturbed_list.append(perturbed_x)
        self.neighborhood = self.encoder.encode(perturbed_list)
        return self.neighborhood

    def perturbate(self, instance):
        """
        Perturb a single vessel instance using feature importance.
        
        This method:
        1. Predicts the class and retrieves corresponding decision tree
        2. Extracts decision path to identify influential features
        3. Adds Gaussian noise (scaled by IQR) to all features
        4. Enforces speed quartile constraints (min <= Q1 <= median <= Q3)
        5. Mutates influential features and randomly selected non-influential features
        
        Args:
            instance: Instance array to perturb
            
        Returns:
            Perturbed instance array
        """
        descriptor = self.dataset.descriptor
        class_label = self.bbox.predict([instance])[0]
        chosen_dt = self.classifiers[class_label]

        # Get the decision path and leaf node for the instance
        decision_path = chosen_dt.decision_path(instance.reshape(1,-1))
        leaf_id = chosen_dt.apply(instance.reshape(1,-1))[0]

        feature = chosen_dt.tree_.feature
        threshold = chosen_dt.tree_.threshold
        # Extract influencing features based on the decision path
        node_indices = decision_path.indices  # Nodes visited along the path
        influencing_features = []

        for node in node_indices:
            if feature[node] != -2:  # -2 indicates a leaf node
                influencing_features.append(
                    (self.X_feat.columns[feature[node]], threshold[node], instance[feature[node]], instance[feature[node]] < threshold[node], node)
                )
        # extract the feature indices
        feature_indices = [f[4] for f in influencing_features]

        # make a deep copy of x in the variable perturbed_arr
        attributes = vessels_utils.vessels_features

        perturbed_arr = instance.copy()
        for i, v in enumerate(perturbed_arr):
            feat_name = attributes[i]
            distribution = descriptor['numeric'][feat_name]
            iqr = distribution['q3'] - distribution['q1']
            noise = np.random.normal(0, 0.2 * iqr)
            perturbed_arr[i] = np.clip(v + noise, distribution['min'], distribution['max'])

        # impose constraint of the first features
        perturbed_arr[1] = np.clip(perturbed_arr[1], perturbed_arr[0], perturbed_arr[1])
        perturbed_arr[2] = np.clip(perturbed_arr[2], perturbed_arr[1], perturbed_arr[2])
        perturbed_arr[3] = np.clip(perturbed_arr[3], perturbed_arr[2], perturbed_arr[3])


        # perturbed_arr[0] = random.uniform(0, 17.93)
        # perturbed_arr[1] = random.uniform(perturbed_arr[0], 20.12)  # always greater than minspeed
        # perturbed_arr[2] = random.uniform(perturbed_arr[1], 20.75)  # always greater than speedQ1
        # perturbed_arr[3] = random.uniform(perturbed_arr[2], 21.65)  # always greater than speedMedian
        # perturbed_arr[4] = random.uniform(0, 2.24)
        # perturbed_arr[5] = random.uniform(-0.24, 0.36)
        # perturbed_arr[6] = random.uniform(-2.80, 1.77)
        # perturbed_arr[7] = random.uniform(0.12, 282.26)
        perturbed_arr[8] = math.log10(random.uniform(math.exp(-3.05), perturbed_arr[7]))  # inverse log transform, identify value less than max dist, then log transform again

        mask_indices = [perturbed_arr[i] if (i in feature_indices or random.random() < self.prob_of_mutation) else instance[i] for i, v in enumerate(instance)] # change importance feature and some of the feature randomly based on prob
        mask_indices_arr = np.array(mask_indices, dtype=float)

        return mask_indices_arr

class VesselsLLMGenerator(NeighborhoodGenerator):
    """
    LLM-inspired neighborhood generator with comprehensive constraint enforcement.
    
    This advanced generator creates realistic synthetic vessel instances by:
    1. Applying controlled perturbations to features
    2. Enforcing domain-specific constraints (speed ordering, value ranges)
    3. Respecting physical relationships between features
    4. Using proper transformations for log-scale features
    
    The generator is "LLM-inspired" in that it follows learned patterns and rules
    to ensure generated data is realistic and respects domain knowledge.
    
    Key constraints enforced:
    - Speed quartile ordering: min <= Q1 <= median <= Q3 <= max
    - Positive curvature values (in log space)
    - Valid angle and distance ranges
    - Interaction constraints (e.g., high speed + high curvature relationships)
    
    Attributes:
        bbox: Black-box classifier to explain
        dataset: Dataset object with feature information
        encoder: Encoder/decoder for data transformation
        perturbation_scale: Scale factor for perturbations (default 0.05)
        feature_names: List of vessel feature names in order
        constraints: Dictionary defining valid ranges and relationships
        log_features: List of features in log10 scale
        neighborhood: Generated neighborhood instances
    """
    
    def __init__(self, bbox: AbstractBBox, dataset: Dataset, encoder: EncDec, perturbation_scale=0.05):
        """
        Initialize the LLM-inspired vessel generator.
        
        Args:
            bbox: Black-box classifier model
            dataset: Dataset object with feature metadata
            encoder: Encoder/decoder for data transformation
            perturbation_scale: Scale of perturbations to apply (default 0.05)
        """
        super().__init__(bbox, dataset, encoder, ocr=0.1)
        self.neighborhood = None
        self.perturbation_scale = perturbation_scale

        self.feature_names = [
            "SpeedMinimum",  #0
            "SpeedQ1",  #1
            "SpeedMedian",  #2
            "SpeedQ3",  #3
            "Log10Curvature",   #4
            "DistStartTrendAngle",  #5
            "Log10DistStartTrendDevAmplitude",  #6
            "MaxDistPort",  #7
            "Log10MinDistPort"  #8
        ]

        # Define feature constraints
        self.constraints = {
            "SpeedQ3_max": 22.0,
            "DistStartTrendAngle_range": [-0.24, 0.36],
            "Log10Curvature_range": [0.0, 2.25], # Curvature cannot be negative
            "Log10DistStartTrendDevAmplitude_range": [-2.8, 1.8],
            "Log10MinDistPort_min": -3.05,
            # Thresholds for "high speed" and "curvy movement" for the interaction constraint
            "high_speed_threshold": 10.0,      # SpeedMedian above this is considered high
            "curvy_movement_threshold": 0.5    # Log10Curvature above this is considered curvy
        }

        # Define which features are logarithmic for precision formatting
        self.log_features = [
            "Log10Curvature",
            "Log10DistStartTrendDevAmplitude",
            "Log10MinDistPort"
        ]

    def _clamp(self, value, min_val, max_val):
        """
        Clamp a value within a specified range.
        
        Args:
            value: Value to clamp
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Clamped value within [min_val, max_val]
        """
        return max(min_val, min(value, max_val))

    def _apply_constraints(self, instance):
        """
        Apply and enforce all domain-specific constraints on a vessel instance.
        
        This method ensures the generated instance respects:
        - Speed quartile ordering (min <= Q1 <= median <= Q3)
        - Valid ranges for all features
        - Physical constraints (non-negative speeds, valid angles)
        - Distance constraints (min dist <= max dist)
        
        Args:
            instance: Dictionary representing a single vessel data instance
        
        Returns:
            Dictionary with enforced constraints
        """
        # Ensure speed quartile relationships
        instance["SpeedQ1"] = self._clamp(instance["SpeedQ1"], instance["SpeedMinimum"], float('inf'))
        instance["SpeedMedian"] = self._clamp(instance["SpeedMedian"], instance["SpeedQ1"], float('inf'))
        instance["SpeedQ3"] = self._clamp(instance["SpeedQ3"], instance["SpeedMedian"], self.constraints["SpeedQ3_max"])

        # Enforce SpeedMinimum >= 0
        instance["SpeedMinimum"] = max(0.0, instance["SpeedMinimum"])
        instance["SpeedQ1"] = max(instance["SpeedMinimum"], instance["SpeedQ1"])
        instance["SpeedMedian"] = max(instance["SpeedQ1"], instance["SpeedMedian"])
        instance["SpeedQ3"] = max(instance["SpeedMedian"], instance["SpeedQ3"])

        # High speed cannot be achieved during curvy movements
        if (instance["SpeedMedian"] > self.constraints["high_speed_threshold"] and
            instance["Log10Curvature"] > self.constraints["curvy_movement_threshold"]):
            # Reduce speed or curvature. Prioritize reducing speed slightly
            # and then curvature to bring it into a more realistic state.
            instance["SpeedMedian"] *= random.uniform(0.7, 0.9) # Reduce speed
            instance["SpeedQ3"] = max(instance["SpeedMedian"], instance["SpeedQ3"] * random.uniform(0.8, 0.95))
            instance["SpeedQ1"] = min(instance["SpeedMedian"], instance["SpeedQ1"] * random.uniform(0.8, 0.95))
            instance["SpeedMinimum"] = min(instance["SpeedQ1"], instance["SpeedMinimum"] * random.uniform(0.8, 0.95))

            # If still problematic, increase curvature slightly to be more realistic for low speed
            if instance["SpeedMedian"] < 5: # If speed becomes very low, curvature might increase
                 instance["Log10Curvature"] = self._clamp(instance["Log10Curvature"] * random.uniform(1.0, 1.2),
                                                           self.constraints["Log10Curvature_range"][0],
                                                           self.constraints["Log10Curvature_range"][1])

        # Enforce range constraints
        instance["DistStartTrendAngle"] = self._clamp(
            instance["DistStartTrendAngle"],
            self.constraints["DistStartTrendAngle_range"][0],
            self.constraints["DistStartTrendAngle_range"][1]
        )
        instance["Log10Curvature"] = self._clamp(
            instance["Log10Curvature"],
            self.constraints["Log10Curvature_range"][0],
            self.constraints["Log10Curvature_range"][1]
        )
        instance["Log10DistStartTrendDevAmplitude"] = self._clamp(
            instance["Log10DistStartTrendDevAmplitude"],
            self.constraints["Log10DistStartTrendDevAmplitude_range"][0],
            self.constraints["Log10DistStartTrendDevAmplitude_range"][1]
        )
        instance["Log10MinDistPort"] = self._clamp(
            instance["Log10MinDistPort"],
            self.constraints["Log10MinDistPort_min"],
            float('inf') # No upper bound specified, so infinity
        )
        # MaxDistPort should also be non-negative
        instance["MaxDistPort"] = max(0.0, instance["MaxDistPort"])

        # Ensure Log10Curvature is non-negative
        instance["Log10Curvature"] = max(0.0, instance["Log10Curvature"])

        return instance

    def _perturb_feature(self, value, scale):
        """Applies a random perturbation to a feature value."""
        return value * (1 + random.uniform(-scale, scale))

    def _generate_single_instance(self, original_instance, target_class_tendency):
        """
        Generates a single synthetic instance with a tendency towards a specific class.

        Args:
            original_instance (dict): The base instance to perturb.
            target_class_tendency (int): An integer (1-6) indicating the desired class tendency.

        Returns:
            dict: A new synthetic instance.
        """
        new_instance = original_instance.copy()
        base_perturbation = self.perturbation_scale

        # Apply general perturbation to all features first
        for feature in self.feature_names:
            if feature not in ["MaxDistPort", "Log10MinDistPort"]: # MaxDistPort can have larger swings
                new_instance[feature] = self._perturb_feature(new_instance[feature], base_perturbation)
            else:
                 # MaxDistPort can have larger absolute changes
                 new_instance[feature] += random.uniform(-base_perturbation * 20, base_perturbation * 20)


        # Apply targeted perturbations based on class tendency
        if target_class_tendency == 1: # Straight movement
            new_instance["SpeedMinimum"] = self._perturb_feature(new_instance["SpeedMinimum"], base_perturbation * 0.5)
            new_instance["SpeedQ1"] = self._perturb_feature(new_instance["SpeedQ1"], base_perturbation * 0.5)
            new_instance["SpeedMedian"] = self._perturb_feature(new_instance["SpeedMedian"], base_perturbation * 0.5)
            new_instance["SpeedQ3"] = self._perturb_feature(new_instance["SpeedQ3"], base_perturbation * 0.5)
            # Aim for high and consistent speed
            avg_speed = (new_instance["SpeedMinimum"] + new_instance["SpeedQ3"]) / 2
            new_instance["SpeedQ3"] = self._clamp(avg_speed + 5, new_instance["SpeedMedian"], self.constraints["SpeedQ3_max"])
            new_instance["SpeedMedian"] = self._clamp(avg_speed + 2, new_instance["SpeedQ1"], new_instance["SpeedQ3"])
            new_instance["SpeedQ1"] = self._clamp(avg_speed - 1, new_instance["SpeedMinimum"], new_instance["SpeedMedian"])
            new_instance["SpeedMinimum"] = self._clamp(avg_speed - 3, 0.0, new_instance["SpeedQ1"])


            new_instance["Log10Curvature"] = random.uniform(0, 0.01) # Very low curvature
            new_instance["Log10DistStartTrendDevAmplitude"] = random.uniform(-1.0, 0.5) # Low deviation
            new_instance["DistStartTrendAngle"] = random.uniform(0.1, 0.3) # Strong positive trend
            new_instance["Log10MinDistPort"] = self._perturb_feature(new_instance["Log10MinDistPort"], base_perturbation * 2) # Can be further from port
            new_instance["MaxDistPort"] = self._perturb_feature(new_instance["MaxDistPort"], base_perturbation * 2) # Can be further from port

        elif target_class_tendency == 2: # Curved movement
            new_instance["SpeedMinimum"] = self._perturb_feature(new_instance["SpeedMinimum"], base_perturbation * 0.8)
            new_instance["SpeedQ1"] = self._perturb_feature(new_instance["SpeedQ1"], base_perturbation * 0.8)
            new_instance["SpeedMedian"] = self._perturb_feature(new_instance["SpeedMedian"], base_perturbation * 0.8)
            new_instance["SpeedQ3"] = self._perturb_feature(new_instance["SpeedQ3"], base_perturbation * 0.8)
            # Moderate to high speed
            avg_speed = (new_instance["SpeedMinimum"] + new_instance["SpeedQ3"]) / 2
            new_instance["SpeedQ3"] = self._clamp(avg_speed + 3, new_instance["SpeedMedian"], self.constraints["SpeedQ3_max"])
            new_instance["SpeedMedian"] = self._clamp(avg_speed + 1, new_instance["SpeedQ1"], new_instance["SpeedQ3"])
            new_instance["SpeedQ1"] = self._clamp(avg_speed - 1, new_instance["SpeedMinimum"], new_instance["SpeedMedian"])
            new_instance["SpeedMinimum"] = self._clamp(avg_speed - 2, 0.0, new_instance["SpeedQ1"])

            new_instance["Log10Curvature"] = random.uniform(0.05, 0.5) # Moderate curvature
            new_instance["Log10DistStartTrendDevAmplitude"] = random.uniform(0.0, 1.0) # Moderate deviation
            new_instance["DistStartTrendAngle"] = random.uniform(-0.1, 0.2) # Can vary
            new_instance["Log10MinDistPort"] = self._perturb_feature(new_instance["Log10MinDistPort"], base_perturbation * 1.5)
            new_instance["MaxDistPort"] = self._perturb_feature(new_instance["MaxDistPort"], base_perturbation * 1.5)

        elif target_class_tendency == 3: # Trawling
            new_instance["SpeedMinimum"] = self._perturb_feature(new_instance["SpeedMinimum"], base_perturbation * 0.5)
            new_instance["SpeedQ1"] = self._perturb_feature(new_instance["SpeedQ1"], base_perturbation * 0.5)
            new_instance["SpeedMedian"] = self._perturb_feature(new_instance["SpeedMedian"], base_perturbation * 0.5)
            new_instance["SpeedQ3"] = self._perturb_feature(new_instance["SpeedQ3"], base_perturbation * 0.5)
            # Low to moderate, consistent speed
            avg_speed = (new_instance["SpeedMinimum"] + new_instance["SpeedQ3"]) / 2
            new_instance["SpeedQ3"] = self._clamp(avg_speed + 1, new_instance["SpeedMedian"], 10.0) # Trawling max speed
            new_instance["SpeedMedian"] = self._clamp(avg_speed + 0.5, new_instance["SpeedQ1"], new_instance["SpeedQ3"])
            new_instance["SpeedQ1"] = self._clamp(avg_speed - 0.2, new_instance["SpeedMinimum"], new_instance["SpeedMedian"])
            new_instance["SpeedMinimum"] = self._clamp(avg_speed - 0.5, 0.0, new_instance["SpeedQ1"])

            new_instance["Log10Curvature"] = random.uniform(0.1, 0.6) # Moderate curvature
            new_instance["Log10DistStartTrendDevAmplitude"] = random.uniform(0.0, 1.0) # Moderate deviation
            new_instance["DistStartTrendAngle"] = random.uniform(-0.05, 0.05) # Small angle, not strong trend
            new_instance["Log10MinDistPort"] = self._perturb_feature(new_instance["Log10MinDistPort"], base_perturbation * 1.5)
            new_instance["MaxDistPort"] = self._perturb_feature(new_instance["MaxDistPort"], base_perturbation * 1.5)

        elif target_class_tendency == 4: # Port-connected
            # Speed can vary, often starts very low then increases
            new_instance["SpeedMinimum"] = random.uniform(0.0, 1.0)
            new_instance["SpeedQ1"] = self._perturb_feature(new_instance["SpeedQ1"], base_perturbation * 1.5)
            new_instance["SpeedMedian"] = self._perturb_feature(new_instance["SpeedMedian"], base_perturbation * 1.5)
            new_instance["SpeedQ3"] = self._perturb_feature(new_instance["SpeedQ3"], base_perturbation * 1.5)

            new_instance["Log10Curvature"] = random.uniform(0.0, 0.5) # Low to moderate
            new_instance["Log10DistStartTrendDevAmplitude"] = random.uniform(0.5, 1.5) # High zigzagging near port
            new_instance["DistStartTrendAngle"] = random.uniform(-0.1, 0.1) # Often close to 0
            new_instance["Log10MinDistPort"] = random.uniform(-2.5, -0.5) # Close to port
            new_instance["MaxDistPort"] = random.uniform(10.0, 30.0) # Moderate max distance

        elif target_class_tendency == 5: # Near port
            # Very low speed
            new_instance["SpeedMinimum"] = random.uniform(0.0, 0.1)
            new_instance["SpeedQ1"] = random.uniform(0.01, 0.2)
            new_instance["SpeedMedian"] = random.uniform(0.05, 0.3)
            new_instance["SpeedQ3"] = random.uniform(0.1, 0.5)

            new_instance["Log10Curvature"] = random.uniform(0.3, 1.0) # Moderate to high curvature (manoeuvring)
            new_instance["Log10DistStartTrendDevAmplitude"] = random.uniform(0.5, 1.8) # High deviation (erratic)
            new_instance["DistStartTrendAngle"] = random.uniform(-0.05, 0.05) # Very close to 0
            new_instance["Log10MinDistPort"] = random.uniform(-1.0, -0.3) # Very close to port
            new_instance["MaxDistPort"] = random.uniform(0.1, 5.0) # Small max distance

        elif target_class_tendency == 6: # Anchored
            # Extremely low speed (near zero)
            new_instance["SpeedMinimum"] = random.uniform(0.0, 0.01)
            new_instance["SpeedQ1"] = random.uniform(0.005, 0.05)
            new_instance["SpeedMedian"] = random.uniform(0.01, 0.1)
            new_instance["SpeedQ3"] = random.uniform(0.05, 0.2)

            new_instance["Log10Curvature"] = random.uniform(1.0, 2.2) # High curvature (small movements)
            new_instance["Log10DistStartTrendDevAmplitude"] = random.uniform(-2.8, -1.5) # Very low deviation (in-place)
            new_instance["DistStartTrendAngle"] = random.uniform(-0.01, 0.01) # Near zero
            new_instance["Log10MinDistPort"] = random.uniform(-0.8, -0.3) # Very close to port
            new_instance["MaxDistPort"] = random.uniform(0.0, 1.0) # Very small max distance

        # Ensure speed quartiles remain ordered after targeted adjustments but before final clamp
        new_instance["SpeedQ1"] = max(new_instance["SpeedMinimum"], new_instance["SpeedQ1"])
        new_instance["SpeedMedian"] = max(new_instance["SpeedQ1"], new_instance["SpeedMedian"])
        new_instance["SpeedQ3"] = max(new_instance["SpeedMedian"], new_instance["SpeedQ3"])

        # Apply constraints before final formatting
        new_instance = self._apply_constraints(new_instance)

        # Apply precision formatting
        formatted_instance = {}
        for feature, value in new_instance.items():
            if feature in self.log_features:
                formatted_instance[feature] = round(value, 5)
            else:
                formatted_instance[feature] = round(value, 3)

        return formatted_instance


    def generate(self, original_instance_values, num_synthetic_instances=60, descriptor: dict=None, encoder=None):
        """
        Generates a specified number of synthetic data instances.

        Args:
            original_instance_values (list or dict): The feature values of the
                original instance. If a list, it must be in the order:
                SpeedMinimum, SpeedQ1, SpeedMedian, SpeedQ3, Log10Curvature,
                DistStartTrendAngle, Log10DistStartTrendDevAmplitude,
                MaxDistPort, Log10MinDistPort.
            num_synthetic_instances (int): The total number of synthetic instances to generate.
                                           This will be distributed across the 6 classes.

        Returns:
            list[dict]: A list of generated synthetic data instances,
                        each as a dictionary of feature_name: value.
        """
        if isinstance(original_instance_values, (list, np.ndarray)):
            if len(original_instance_values) != len(self.feature_names):
                raise ValueError(
                    f"Original instance list must have {len(self.feature_names)} features. "
                    f"Got {len(original_instance_values)}."
                )
            original_instance = dict(zip(self.feature_names, original_instance_values))
        elif isinstance(original_instance_values, dict):
            # Validate if all expected features are present
            if not all(feature in original_instance_values for feature in self.feature_names):
                raise ValueError("Original instance dictionary is missing required features.")
            original_instance = original_instance_values
        else:
            raise TypeError("original_instance_values must be a list or a dictionary.")

        synthetic_data = []
        instances_per_class = num_synthetic_instances // 6
        remaining_instances = num_synthetic_instances % 6

        for class_tendency in range(1, 7): # Iterate through each of the 6 classes
            count = instances_per_class + (1 if class_tendency <= remaining_instances else 0)
            for _ in range(count):
                synthetic_instance = self._generate_single_instance(original_instance, class_tendency)
                synthetic_data.append(synthetic_instance)

        # Shuffle the data to mix instances from different "target" classes
        random.shuffle(synthetic_data)
        array_output = np.array([[instance[feature] for feature in self.feature_names]
                             for instance in synthetic_data], dtype=np.float64)

        return array_output


class GeneticVesselsGenerator(GeneticGenerator):

    def __init__(self, bbox: AbstractBBox, dataset: Dataset, encoder: EncDec, X_feat, classifiers: dict, prob_of_mutation: float, ocr=0.1):
        super().__init__(bbox, dataset, encoder, ocr,
                         alpha1=0.5, alpha2=0.5, metric=neuclidean, ngen=30, mutpb=0.2, cxpb=0.5,
                         tournsize=3, halloffame_ratio=0.1, random_seed=None
                         )
        self.vessels_generator = VesselsGenerator(bbox, dataset, encoder, X_feat, classifiers, prob_of_mutation, ocr)

    def mutate(self, toolbox, x):
        """
         This function specializes the mutation operator of the genetic algorithm to explicitly
         use the implementation of the perturbation of the VesselsGenerator class. This guarantees
            that the perturbation is consistent with the perturbation of the VesselsGenerator class.
        """
        z = toolbox.clone(x)
        z1 = self.vessels_generator.perturbate(z)

        for i,v in enumerate(z1):
            z[i] = v

        return z,

    def mate(self, ind1, ind2):
        """Executes a two-point crossover on the input :term:`sequence`
        individuals. The two individuals are modified in place and both keep
        their original length.
        This implementation uses the original implementation of the DEAP library. It adds a special case for the
        one-hot encoding, where the crossover is done taking into account the intervals of values imposed by
        the one-hot encoding.

        This overrides also keeps the blocks of the speed attributes together, to ensure that the relative values
        are maintained.

        :param ind1: The first individual participating in the crossover.
        :param ind2: The second individual participating in the crossover.
        :returns: A tuple of two individuals.

        This function uses the :func:`~random.randint` function from the Python
        base :mod:`random` module.
        """
        if self.encoder.type == 'one-hot':
            intervals = [[0, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]]
            cxInterval1 = random.randint(0, len(intervals) - 1)
            cxInterval2 = random.randint(0, len(intervals) - 1)
            if cxInterval1 > cxInterval2:
                # Swap the two cx intervals
                cxInterval1, cxInterval2 = cxInterval2, cxInterval1

            cxpoint1 = intervals[cxInterval1][0]
            cxpoint2 = intervals[cxInterval2][1]
            ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
                = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
        else:
            size = min(len(ind1), len(ind2))
            cxpoint1 = random.randint(1, size)
            cxpoint2 = random.randint(1, size - 1)
            if cxpoint2 >= cxpoint1:
                cxpoint2 += 1
            else:  # Swap the two cx points
                cxpoint1, cxpoint2 = cxpoint2, cxpoint1

            ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
                = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]

        return ind1, ind2





class BaselineTrainingGenerator(NeighborhoodGenerator):
    """
    This generator serves as a baseline for the generation of rules. Since many of our rules are dependent on the
    synthetic data used for training during the generation, this generator will be used to generate the training data
    in a consistent way. This will allow us to compare the performance of the rules generated by the other generator we
    may want to implement.
    """
    def __init__(self, bbox: AbstractBBox, dataset: Dataset, encoder: EncDec, ocr=0.1, df_train: pd.DataFrame = None):
        super().__init__(bbox, dataset, encoder, ocr)
        self.df_train = df_train

    def generate(self, x, num_instances:int=10000, descriptor: dict=None, encoder=None, list=None):
        distances = pairwise_distances(self.df_train, x.reshape(1, -1), metric='euclidean')
        # distances = self.df_train.apply(lambda row: euclidean_distances(x_, row), axis=1)
        df_neighb = self.df_train.copy()
        df_neighb['Distance'] = distances
        df_neighb.sort_values('Distance', inplace=True)

        max_rows = min(num_instances, len(df_neighb))

        return self.encoder.encode(df_neighb.iloc[:max_rows, :-1].values)




        '''
        in the class there are already the stored decision tree, 
        I retrieve the decision tree
        '''
        '''
        New alg description:
        - check the class label of x
        - find the corresponding decision tree for the class label
        - find the corresponding branch in the DT for the instance
        - generate the perturbations by altering feature values to 
          fit into other branches of the DT (threshold of same class vs different class) and ensure
          that the soft rules are maintained)
        - Keep the feature that do not influence the class decision as constants.
        
        ---------------
        def create_and_train_model(df):
            return bbox
        #create class with the precalc decision tree
        def decision_trees(x)
            classifiers = {}
            for target_class in np.unique(y):
                binary_y = (y == target_class).astype(int)
                X_train, X_test, y_train, y_test = train_test_split(X_feat, binary_y, test_size=0.3, random_state=42)
                clf = DecisionTreeClassifier(random_state=42)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                #print(f"Classification Report for class {target_class}:\n")
                #print(classification_report(y_test, y_pred))
                classifiers[target_class] = clf
            return classifiers
        x = [instance]
        x_class= bbox.predict(x) #check class label for x
        chosen_dt = classifiers[x_class] # find the corresponding DT
        check_x = x.reshape(1, -1)
        # Step 1: Get the decision path
        node_indicator = clf.decision_path(check_x)

        # Step 2: Find the leaf node for the instance
        leaf_id = clf.apply(check_x)[0]

        # Step 3: Extract conditions along the path
        feature = clf.tree_.feature
        threshold = clf.tree_.threshold
        #generate the perturbations by altering feature values to fit into other branches of the DT (threshold of same 
        class vs different class)
        #Keep the feature that do not influence the class decision as constants
        candidate_neigh = [x.perturb] # perturb everything but the irrelevant feature, leverage on the def generate
        check what is inside node indicator.
        for n in candidate_neigh: #ensure that the soft rules are maintained
            check if it fits the soft rules
            if not:
                modify accordingly
            proper_neigh   
        '''

        return self.neighborhood

def load_data_from_csv():
    """
    Load and preprocess the vessel movement dataset from CSV file.
    
    This function:
    1. Defines human-readable class names for vessel movement patterns
    2. Loads the dataset from datasets/final_df_addedfeat.csv
    3. Selects the vessel features defined in vessels_utils
    4. Ensures class labels are string type for consistency
    
    Class mapping:
        '1': Straight trajectory
        '2': Curved trajectory
        '3': Trawling pattern
        '4': Port connected
        '5': Near port
        '6': Anchored
    
    Returns:
        DataFrame: Vessel movement dataset with selected features and class labels
    """
    class_names = {
        '1': 'Straight',
        '2': 'Curved',
        '3': 'Trawling',
        '4': 'Port connected',
        '5': 'Near port',
        '6': 'Anchored',
    }


    df = pd.read_csv("datasets/final_df_addedfeat.csv")
    features = vessels_utils.vessels_features  + ['class N']
    df['class N'] = df['class N'].astype(str)

    return df[features]

def create_and_train_model(df):
    """
    Create and train a Random Forest classifier for vessel movement classification.
    
    This function:
    1. Separates features from class labels
    2. Creates preprocessing pipeline:
       - Ordinal encoding for categorical features (if any)
       - Standard scaling for all 9 numerical vessel features
    3. Trains a Random Forest classifier with 100 estimators
    4. Evaluates model performance on test set
    5. Returns the trained model and training data
    
    Args:
        df: Preprocessed DataFrame with vessel features and class labels
        
    Returns:
        tuple: (trained_model, training_features, training_labels)
            - trained_model: Scikit-learn pipeline with preprocessor and classifier
            - training_features: DataFrame of training set features
            - training_labels: Array of training set labels
    """

    label = 'class N'
    features = df.columns[:-1]

    X_feat = df[features]
    y = df[label].values.astype('str')

    numerical_columns_selector = selector(dtype_exclude=object)
    categorical_columns_selector = selector(dtype_include=object)

    numerical_columns = numerical_columns_selector(X_feat)
    categorical_columns = categorical_columns_selector(X_feat)

    enc = OrdinalEncoder()
    numerical_preprocessor = StandardScaler()

    preprocessor = ColumnTransformer(
        [
            ("ordinal-encoder", enc, categorical_columns),
            ("standard_scaler", numerical_preprocessor, list(range(0,9))),
        ]
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    model = make_pipeline(preprocessor, clf)

    data_train, data_test, target_train, target_test = train_test_split(
        X_feat, y, random_state=0
    )

    _ = model.fit(data_train, target_train)

    print('Model score: ', model.score(data_test, target_test))
    return model, data_train, target_train

def neighborhood_type_to_generators(neighborhood_types: [str], bbox, ds, encoder, data_train, target_train):
    """
    Create neighborhood generators based on requested types.
    
    This factory function instantiates the appropriate neighborhood generator
    classes based on the requested types. It handles initialization of generators
    that require additional setup (e.g., decision tree classifiers for custom generators).
    
    Available generator types:
    - 'random': Uniform random sampling from feature space
    - 'custom': Feature importance-based with decision tree guidance
    - 'genetic': Genetic algorithm-based generation
    - 'custom_genetic': Genetic algorithm with feature importance
    - 'llm': LLM-inspired with constraint enforcement
    - 'train': Uses training data as neighborhood (nearest neighbors)
    
    Args:
        neighborhood_types: List of generator type strings
        bbox: Black-box classifier to explain
        ds: Dataset object with feature metadata
        encoder: Encoder/decoder for data transformation
        data_train: Training feature data (DataFrame)
        target_train: Training labels
        
    Returns:
        List of tuples: [(generator_name, generator_instance), ...]
    """
    generators = []
    if 'random' in neighborhood_types:
        random_n_generator = RandomGenerator(bbox, ds, encoder)
        generators.append(('random', random_n_generator))
    if 'custom' in neighborhood_types:
        classifiers_generator = GenerateDecisionTrees()
        classifiers = classifiers_generator.decision_trees(data_train, target_train)
        custom_generator = VesselsGenerator(bbox, ds, encoder, data_train, classifiers, 0.05)
        generators.append(('custom', custom_generator))
    if 'genetic' in neighborhood_types:
        genetic_n_generator = GeneticGenerator(bbox, ds, encoder)
        generators.append(('genetic', genetic_n_generator))
    if 'custom_genetic' in neighborhood_types:
        classifiers_generator = GenerateDecisionTrees()
        classifiers = classifiers_generator.decision_trees(data_train, target_train)
        custom_gen_generator = GeneticVesselsGenerator(bbox, ds, encoder, data_train, classifiers, 0.05)
        generators.append(('custom_genetic', custom_gen_generator))
    if 'baseline' in neighborhood_types:
        baseline_generator = BaselineTrainingGenerator(bbox, ds, encoder, df_train=data_train)
        generators.append(('baseline', baseline_generator))
    if 'llm' in neighborhood_types:
        llm_generator = VesselsLLMGenerator(bbox, ds, encoder, perturbation_scale=0.05)
        generators.append(('llm', llm_generator))

    return generators

def generate_neighborhoods(x, model, data, X_feat, y, num_instances=100, neighborhood_types=['train', 'random']):
    """
    Generate neighborhoods for a given instance x and model. The generators are selected from the neighborhood_types.
    Each label corresponds to one of the generators avaialble for this case study. See the function `neighborhood_type_to_generators`
    for more details.

    :param x: The instance for which the neighborhoods are generated
    :param model: The model used for the prediction
    :param data: The data used for the statistics of the training data. It is used to set up the generators metadata.
    :param X_feat: The features of the data to be used as training set. This is used when the 'train' neighborhood is selected.
    :param y: The target labels of the data to be used as training set. This is used when the 'train' neighborhood is selected.
    :param num_instances: The number of instances to generate for each neighborhood generation
    :param neighborhood_types: The types of neighborhoods to generate. The available types are: 'train', 'random', 'custom', 'genetic', 'custom_genetic', 'baseline'

    :return: A list of tuples with the neighborhood type and the generated neighborhoods. The tuple is in the form (neighborhood_type, neighborhoods)
    """
    # Check if the neighborhoods already exist
    ds = TabularDataset(data=data, class_name='class N',categorial_columns=['class N'])
    encoder = ColumnTransformerEnc(ds.descriptor)
    bbox = sklearn_classifier_bbox.sklearnBBox(model)
    result = ()

    if 'train' in neighborhood_types:
        result = result + (('train', X_feat.values),)

    generators = neighborhood_type_to_generators(neighborhood_types, bbox, ds, encoder, X_feat, y)
    for (n, g) in generators:
        gen_neighb = g.generate(x, num_instances, ds.descriptor, encoder)
        result = result + ((n, gen_neighb), )
    return result

def generate_neighborhood_statistics(x, model, data, X_feat, y, num_instances=100,
                                     num_repeation=10, neighborhood_types=['train', 'random'], an_array=None):

    ds = TabularDataset(data=data, class_name='class N',categorial_columns=['class N'])
    encoder = ColumnTransformerEnc(ds.descriptor)
    bbox = sklearn_classifier_bbox.sklearnBBox(model)
    result = ()
    compl_times = ()

    generators = neighborhood_type_to_generators(neighborhood_types, bbox, ds, encoder, X_feat, y)
    for (n, g) in generators:
        print(f'Generator: {n}')
        global_mins_training = []
        global_mins_instance = []
        global_times = []
        for i in range(num_repeation):
            if i % 2 == 0:
                print(f"Repetition {i}")
            start = time.time()
            gen_neighb = g.generate(x.copy(), num_instances, ds.descriptor, encoder)
            gen_neighb = gen_neighb[:num_instances]
            end = time.time()
            dists_training = compute_distance(gen_neighb, an_array)
            dists_instance = compute_distance(gen_neighb, x.reshape(1, -1))
            global_mins_training.append(dists_training)
            global_mins_instance.append(dists_instance)
            global_times.append(end - start)

        np_mean_training = np.mean(np.array(global_mins_training), axis=0)
        np_mean_instance = np.mean(np.array(global_mins_instance), axis=0)
        result = result + ((n, 'training', np_mean_training), )
        result = result + ((n, 'instance', np_mean_instance), )
        compl_times = compl_times + ((n, global_times),)
    return result, compl_times



def compute_distance(X1, X2, metric:str='euclidean'):
    dists = pairwise_distances(X1, X2, metric=metric)
    dists = np.min(dists, axis=1)
    return dists

def measure_distance(neighborhoods, an_array):# an_array can be a point or X_feat
    """
    Measure the distance of 'an_array' to the neighborhoods. The neighborhoods are in the form of a list of tuples
    where the first element is the neighborhood type and the second element is the neighborhood data.

    :param neighborhoods: The neighborhoods to measure the distance to
    :param an_array: The array to measure the distance to the neighborhoods

    :return: A list of tuples with the neighborhood type and the distances to the neighborhoods. The tuple is in the form (neighborhood_type, distances)
    """
    distances = []
    for n, neigh in neighborhoods:
        dists = compute_distance(neigh, an_array)
        distances.append((n, dists))

    return distances



def new_lore(data, bb):
    features = ['SpeedMinimum', 'SpeedQ1', 'SpeedMedian', 'SpeedQ3', 'DistanceStartShapeCurvature',
                'DistStartTrendAngle', 'DistStartTrendDevAmplitude', 'MaxDistPort', 'MinDistPort', 'class N']
    data = data.loc[:, features]
    #instance = data.values[5, : -1]
    #prediction = model.predict([instance])
    instance = data.iloc[9, :-1].values
    ds = TabularDataset(data=data, class_name='class N', categorial_columns=['class N'])
    bbox = sklearn_classifier_bbox.sklearnBBox(bb)
    print(bb.predict([instance]))

    print("instance is:", instance)
    x = instance
    classifiers_generator = GenerateDecisionTrees()
    classifiers = classifiers_generator.decision_trees(X_feat, y)

    #print('model prediction is', model.predict([x]))
    #lore = TabularRandomGeneratorLore(bbox, x)![](../../Desktop/Schermata 2024-12-23 alle 15.10.01.png)
    encoder = ColumnTransformerEnc(ds.descriptor)
    surrogate = DecisionTreeSurrogate()

    generator = VesselsGenerator(bbox, ds, encoder, classifiers, 0.05)


    proba_lore = Lore(bbox, ds, encoder, generator, surrogate)
    print('Prediction:', bb.predict([x]))

    rule = proba_lore.explain(x)
    print('Rule:', rule['rule'])
    print(rule)
    print('----- ')
    print('----- counterfactuals ')
    for cr in rule['counterfactuals']:
        print(cr)
        print('-----')

def plot_boxplot(df, df_times, basefilename):
    #domain_ = ['Random', 'Custom', 'Genetic', 'GPT']
    #range_ = ['#102ce0', '#fa7907', '#027037', '#FF5733']

    text_times = alt.Chart(df_times).mark_text(
        align='left',
        baseline='middle',
        dx=7
    ).encode(
        text='mean(Time):Q',
        y=alt.Y('Neighborhood:N'),
    ).configure_view(
        stroke=None
    )
    box_plot_times = text_times

    box_plot_times.save(f'plot/instance_vs_neigh/{basefilename}_times.pdf')

    box_plot_training = alt.Chart(df[df['Reference']=='training']).mark_boxplot().encode(
        x=alt.X("Distance:Q"),
        y=alt.Y("Neighborhood:N"),
        color=alt.Color("Neighborhood:N") #scale=alt.Scale(domain=domain_, range=range_)
    ).properties(
        height=200,
        width=200,
        title='Euclidean distance to training data'
    )
    box_plot_training.save(f'plot/instance_vs_neigh/{basefilename}_training.pdf')

    box_plot_instance = alt.Chart(df[df['Reference']=='instance']).mark_boxplot().encode(
        x=alt.X("Distance:Q"),
        y=alt.Y("Neighborhood:N"),
        color=alt.Color("Neighborhood:N") #scale=alt.Scale(domain=domain_, range=range_)
    ).properties(
        height=200,
        width=200,
        title='Euclidean distance to instance'
    )
    box_plot_instance.save(f'plot/instance_vs_neigh/{basefilename}_instance.pdf')

    box_plot_both = box_plot_instance | box_plot_training
    box_plot_both.save(f'plot/instance_vs_neigh/{basefilename}_both.pdf')


if __name__ == '__main__':
    res = load_data_from_csv()
    model, X_feat, y = create_and_train_model(res)
    print(model)

    # id_instance = 192
    # instance = res.iloc[id_instance, :-1].values  # Example instance

    id_instance = 1003
    instance = np.array([0.91, 2.37, 2.49, 2.78, 2.45, -0.01, 5.07, 27.55, 20.04])  # Example instance

    # result_n = generate_neighborhoods(instance, model, res, X_feat, y, num_instances=100, neighborhood_types=['train', 'random',])
    print(f'Instance {id_instance} is: {instance}')

    #new_lore(res, model)
    #instance = res.iloc[9, :-1].values
    #distances_instance = measure_distance(result_n, instance.reshape(1, -1))
    #print(random_distance, custom_distance, genetic_distance)

    #distances_train = measure_distance(result_n, X_feat.values)
    #Example: extracting a column

    num_repeation = 50
    num_instances = 2000
    neighborhood_types = [
        'train',
        'random',
        'custom',
        'genetic',
        'custom_genetic',
        'baseline'
    ]

    basefilename = f'vessels_neighborhoods_x_{id_instance}_{num_instances}_{num_repeation}rep'
    csv = f'{basefilename}.csv'
    if not os.path.exists('%s' % csv):
        dists, compl_times = generate_neighborhood_statistics(instance, model, res, res.loc[:,vessels_utils.vessels_features], res['class N'], num_instances=num_instances,
                                                 num_repeation=num_repeation,
                                                 neighborhood_types=neighborhood_types, an_array=X_feat.values)
        df = pd.DataFrame([], columns=['Neighborhood', 'Reference', 'Distance'])
        df_times = pd.DataFrame([], columns=['Neighborhood', 'Time'])
        for (n, t, d) in dists:
            df = pd.concat([df, pd.DataFrame({'Neighborhood': [n] * len(d), 'Reference': [t] * len(d),'Distance': d})], axis=0)
        for (n, t) in compl_times:
            df_times = pd.concat([df_times, pd.DataFrame({'Neighborhood': [n] * len(t), 'Time': t})], axis=0)
        df.to_csv(csv, index=False)
        df_times.to_csv(f'{basefilename}_times.csv', index=False)
    else:
        df = pd.read_csv(csv)
        df_times = pd.read_csv(f'{basefilename}_times.csv')

    plot_boxplot(df, df_times, basefilename)





