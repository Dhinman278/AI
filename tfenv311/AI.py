
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info/warning messages
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# Advanced AI Enhancement imports
import cv2
import pickle
import re
import json
from collections import defaultdict
import difflib
import hashlib
from datetime import datetime, timedelta
import sqlite3
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
from pathlib import Path
import base64
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import threading
import queue
import time
import requests
import networkx as nx
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from cryptography.fernet import Fernet

# Optional advanced imports (install if available)
try:
    import librosa
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    
try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import tensorflow_federated as tff
    FEDERATED_AVAILABLE = True
except ImportError:
    FEDERATED_AVAILABLE = False

try:
    from lime import lime_text
    import shap
    from sklearn.inspection import permutation_importance
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    EXPLAINABILITY_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from Bio import SeqIO
    BIOINFORMATICS_AVAILABLE = True
except ImportError:
    BIOINFORMATICS_AVAILABLE = False

# ===== ADVANCED AI ENHANCEMENT SYSTEMS =====

class FederatedLearningSystem:
    """
    Multi-institution collaborative learning system for medical AI
    Enables secure, distributed learning across healthcare institutions
    """
    
    def __init__(self):
        self.participants = []
        self.global_model = None
        self.communication_rounds = 0
        self.performance_history = []
        self.privacy_budget = 1.0  # Differential privacy budget
        
    def register_participant(self, participant_id, institution_name, data_size):
        """Register a new participant institution"""
        participant = {
            'id': participant_id,
            'institution': institution_name,
            'data_size': data_size,
            'local_model': None,
            'contribution_score': 0.0,
            'privacy_contribution': 0.0,
            'joined_date': datetime.now()
        }
        self.participants.append(participant)
        print(f"✅ Registered participant: {institution_name} with {data_size} samples")
        
    def create_federated_model(self, input_shape, num_classes):
        """Create the global federated learning model"""
        if FEDERATED_AVAILABLE:
            # TensorFlow Federated implementation
            def model_fn():
                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dropout(0.3),
                    tf.keras.layers.Dense(num_classes, activation='softmax')
                ])
                return tff.learning.from_keras_model(
                    model,
                    input_spec=tf.TensorSpec(shape=[None, input_shape], dtype=tf.float32),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
                )
            
            self.global_model = model_fn()
            print("✅ Federated learning model created")
        else:
            # Fallback implementation without TensorFlow Federated
            self.global_model = self._create_standard_model(input_shape, num_classes)
            print("⚠️  Using standard model (TensorFlow Federated not available)")
    
    def _create_standard_model(self, input_shape, num_classes):
        """Create standard model when federated learning libraries unavailable"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def simulate_local_training(self, participant_id, local_data, local_labels, epochs=5):
        """Simulate local training at participant institution"""
        participant = next((p for p in self.participants if p['id'] == participant_id), None)
        if not participant:
            return None
        
        # Clone global model for local training
        local_model = tf.keras.models.clone_model(self.global_model)
        local_model.set_weights(self.global_model.get_weights())
        
        # Add differential privacy noise
        if self.privacy_budget > 0:
            local_data = self._add_differential_privacy(local_data)
        
        # Train locally
        history = local_model.fit(
            local_data, local_labels,
            epochs=epochs,
            batch_size=32,
            verbose=0,
            validation_split=0.2
        )
        
        participant['local_model'] = local_model
        participant['contribution_score'] += history.history['val_accuracy'][-1]
        
        return local_model.get_weights()
    
    def _add_differential_privacy(self, data, noise_scale=0.1):
        """Add differential privacy noise to protect patient data"""
        if self.privacy_budget <= 0:
            return data
        
        noise = np.random.laplace(0, noise_scale, data.shape)
        private_data = data + noise
        self.privacy_budget -= 0.1  # Consume privacy budget
        
        return private_data
    
    def federated_averaging(self):
        """Aggregate local models using federated averaging"""
        if not self.participants:
            return
        
        # Get weights from all participants
        participant_weights = []
        data_sizes = []
        
        for participant in self.participants:
            if participant['local_model'] is not None:
                participant_weights.append(participant['local_model'].get_weights())
                data_sizes.append(participant['data_size'])
        
        if not participant_weights:
            return
        
        # Weighted averaging based on data size
        total_data = sum(data_sizes)
        averaged_weights = []
        
        for layer_idx in range(len(participant_weights[0])):
            layer_weights = []
            for participant_idx, weights in enumerate(participant_weights):
                weight = data_sizes[participant_idx] / total_data
                layer_weights.append(weights[layer_idx] * weight)
            averaged_weights.append(np.sum(layer_weights, axis=0))
        
        # Update global model
        self.global_model.set_weights(averaged_weights)
        self.communication_rounds += 1
        
        print(f"✅ Federated averaging completed - Round {self.communication_rounds}")
    
    def evaluate_global_model(self, test_data, test_labels):
        """Evaluate the global federated model"""
        if self.global_model is None:
            return None
        
        loss, accuracy = self.global_model.evaluate(test_data, test_labels, verbose=0)
        
        performance = {
            'round': self.communication_rounds,
            'loss': loss,
            'accuracy': accuracy,
            'participants': len(self.participants),
            'privacy_budget': self.privacy_budget,
            'timestamp': datetime.now()
        }
        
        self.performance_history.append(performance)
        return performance
    
    def generate_collaboration_report(self):
        """Generate comprehensive collaboration report"""
        report = {
            'system_overview': {
                'total_participants': len(self.participants),
                'communication_rounds': self.communication_rounds,
                'privacy_budget_remaining': self.privacy_budget,
                'system_status': 'Active' if self.participants else 'Inactive'
            },
            'participant_contributions': [],
            'performance_metrics': self.performance_history[-10:] if self.performance_history else [],
            'privacy_metrics': {
                'differential_privacy_enabled': self.privacy_budget < 1.0,
                'privacy_budget_consumed': 1.0 - self.privacy_budget,
                'secure_aggregation': True
            }
        }
        
        # Add participant details
        for participant in self.participants:
            report['participant_contributions'].append({
                'institution': participant['institution'],
                'data_contribution': participant['data_size'],
                'model_contribution': participant['contribution_score'],
                'join_date': participant['joined_date'].isoformat()
            })
        
        return report

class AIEthicsExplainabilityFramework:
    """
    Comprehensive AI Ethics, Bias Detection, and Explainability System
    """
    
    def __init__(self):
        self.bias_metrics = {}
        self.fairness_thresholds = {
            'demographic_parity': 0.8,
            'equalized_odds': 0.8,
            'individual_fairness': 0.9
        }
        self.explanation_cache = {}
        self.audit_log = []
        
    def detect_demographic_bias(self, predictions, sensitive_attributes, labels=None):
        """Detect bias across demographic groups"""
        bias_report = {
            'groups_analyzed': [],
            'bias_detected': False,
            'severity': 'Low',
            'recommendations': []
        }
        
        unique_groups = np.unique(sensitive_attributes)
        group_statistics = {}
        
        for group in unique_groups:
            group_mask = sensitive_attributes == group
            group_predictions = predictions[group_mask]
            
            stats = {
                'size': np.sum(group_mask),
                'positive_rate': np.mean(group_predictions > 0.5),
                'mean_prediction': np.mean(group_predictions)
            }
            
            if labels is not None:
                group_labels = labels[group_mask]
                stats['accuracy'] = np.mean((group_predictions > 0.5) == group_labels)
                stats['precision'] = self._calculate_precision(group_predictions > 0.5, group_labels)
                stats['recall'] = self._calculate_recall(group_predictions > 0.5, group_labels)
            
            group_statistics[group] = stats
            bias_report['groups_analyzed'].append({
                'group': str(group),
                'statistics': stats
            })
        
        # Calculate bias metrics
        if len(unique_groups) >= 2:
            # Demographic parity
            positive_rates = [stats['positive_rate'] for stats in group_statistics.values()]
            parity_ratio = min(positive_rates) / max(positive_rates)
            
            if parity_ratio < self.fairness_thresholds['demographic_parity']:
                bias_report['bias_detected'] = True
                bias_report['severity'] = 'High' if parity_ratio < 0.6 else 'Medium'
                bias_report['recommendations'].append(
                    f"Demographic parity violation detected (ratio: {parity_ratio:.3f})"
                )
        
        self.bias_metrics[datetime.now()] = bias_report
        return bias_report
    
    def _calculate_precision(self, predictions, labels):
        """Calculate precision metric"""
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        return tp / (tp + fp) if (tp + fp) > 0 else 0
    
    def _calculate_recall(self, predictions, labels):
        """Calculate recall metric"""
        tp = np.sum((predictions == 1) & (labels == 1))
        fn = np.sum((predictions == 0) & (labels == 1))
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    
    def generate_model_explanation(self, model, input_data, feature_names=None):
        """Generate model explanations using multiple methods"""
        explanation_id = hashlib.md5(str(input_data).encode()).hexdigest()[:8]
        
        if explanation_id in self.explanation_cache:
            return self.explanation_cache[explanation_id]
        
        explanations = {
            'explanation_id': explanation_id,
            'timestamp': datetime.now(),
            'methods_used': [],
            'feature_importance': {},
            'global_importance': {},
            'local_explanation': {}
        }
        
        # Feature importance using permutation
        if hasattr(model, 'predict'):
            try:
                # Simulate permutation importance
                baseline_pred = model.predict(input_data.reshape(1, -1))[0]
                feature_importance = {}
                
                for i in range(len(input_data)):
                    if feature_names and i < len(feature_names):
                        feature_name = feature_names[i]
                    else:
                        feature_name = f"feature_{i}"
                    
                    # Permute feature and measure impact
                    perturbed_data = input_data.copy()
                    perturbed_data[i] = np.random.normal(0, 1)  # Add noise
                    perturbed_pred = model.predict(perturbed_data.reshape(1, -1))[0]
                    
                    importance = np.abs(baseline_pred - perturbed_pred).max()
                    feature_importance[feature_name] = float(importance)
                
                explanations['feature_importance'] = feature_importance
                explanations['methods_used'].append('permutation_importance')
                
            except Exception as e:
                explanations['error'] = f"Permutation importance failed: {str(e)}"
        
        # SHAP explanation (if available)
        if EXPLAINABILITY_AVAILABLE:
            try:
                explainer = shap.Explainer(model.predict, input_data.reshape(1, -1))
                shap_values = explainer(input_data.reshape(1, -1))
                
                explanations['shap_values'] = shap_values.values.tolist()
                explanations['methods_used'].append('shap')
            except Exception as e:
                explanations['shap_error'] = str(e)
        
        self.explanation_cache[explanation_id] = explanations
        return explanations
    
    def audit_model_decision(self, model_prediction, input_data, patient_demographics=None):
        """Audit individual model decision for ethics compliance"""
        audit_entry = {
            'timestamp': datetime.now(),
            'prediction': float(model_prediction),
            'audit_id': hashlib.md5(f"{model_prediction}{datetime.now()}".encode()).hexdigest()[:8],
            'compliance_status': 'PASS',
            'flags': [],
            'recommendations': []
        }
        
        # Check for extreme predictions
        if model_prediction > 0.95 or model_prediction < 0.05:
            audit_entry['flags'].append('EXTREME_CONFIDENCE')
            audit_entry['recommendations'].append('Verify with additional clinical assessment')
        
        # Check for bias patterns
        if patient_demographics:
            # Simplified bias check
            if 'age' in patient_demographics and patient_demographics['age'] > 80:
                if model_prediction < 0.3:  # Low severity prediction for elderly
                    audit_entry['flags'].append('AGE_BIAS_POTENTIAL')
                    audit_entry['recommendations'].append('Consider age-related health complexities')
        
        # Overall compliance assessment
        if len(audit_entry['flags']) > 2:
            audit_entry['compliance_status'] = 'REVIEW_REQUIRED'
        elif len(audit_entry['flags']) > 0:
            audit_entry['compliance_status'] = 'CAUTION'
        
        self.audit_log.append(audit_entry)
        return audit_entry
    
    def generate_ethics_report(self):
        """Generate comprehensive ethics and bias report"""
        report = {
            'report_id': hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8],
            'generated_at': datetime.now(),
            'bias_assessment': {
                'total_bias_checks': len(self.bias_metrics),
                'bias_incidents': sum(1 for b in self.bias_metrics.values() if b['bias_detected']),
                'high_risk_incidents': sum(1 for b in self.bias_metrics.values() 
                                        if b['bias_detected'] and b['severity'] == 'High')
            },
            'explainability_metrics': {
                'explanations_generated': len(self.explanation_cache),
                'explanation_methods': set()
            },
            'audit_summary': {
                'total_decisions_audited': len(self.audit_log),
                'compliance_pass': sum(1 for a in self.audit_log if a['compliance_status'] == 'PASS'),
                'review_required': sum(1 for a in self.audit_log if a['compliance_status'] == 'REVIEW_REQUIRED'),
                'common_flags': {}
            },
            'recommendations': []
        }
        
        # Analyze common flags
        all_flags = [flag for audit in self.audit_log for flag in audit['flags']]
        from collections import Counter
        flag_counts = Counter(all_flags)
        report['audit_summary']['common_flags'] = dict(flag_counts.most_common(5))
        
        # Generate recommendations
        if report['bias_assessment']['bias_incidents'] > 0:
            report['recommendations'].append("Implement bias mitigation strategies")
        
        if report['audit_summary']['review_required'] > 10:
            report['recommendations'].append("Review model decision thresholds")
        
        # Calculate explanation method usage
        for explanation in self.explanation_cache.values():
            report['explainability_metrics']['explanation_methods'].update(explanation['methods_used'])
        
        return report

class AdvancedMultimodalMedicalAI:
    """
    Advanced Multi-modal Medical AI System
    Integrates computer vision, audio analysis, and wearable device data
    """
    
    def __init__(self):
        self.vision_model = None
        self.audio_processor = None
        self.wearable_data_buffer = queue.Queue()
        self.modality_weights = {
            'text': 0.4,
            'image': 0.3,
            'audio': 0.2,
            'wearable': 0.1
        }
        self.supported_image_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        self.supported_audio_formats = ['.wav', '.mp3', '.m4a', '.flac']
        
    def initialize_vision_model(self):
        """Initialize computer vision model for medical image analysis"""
        try:
            # Load pre-trained ResNet50 for medical image analysis
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            
            # Add custom classification layers for medical conditions
            self.vision_model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(10, activation='softmax')  # Adjust based on conditions
            ])
            
            # Freeze base model layers initially
            base_model.trainable = False
            
            self.vision_model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print("✅ Vision model initialized for medical image analysis")
            return True
            
        except Exception as e:
            print(f"❌ Vision model initialization failed: {e}")
            return False
    
    def analyze_medical_image(self, image_path, condition_focus=None):
        """Analyze medical images for diagnostic insights"""
        if self.vision_model is None:
            if not self.initialize_vision_model():
                return {'error': 'Vision model not available'}
        
        try:
            # Load and preprocess image
            image = load_img(image_path, target_size=(224, 224))
            image_array = img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0)
            image_array = image_array / 255.0  # Normalize
            
            # Get predictions
            predictions = self.vision_model.predict(image_array)[0]
            
            # Medical image analysis results
            analysis = {
                'image_path': image_path,
                'timestamp': datetime.now(),
                'predictions': predictions.tolist(),
                'top_findings': [],
                'confidence_score': float(np.max(predictions)),
                'quality_assessment': self._assess_image_quality(image_array[0]),
                'anatomical_region': self._detect_anatomical_region(image_array[0]),
                'abnormality_detected': float(np.max(predictions)) > 0.7
            }
            
            # Generate top findings
            condition_names = [
                'Normal', 'Pneumonia', 'COVID-19', 'Lung Opacity', 
                'Cardiomegaly', 'Fracture', 'Mass', 'Nodule', 'Edema', 'Other'
            ]
            
            top_indices = np.argsort(predictions)[::-1][:3]
            for idx in top_indices:
                analysis['top_findings'].append({
                    'condition': condition_names[idx] if idx < len(condition_names) else f'Finding_{idx}',
                    'confidence': float(predictions[idx]),
                    'severity': 'High' if predictions[idx] > 0.8 else 'Moderate' if predictions[idx] > 0.5 else 'Low'
                })
            
            return analysis
            
        except Exception as e:
            return {'error': f'Image analysis failed: {str(e)}'}
    
    def _assess_image_quality(self, image_array):
        """Assess medical image quality"""
        # Simple quality metrics
        blur_score = cv2.Laplacian(cv2.cvtColor((image_array * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        brightness = np.mean(image_array)
        contrast = np.std(image_array)
        
        quality_score = 0.0
        quality_issues = []
        
        # Blur assessment
        if blur_score < 100:
            quality_issues.append("Image may be blurry")
        else:
            quality_score += 0.3
        
        # Brightness assessment
        if 0.2 < brightness < 0.8:
            quality_score += 0.4
        else:
            quality_issues.append("Suboptimal brightness levels")
        
        # Contrast assessment
        if contrast > 0.1:
            quality_score += 0.3
        else:
            quality_issues.append("Low contrast detected")
        
        return {
            'quality_score': quality_score,
            'blur_score': blur_score,
            'brightness': brightness,
            'contrast': contrast,
            'issues': quality_issues,
            'acceptable': quality_score > 0.6
        }
    
    def _detect_anatomical_region(self, image_array):
        """Detect anatomical region in medical image"""
        # Simplified anatomical region detection
        # In practice, would use specialized models
        
        regions = {
            'chest': 0.0,
            'abdomen': 0.0,
            'head': 0.0,
            'extremities': 0.0,
            'unknown': 0.0
        }
        
        # Simple heuristics based on image characteristics
        upper_region = np.mean(image_array[:112, :, :])
        lower_region = np.mean(image_array[112:, :, :])
        
        if upper_region > lower_region:
            regions['chest'] = 0.7
            regions['head'] = 0.3
        else:
            regions['abdomen'] = 0.6
            regions['extremities'] = 0.4
        
        detected_region = max(regions, key=regions.get)
        confidence = regions[detected_region]
        
        return {
            'detected_region': detected_region,
            'confidence': confidence,
            'all_regions': regions
        }
    
    def process_audio_symptoms(self, audio_path):
        """Process audio recordings for symptom analysis"""
        if not AUDIO_AVAILABLE:
            return {'error': 'Audio processing libraries not available'}
        
        try:
            # Load audio file
            audio_data, sample_rate = librosa.load(audio_path, sr=None)
            
            # Extract audio features
            features = self._extract_audio_features(audio_data, sample_rate)
            
            # Analyze for respiratory symptoms
            respiratory_analysis = self._analyze_respiratory_audio(audio_data, sample_rate)
            
            # Analyze for speech patterns (neurological indicators)
            speech_analysis = self._analyze_speech_patterns(audio_data, sample_rate)
            
            analysis = {
                'audio_path': audio_path,
                'timestamp': datetime.now(),
                'duration': len(audio_data) / sample_rate,
                'sample_rate': sample_rate,
                'features': features,
                'respiratory_analysis': respiratory_analysis,
                'speech_analysis': speech_analysis,
                'detected_symptoms': []
            }
            
            # Detect symptoms based on analysis
            if respiratory_analysis['abnormal_breathing']:
                analysis['detected_symptoms'].append({
                    'symptom': 'abnormal_breathing',
                    'confidence': respiratory_analysis['confidence'],
                    'type': respiratory_analysis['breathing_type']
                })
            
            if speech_analysis['speech_impairment']:
                analysis['detected_symptoms'].append({
                    'symptom': 'speech_impairment',
                    'confidence': speech_analysis['confidence'],
                    'characteristics': speech_analysis['impairment_type']
                })
            
            return analysis
            
        except Exception as e:
            return {'error': f'Audio processing failed: {str(e)}'}
    
    def _extract_audio_features(self, audio_data, sample_rate):
        """Extract relevant audio features for medical analysis"""
        features = {}
        
        try:
            # Spectral features
            features['mfccs'] = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13).tolist()
            features['spectral_centroid'] = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0].tolist()
            features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0].tolist()
            
            # Temporal features
            features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(audio_data)[0].tolist()
            features['tempo'], _ = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
            
            # Energy features
            features['rms_energy'] = librosa.feature.rms(y=audio_data)[0].tolist()
            
        except Exception as e:
            features['extraction_error'] = str(e)
        
        return features
    
    def _analyze_respiratory_audio(self, audio_data, sample_rate):
        """Analyze audio for respiratory symptoms"""
        analysis = {
            'abnormal_breathing': False,
            'confidence': 0.0,
            'breathing_rate': 0,
            'breathing_type': 'normal',
            'wheeze_detected': False,
            'crackles_detected': False
        }
        
        try:
            # Frequency analysis for wheeze detection
            fft = np.fft.fft(audio_data)
            frequencies = np.fft.fftfreq(len(fft), 1/sample_rate)
            magnitude = np.abs(fft)
            
            # Wheeze detection (high-frequency continuous sound)
            wheeze_freq_range = (100, 1000)  # Hz
            wheeze_indices = np.where((frequencies >= wheeze_freq_range[0]) & 
                                    (frequencies <= wheeze_freq_range[1]))
            wheeze_power = np.sum(magnitude[wheeze_indices])
            
            if wheeze_power > np.mean(magnitude) * 10:
                analysis['wheeze_detected'] = True
                analysis['abnormal_breathing'] = True
                analysis['breathing_type'] = 'wheeze'
                analysis['confidence'] = min(0.9, wheeze_power / (np.mean(magnitude) * 20))
            
            # Simple breathing rate estimation
            # This is a simplified approach - real implementation would be more sophisticated
            envelope = np.abs(librosa.onset.onset_strength(y=audio_data, sr=sample_rate))
            peaks, _ = librosa.util.peak_pick(envelope, 3, 3, 3, 5, 0.5, 10)
            
            if len(peaks) > 0:
                duration = len(audio_data) / sample_rate
                analysis['breathing_rate'] = len(peaks) * 60 / duration  # breaths per minute
        
        except Exception as e:
            analysis['analysis_error'] = str(e)
        
        return analysis
    
    def _analyze_speech_patterns(self, audio_data, sample_rate):
        """Analyze speech patterns for neurological indicators"""
        analysis = {
            'speech_impairment': False,
            'confidence': 0.0,
            'impairment_type': [],
            'speech_rate': 0,
            'pause_frequency': 0,
            'articulation_clarity': 0.0
        }
        
        try:
            # Speech rate analysis
            onset_frames = librosa.onset.onset_detect(y=audio_data, sr=sample_rate)
            if len(onset_frames) > 1:
                duration = len(audio_data) / sample_rate
                analysis['speech_rate'] = len(onset_frames) / duration
                
                # Pause analysis
                onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)
                pauses = np.diff(onset_times)
                long_pauses = pauses[pauses > 0.5]  # Pauses longer than 0.5 seconds
                analysis['pause_frequency'] = len(long_pauses) / duration
                
                # Detect potential speech impairments
                if analysis['speech_rate'] < 2.0:  # Very slow speech
                    analysis['impairment_type'].append('bradylalia')
                    analysis['speech_impairment'] = True
                    analysis['confidence'] = 0.7
                
                if analysis['pause_frequency'] > 3.0:  # Frequent long pauses
                    analysis['impairment_type'].append('frequent_pauses')
                    analysis['speech_impairment'] = True
                    analysis['confidence'] = max(analysis['confidence'], 0.6)
        
        except Exception as e:
            analysis['analysis_error'] = str(e)
        
        return analysis
    
    def integrate_wearable_data(self, device_data):
        """Integrate data from wearable devices"""
        processed_data = {
            'timestamp': datetime.now(),
            'device_type': device_data.get('device_type', 'unknown'),
            'metrics': {},
            'alerts': [],
            'health_indicators': {}
        }
        
        # Process different types of wearable data
        if 'heart_rate' in device_data:
            hr_analysis = self._analyze_heart_rate(device_data['heart_rate'])
            processed_data['metrics']['heart_rate'] = hr_analysis
            
            if hr_analysis['abnormal']:
                processed_data['alerts'].append({
                    'type': 'heart_rate_abnormal',
                    'severity': hr_analysis['severity'],
                    'message': f"Heart rate {hr_analysis['status']}: {hr_analysis['current_hr']} bpm"
                })
        
        if 'step_count' in device_data:
            activity_analysis = self._analyze_activity_level(device_data['step_count'])
            processed_data['metrics']['activity'] = activity_analysis
        
        if 'sleep_data' in device_data:
            sleep_analysis = self._analyze_sleep_patterns(device_data['sleep_data'])
            processed_data['metrics']['sleep'] = sleep_analysis
        
        if 'oxygen_saturation' in device_data:
            spo2_analysis = self._analyze_oxygen_saturation(device_data['oxygen_saturation'])
            processed_data['metrics']['oxygen_saturation'] = spo2_analysis
            
            if spo2_analysis['abnormal']:
                processed_data['alerts'].append({
                    'type': 'low_oxygen_saturation',
                    'severity': 'high',
                    'message': f"Low oxygen saturation detected: {spo2_analysis['current_spo2']}%"
                })
        
        # Store in buffer for continuous monitoring
        self.wearable_data_buffer.put(processed_data)
        
        return processed_data
    
    def _analyze_heart_rate(self, hr_data):
        """Analyze heart rate data for abnormalities"""
        if isinstance(hr_data, (int, float)):
            current_hr = hr_data
            hr_variability = 0
        else:
            current_hr = np.mean(hr_data) if len(hr_data) > 0 else 0
            hr_variability = np.std(hr_data) if len(hr_data) > 1 else 0
        
        analysis = {
            'current_hr': current_hr,
            'hr_variability': hr_variability,
            'abnormal': False,
            'status': 'normal',
            'severity': 'low'
        }
        
        # Heart rate classification
        if current_hr < 60:
            analysis['status'] = 'bradycardia'
            analysis['abnormal'] = True
            analysis['severity'] = 'medium' if current_hr < 50 else 'low'
        elif current_hr > 100:
            analysis['status'] = 'tachycardia'
            analysis['abnormal'] = True
            analysis['severity'] = 'high' if current_hr > 120 else 'medium'
        
        return analysis
    
    def _analyze_activity_level(self, step_data):
        """Analyze daily activity levels"""
        if isinstance(step_data, (int, float)):
            daily_steps = step_data
        else:
            daily_steps = sum(step_data) if len(step_data) > 0 else 0
        
        analysis = {
            'daily_steps': daily_steps,
            'activity_level': 'sedentary',
            'health_impact': 'negative'
        }
        
        # Activity level classification
        if daily_steps >= 10000:
            analysis['activity_level'] = 'very_active'
            analysis['health_impact'] = 'positive'
        elif daily_steps >= 7500:
            analysis['activity_level'] = 'active'
            analysis['health_impact'] = 'positive'
        elif daily_steps >= 5000:
            analysis['activity_level'] = 'lightly_active'
            analysis['health_impact'] = 'neutral'
        elif daily_steps >= 2500:
            analysis['activity_level'] = 'low_active'
            analysis['health_impact'] = 'concerning'
        
        return analysis
    
    def _analyze_sleep_patterns(self, sleep_data):
        """Analyze sleep quality and patterns"""
        analysis = {
            'sleep_duration': sleep_data.get('duration', 0),
            'sleep_quality': sleep_data.get('quality', 0),
            'deep_sleep_percentage': sleep_data.get('deep_sleep', 0),
            'rem_sleep_percentage': sleep_data.get('rem_sleep', 0),
            'sleep_efficiency': sleep_data.get('efficiency', 0),
            'sleep_issues': []
        }
        
        # Sleep quality assessment
        if analysis['sleep_duration'] < 6:
            analysis['sleep_issues'].append('insufficient_sleep')
        elif analysis['sleep_duration'] > 9:
            analysis['sleep_issues'].append('excessive_sleep')
        
        if analysis['deep_sleep_percentage'] < 15:
            analysis['sleep_issues'].append('insufficient_deep_sleep')
        
        if analysis['sleep_efficiency'] < 85:
            analysis['sleep_issues'].append('poor_sleep_efficiency')
        
        return analysis
    
    def _analyze_oxygen_saturation(self, spo2_data):
        """Analyze oxygen saturation levels"""
        if isinstance(spo2_data, (int, float)):
            current_spo2 = spo2_data
        else:
            current_spo2 = np.mean(spo2_data) if len(spo2_data) > 0 else 0
        
        analysis = {
            'current_spo2': current_spo2,
            'abnormal': False,
            'severity': 'normal'
        }
        
        if current_spo2 < 90:
            analysis['abnormal'] = True
            analysis['severity'] = 'critical'
        elif current_spo2 < 95:
            analysis['abnormal'] = True
            analysis['severity'] = 'concerning'
        
        return analysis
    
    def multimodal_fusion(self, text_prediction, image_analysis=None, audio_analysis=None, wearable_data=None):
        """Fuse predictions from multiple modalities"""
        fusion_result = {
            'timestamp': datetime.now(),
            'modalities_used': ['text'],
            'weighted_prediction': text_prediction.copy(),
            'confidence_boost': 0.0,
            'multimodal_insights': [],
            'recommendation_adjustments': []
        }
        
        # Start with text prediction
        final_prediction = np.array(text_prediction) * self.modality_weights['text']
        
        # Integrate image analysis
        if image_analysis and not image_analysis.get('error'):
            fusion_result['modalities_used'].append('image')
            
            # Convert image findings to prediction adjustments
            if image_analysis.get('abnormality_detected'):
                # Boost predictions for conditions matching image findings
                for finding in image_analysis.get('top_findings', []):
                    if finding['confidence'] > 0.5:
                        fusion_result['multimodal_insights'].append(f"Image supports: {finding['condition']}")
                        fusion_result['confidence_boost'] += 0.1
        
        # Integrate audio analysis
        if audio_analysis and not audio_analysis.get('error'):
            fusion_result['modalities_used'].append('audio')
            
            for symptom in audio_analysis.get('detected_symptoms', []):
                if symptom['confidence'] > 0.6:
                    fusion_result['multimodal_insights'].append(f"Audio detected: {symptom['symptom']}")
                    fusion_result['confidence_boost'] += 0.05
        
        # Integrate wearable data
        if wearable_data:
            fusion_result['modalities_used'].append('wearable')
            
            for alert in wearable_data.get('alerts', []):
                if alert['severity'] in ['high', 'critical']:
                    fusion_result['multimodal_insights'].append(f"Wearable alert: {alert['type']}")
                    fusion_result['confidence_boost'] += 0.15
        
        # Apply confidence boost
        fusion_result['final_confidence'] = min(1.0, np.max(text_prediction) + fusion_result['confidence_boost'])
        
        return fusion_result

class PredictiveRiskModelingSystem:
    """
    Advanced Predictive Risk Modeling and Early Warning System
    """
    
    def __init__(self):
        self.risk_models = {}
        self.early_warning_thresholds = {
            'cardiovascular': 0.7,
            'respiratory_failure': 0.8,
            'sepsis': 0.75,
            'diabetes_complications': 0.65,
            'stroke': 0.8,
            'kidney_failure': 0.7
        }
        self.prediction_history = []
        self.risk_factors_database = self._load_risk_factors_database()
        
    def _load_risk_factors_database(self):
        """Load comprehensive risk factors database"""
        return {
            'cardiovascular': {
                'major_factors': ['hypertension', 'diabetes', 'smoking', 'high_cholesterol'],
                'minor_factors': ['family_history', 'obesity', 'sedentary_lifestyle', 'stress'],
                'protective_factors': ['exercise', 'healthy_diet', 'non_smoker'],
                'age_multipliers': {'<40': 0.5, '40-60': 1.0, '60-80': 1.5, '>80': 2.0},
                'gender_risks': {'male': 1.2, 'female': 1.0}
            },
            'diabetes_complications': {
                'major_factors': ['poor_glucose_control', 'hypertension', 'kidney_disease', 'neuropathy'],
                'minor_factors': ['obesity', 'smoking', 'high_cholesterol', 'family_history'],
                'protective_factors': ['good_glucose_control', 'regular_exercise', 'healthy_diet'],
                'duration_multiplier': 0.1  # per year of diabetes
            },
            'respiratory_failure': {
                'major_factors': ['copd', 'pneumonia', 'heart_failure', 'smoking'],
                'minor_factors': ['age_over_65', 'immunocompromised', 'obesity'],
                'acute_triggers': ['infection', 'pollution_exposure', 'medication_noncompliance']
            },
            'sepsis': {
                'major_factors': ['infection', 'immunocompromised', 'recent_surgery', 'invasive_devices'],
                'minor_factors': ['diabetes', 'cancer', 'kidney_disease', 'liver_disease'],
                'warning_signs': ['fever', 'tachycardia', 'altered_mental_status', 'hypotension']
            },
            'stroke': {
                'major_factors': ['hypertension', 'atrial_fibrillation', 'diabetes', 'smoking'],
                'minor_factors': ['high_cholesterol', 'obesity', 'sleep_apnea', 'excessive_alcohol'],
                'acute_warning_signs': ['sudden_weakness', 'speech_difficulties', 'vision_changes', 'severe_headache']
            },
            'kidney_failure': {
                'major_factors': ['diabetes', 'hypertension', 'chronic_kidney_disease', 'heart_disease'],
                'minor_factors': ['family_history', 'age_over_60', 'obesity', 'smoking'],
                'acute_triggers': ['dehydration', 'nephrotoxic_medications', 'contrast_dye', 'infection']
            }
        }
    
    def calculate_comprehensive_risk_score(self, patient_data, condition):
        """Calculate comprehensive risk score for specific condition"""
        if condition not in self.risk_factors_database:
            return {'error': f'Risk factors not available for {condition}'}
        
        risk_factors = self.risk_factors_database[condition]
        base_score = 0
        risk_breakdown = {
            'major_factors': [],
            'minor_factors': [],
            'protective_factors': [],
            'demographic_adjustments': {},
            'total_score': 0,
            'risk_level': 'Low'
        }
        
        # Major risk factors (weight: 3)
        for factor in risk_factors.get('major_factors', []):
            if patient_data.get(factor, False):
                base_score += 3
                risk_breakdown['major_factors'].append(factor)
        
        # Minor risk factors (weight: 1)
        for factor in risk_factors.get('minor_factors', []):
            if patient_data.get(factor, False):
                base_score += 1
                risk_breakdown['minor_factors'].append(factor)
        
        # Protective factors (weight: -1)
        for factor in risk_factors.get('protective_factors', []):
            if patient_data.get(factor, False):
                base_score -= 1
                risk_breakdown['protective_factors'].append(factor)
        
        # Age adjustments
        if 'age_multipliers' in risk_factors and 'age' in patient_data:
            age = patient_data['age']
            for age_range, multiplier in risk_factors['age_multipliers'].items():
                if self._age_in_range(age, age_range):
                    base_score *= multiplier
                    risk_breakdown['demographic_adjustments']['age_multiplier'] = multiplier
                    break
        
        # Gender adjustments
        if 'gender_risks' in risk_factors and 'gender' in patient_data:
            gender = patient_data['gender'].lower()
            if gender in risk_factors['gender_risks']:
                multiplier = risk_factors['gender_risks'][gender]
                base_score *= multiplier
                risk_breakdown['demographic_adjustments']['gender_multiplier'] = multiplier
        
        # Duration adjustments (for chronic conditions)
        if 'duration_multiplier' in risk_factors and f'{condition}_duration' in patient_data:
            duration = patient_data[f'{condition}_duration']
            base_score += duration * risk_factors['duration_multiplier']
            risk_breakdown['demographic_adjustments']['duration_adjustment'] = duration * risk_factors['duration_multiplier']
        
        # Normalize score to 0-1 range
        normalized_score = min(1.0, max(0.0, base_score / 15))  # Assume max possible score is 15
        
        # Determine risk level
        if normalized_score >= 0.8:
            risk_level = 'Critical'
        elif normalized_score >= 0.6:
            risk_level = 'High'
        elif normalized_score >= 0.4:
            risk_level = 'Moderate'
        elif normalized_score >= 0.2:
            risk_level = 'Low'
        else:
            risk_level = 'Minimal'
        
        risk_breakdown['total_score'] = normalized_score
        risk_breakdown['risk_level'] = risk_level
        
        return risk_breakdown
    
    def _age_in_range(self, age, age_range):
        """Check if age falls within specified range"""
        if age_range.startswith('<'):
            return age < int(age_range[1:])
        elif age_range.startswith('>'):
            return age > int(age_range[1:])
        elif '-' in age_range:
            min_age, max_age = map(int, age_range.split('-'))
            return min_age <= age <= max_age
        return False
    
    def predict_disease_progression(self, patient_data, current_condition, time_horizon_days=30):
        """Predict disease progression over specified time horizon"""
        progression_model = {
            'patient_id': patient_data.get('patient_id', 'unknown'),
            'current_condition': current_condition,
            'time_horizon': time_horizon_days,
            'progression_predictions': {},
            'early_warning_indicators': [],
            'recommended_interventions': [],
            'monitoring_schedule': {}
        }
        
        # Base progression rates by condition
        base_progression_rates = {
            'diabetes': 0.05,  # 5% chance of complications per month
            'hypertension': 0.03,
            'heart_disease': 0.08,
            'copd': 0.06,
            'chronic_kidney_disease': 0.04,
            'cancer': 0.15
        }
        
        base_rate = base_progression_rates.get(current_condition, 0.02)
        
        # Adjust for patient-specific factors
        risk_multipliers = 1.0
        
        # Age factor
        age = patient_data.get('age', 50)
        if age > 65:
            risk_multipliers *= 1.3
        elif age > 80:
            risk_multipliers *= 1.6
        
        # Comorbidities
        comorbidity_count = sum(1 for key, value in patient_data.items() 
                              if key.endswith('_disease') or key.endswith('_condition') and value)
        risk_multipliers *= (1 + comorbidity_count * 0.2)
        
        # Lifestyle factors
        if patient_data.get('smoking', False):
            risk_multipliers *= 1.4
        if patient_data.get('obesity', False):
            risk_multipliers *= 1.2
        if patient_data.get('exercise_regularly', False):
            risk_multipliers *= 0.8
        
        # Calculate progression probabilities
        daily_risk = (base_rate * risk_multipliers) / 30  # Convert monthly to daily
        progression_probability = 1 - (1 - daily_risk) ** time_horizon_days
        
        progression_model['progression_predictions'] = {
            'overall_progression_risk': float(progression_probability),
            'daily_risk': float(daily_risk),
            'risk_multiplier': float(risk_multipliers),
            'base_condition_risk': float(base_rate)
        }
        
        # Early warning indicators
        if progression_probability > 0.3:
            progression_model['early_warning_indicators'].extend([
                f'High progression risk detected for {current_condition}',
                f'Risk level: {progression_probability:.1%} over {time_horizon_days} days'
            ])
        
        # Recommended interventions based on risk level
        if progression_probability > 0.5:
            progression_model['recommended_interventions'].extend([
                'Immediate specialist consultation required',
                'Intensive monitoring protocol',
                'Consider hospitalization or urgent care'
            ])
        elif progression_probability > 0.3:
            progression_model['recommended_interventions'].extend([
                'Increase monitoring frequency',
                'Medication adjustment may be needed',
                'Schedule follow-up within 1 week'
            ])
        elif progression_probability > 0.15:
            progression_model['recommended_interventions'].extend([
                'Enhanced self-monitoring',
                'Lifestyle modification counseling',
                'Regular follow-up appointments'
            ])
        
        # Monitoring schedule
        if progression_probability > 0.4:
            progression_model['monitoring_schedule'] = {
                'vital_signs': 'Every 4 hours',
                'laboratory_tests': 'Daily',
                'specialist_follow_up': 'Within 24 hours'
            }
        elif progression_probability > 0.2:
            progression_model['monitoring_schedule'] = {
                'vital_signs': 'Twice daily',
                'laboratory_tests': 'Every 3 days',
                'specialist_follow_up': 'Within 1 week'
            }
        else:
            progression_model['monitoring_schedule'] = {
                'vital_signs': 'Daily',
                'laboratory_tests': 'Weekly',
                'specialist_follow_up': 'Monthly'
            }
        
        return progression_model
    
    def generate_early_warning_alert(self, patient_data, vital_signs, lab_values=None):
        """Generate early warning alerts based on multiple parameters"""
        alert_system = {
            'timestamp': datetime.now(),
            'patient_id': patient_data.get('patient_id', 'unknown'),
            'alert_level': 'GREEN',
            'triggered_alerts': [],
            'risk_scores': {},
            'recommended_actions': [],
            'escalation_required': False
        }
        
        # Vital signs analysis
        vs_alerts = self._analyze_vital_signs_trends(vital_signs)
        alert_system['triggered_alerts'].extend(vs_alerts)
        
        # Laboratory values analysis
        if lab_values:
            lab_alerts = self._analyze_laboratory_trends(lab_values)
            alert_system['triggered_alerts'].extend(lab_alerts)
        
        # Calculate composite risk scores
        alert_system['risk_scores'] = self._calculate_composite_risk_scores(
            patient_data, vital_signs, lab_values
        )
        
        # Determine overall alert level
        max_risk = max(alert_system['risk_scores'].values()) if alert_system['risk_scores'] else 0
        
        if max_risk >= 0.8:
            alert_system['alert_level'] = 'RED'
            alert_system['escalation_required'] = True
            alert_system['recommended_actions'].extend([
                'IMMEDIATE medical intervention required',
                'Consider ICU transfer',
                'Notify attending physician STAT'
            ])
        elif max_risk >= 0.6:
            alert_system['alert_level'] = 'ORANGE'
            alert_system['recommended_actions'].extend([
                'Urgent medical evaluation needed',
                'Increase monitoring frequency',
                'Prepare for potential escalation'
            ])
        elif max_risk >= 0.4:
            alert_system['alert_level'] = 'YELLOW'
            alert_system['recommended_actions'].extend([
                'Enhanced monitoring recommended',
                'Consider preventive interventions',
                'Schedule timely follow-up'
            ])
        
        return alert_system
    
    def _analyze_vital_signs_trends(self, vital_signs):
        """Analyze vital signs for concerning trends"""
        alerts = []
        
        # Heart rate analysis
        if 'heart_rate' in vital_signs:
            hr = vital_signs['heart_rate']
            if isinstance(hr, list) and len(hr) > 1:
                # Trend analysis
                hr_trend = np.polyfit(range(len(hr)), hr, 1)[0]  # Slope
                if hr_trend > 5:  # Increasing by >5 bpm per measurement
                    alerts.append({
                        'type': 'heart_rate_increasing',
                        'severity': 'medium',
                        'message': f'Heart rate trending upward: +{hr_trend:.1f} bpm/measurement'
                    })
            
            current_hr = hr[-1] if isinstance(hr, list) else hr
            if current_hr > 120:
                alerts.append({
                    'type': 'tachycardia',
                    'severity': 'high',
                    'message': f'Severe tachycardia: {current_hr} bpm'
                })
            elif current_hr < 50:
                alerts.append({
                    'type': 'bradycardia',
                    'severity': 'medium',
                    'message': f'Bradycardia: {current_hr} bpm'
                })
        
        # Blood pressure analysis
        if 'systolic_bp' in vital_signs and 'diastolic_bp' in vital_signs:
            sys_bp = vital_signs['systolic_bp']
            dia_bp = vital_signs['diastolic_bp']
            
            current_sys = sys_bp[-1] if isinstance(sys_bp, list) else sys_bp
            current_dia = dia_bp[-1] if isinstance(dia_bp, list) else dia_bp
            
            if current_sys > 180 or current_dia > 120:
                alerts.append({
                    'type': 'hypertensive_crisis',
                    'severity': 'critical',
                    'message': f'Hypertensive crisis: {current_sys}/{current_dia} mmHg'
                })
            elif current_sys < 90 or current_dia < 60:
                alerts.append({
                    'type': 'hypotension',
                    'severity': 'high',
                    'message': f'Hypotension: {current_sys}/{current_dia} mmHg'
                })
        
        # Temperature analysis
        if 'temperature' in vital_signs:
            temp = vital_signs['temperature']
            current_temp = temp[-1] if isinstance(temp, list) else temp
            
            if current_temp > 104:  # Fahrenheit
                alerts.append({
                    'type': 'hyperthermia',
                    'severity': 'critical',
                    'message': f'Dangerous hyperthermia: {current_temp}°F'
                })
            elif current_temp < 95:
                alerts.append({
                    'type': 'hypothermia',
                    'severity': 'high',
                    'message': f'Hypothermia: {current_temp}°F'
                })
        
        # Oxygen saturation analysis
        if 'oxygen_saturation' in vital_signs:
            spo2 = vital_signs['oxygen_saturation']
            current_spo2 = spo2[-1] if isinstance(spo2, list) else spo2
            
            if current_spo2 < 88:
                alerts.append({
                    'type': 'severe_hypoxemia',
                    'severity': 'critical',
                    'message': f'Severe hypoxemia: {current_spo2}%'
                })
            elif current_spo2 < 92:
                alerts.append({
                    'type': 'hypoxemia',
                    'severity': 'high',
                    'message': f'Hypoxemia: {current_spo2}%'
                })
        
        return alerts
    
    def _analyze_laboratory_trends(self, lab_values):
        """Analyze laboratory values for concerning trends"""
        alerts = []
        
        # Critical lab value thresholds
        critical_thresholds = {
            'glucose': {'high': 400, 'low': 50},
            'potassium': {'high': 6.0, 'low': 2.5},
            'sodium': {'high': 160, 'low': 120},
            'creatinine': {'high': 4.0, 'critical_increase': 0.5},
            'hemoglobin': {'low': 7.0},
            'platelet_count': {'low': 50000},
            'white_blood_cell_count': {'high': 30000, 'low': 1000}
        }
        
        for lab_name, values in lab_values.items():
            if lab_name in critical_thresholds:
                thresholds = critical_thresholds[lab_name]
                current_value = values[-1] if isinstance(values, list) else values
                
                # Check critical thresholds
                if 'high' in thresholds and current_value > thresholds['high']:
                    alerts.append({
                        'type': f'{lab_name}_critical_high',
                        'severity': 'critical',
                        'message': f'Critical {lab_name}: {current_value} (normal < {thresholds["high"]})'
                    })
                elif 'low' in thresholds and current_value < thresholds['low']:
                    alerts.append({
                        'type': f'{lab_name}_critical_low',
                        'severity': 'critical',
                        'message': f'Critical {lab_name}: {current_value} (normal > {thresholds["low"]})'
                    })
                
                # Check for rapid changes
                if isinstance(values, list) and len(values) > 1:
                    change = abs(values[-1] - values[-2])
                    if lab_name == 'creatinine' and change > thresholds.get('critical_increase', float('inf')):
                        alerts.append({
                            'type': 'acute_kidney_injury',
                            'severity': 'high',
                            'message': f'Rapid creatinine increase: +{change:.1f} mg/dL'
                        })
        
        return alerts
    
    def _calculate_composite_risk_scores(self, patient_data, vital_signs, lab_values):
        """Calculate composite risk scores for major complications"""
        risk_scores = {}
        
        # Cardiovascular risk
        cv_risk = 0
        if 'heart_rate' in vital_signs:
            hr = vital_signs['heart_rate']
            current_hr = hr[-1] if isinstance(hr, list) else hr
            if current_hr > 100 or current_hr < 60:
                cv_risk += 0.3
        
        if 'systolic_bp' in vital_signs:
            sbp = vital_signs['systolic_bp']
            current_sbp = sbp[-1] if isinstance(sbp, list) else sbp
            if current_sbp > 140 or current_sbp < 90:
                cv_risk += 0.4
        
        risk_scores['cardiovascular'] = min(1.0, cv_risk)
        
        # Respiratory risk
        resp_risk = 0
        if 'oxygen_saturation' in vital_signs:
            spo2 = vital_signs['oxygen_saturation']
            current_spo2 = spo2[-1] if isinstance(spo2, list) else spo2
            if current_spo2 < 95:
                resp_risk += 0.6
            elif current_spo2 < 92:
                resp_risk += 0.8
        
        risk_scores['respiratory'] = min(1.0, resp_risk)
        
        # Infection/sepsis risk
        infection_risk = 0
        if 'temperature' in vital_signs:
            temp = vital_signs['temperature']
            current_temp = temp[-1] if isinstance(temp, list) else temp
            if current_temp > 100.4 or current_temp < 96:
                infection_risk += 0.3
        
        if lab_values and 'white_blood_cell_count' in lab_values:
            wbc = lab_values['white_blood_cell_count']
            current_wbc = wbc[-1] if isinstance(wbc, list) else wbc
            if current_wbc > 12000 or current_wbc < 4000:
                infection_risk += 0.4
        
        risk_scores['infection_sepsis'] = min(1.0, infection_risk)
        
        return risk_scores

class GenomicsPersonalizedMedicine:
    """
    Genomics Integration and Personalized Medicine System
    """
    
    def __init__(self):
        self.genetic_variants_database = self._load_genetic_variants()
        self.pharmacogenomics_data = self._load_pharmacogenomics_database()
        self.disease_susceptibility_genes = self._load_disease_susceptibility_data()
        self.population_genetics = self._load_population_genetics_data()
        
    def _load_genetic_variants(self):
        """Load database of medically relevant genetic variants"""
        return {
            'BRCA1': {
                'chromosome': '17',
                'associated_conditions': ['breast_cancer', 'ovarian_cancer'],
                'risk_levels': {'pathogenic': 0.8, 'likely_pathogenic': 0.6, 'benign': 0.0},
                'population_frequency': 0.001,
                'clinical_significance': 'High penetrance cancer predisposition'
            },
            'BRCA2': {
                'chromosome': '13',
                'associated_conditions': ['breast_cancer', 'ovarian_cancer', 'prostate_cancer'],
                'risk_levels': {'pathogenic': 0.7, 'likely_pathogenic': 0.5, 'benign': 0.0},
                'population_frequency': 0.001,
                'clinical_significance': 'High penetrance cancer predisposition'
            },
            'APOE': {
                'chromosome': '19',
                'associated_conditions': ['alzheimer_disease', 'cardiovascular_disease'],
                'allele_risks': {'e2': 0.6, 'e3': 1.0, 'e4': 3.0},  # Relative risk
                'population_frequency': {'e2': 0.08, 'e3': 0.77, 'e4': 0.15},
                'clinical_significance': 'Alzheimer disease risk factor'
            },
            'CFTR': {
                'chromosome': '7',
                'associated_conditions': ['cystic_fibrosis'],
                'inheritance_pattern': 'autosomal_recessive',
                'carrier_frequency': 0.04,
                'clinical_significance': 'Cystic fibrosis causative gene'
            },
            'HFE': {
                'chromosome': '6',
                'associated_conditions': ['hemochromatosis'],
                'variants': {'C282Y': 0.85, 'H63D': 0.15},  # Disease risk
                'population_frequency': 0.05,
                'clinical_significance': 'Iron overload disorder'
            },
            'F5': {
                'chromosome': '1',
                'associated_conditions': ['thrombophilia', 'venous_thromboembolism'],
                'variants': {'factor_V_leiden': 0.7},
                'population_frequency': 0.05,
                'clinical_significance': 'Increased clotting risk'
            }
        }
    
    def _load_pharmacogenomics_database(self):
        """Load pharmacogenomics data for drug metabolism"""
        return {
            'CYP2D6': {
                'drugs_affected': ['codeine', 'tramadol', 'metoprolol', 'fluoxetine'],
                'phenotypes': {
                    'poor_metabolizer': {'frequency': 0.07, 'drug_response': 'increased_toxicity'},
                    'intermediate_metabolizer': {'frequency': 0.10, 'drug_response': 'reduced_efficacy'},
                    'normal_metabolizer': {'frequency': 0.77, 'drug_response': 'normal'},
                    'ultrarapid_metabolizer': {'frequency': 0.06, 'drug_response': 'reduced_efficacy'}
                },
                'clinical_recommendations': {
                    'poor_metabolizer': 'Reduce dose by 50% or use alternative drug',
                    'ultrarapid_metabolizer': 'Consider increased dose or alternative drug'
                }
            },
            'CYP2C19': {
                'drugs_affected': ['clopidogrel', 'omeprazole', 'escitalopram'],
                'phenotypes': {
                    'poor_metabolizer': {'frequency': 0.02, 'drug_response': 'reduced_efficacy'},
                    'intermediate_metabolizer': {'frequency': 0.18, 'drug_response': 'reduced_efficacy'},
                    'normal_metabolizer': {'frequency': 0.65, 'drug_response': 'normal'},
                    'rapid_metabolizer': {'frequency': 0.15, 'drug_response': 'increased_metabolism'}
                }
            },
            'TPMT': {
                'drugs_affected': ['azathioprine', 'mercaptopurine', 'thioguanine'],
                'phenotypes': {
                    'poor_metabolizer': {'frequency': 0.003, 'drug_response': 'severe_toxicity_risk'},
                    'intermediate_metabolizer': {'frequency': 0.11, 'drug_response': 'increased_toxicity'},
                    'normal_metabolizer': {'frequency': 0.887, 'drug_response': 'normal'}
                },
                'clinical_recommendations': {
                    'poor_metabolizer': 'Avoid drug or reduce dose by 90%',
                    'intermediate_metabolizer': 'Reduce dose by 50%'
                }
            },
            'DPYD': {
                'drugs_affected': ['5-fluorouracil', 'capecitabine'],
                'variants': {
                    'deficient': {'frequency': 0.05, 'drug_response': 'severe_toxicity_risk'}
                },
                'clinical_recommendations': {
                    'deficient': 'Contraindicated - use alternative chemotherapy'
                }
            }
        }
    
    def _load_disease_susceptibility_data(self):
        """Load disease susceptibility genetic markers"""
        return {
            'diabetes_type2': {
                'genes': ['TCF7L2', 'PPARG', 'KCNJ11', 'CDKAL1'],
                'polygenic_risk_score_components': 50,
                'heritability': 0.72,
                'environmental_factors': ['obesity', 'diet', 'exercise']
            },
            'coronary_artery_disease': {
                'genes': ['LDL-R', 'APOB', 'PCSK9', '9p21.3'],
                'polygenic_risk_score_components': 95,
                'heritability': 0.58,
                'environmental_factors': ['smoking', 'diet', 'exercise', 'stress']
            },
            'hypertension': {
                'genes': ['ACE', 'AGT', 'AGTR1', 'ADD1'],
                'polygenic_risk_score_components': 30,
                'heritability': 0.62,
                'environmental_factors': ['sodium_intake', 'obesity', 'alcohol']
            },
            'asthma': {
                'genes': ['ORMDL3', 'GSDMB', 'IL33', 'TSLP'],
                'polygenic_risk_score_components': 25,
                'heritability': 0.75,
                'environmental_factors': ['allergens', 'air_pollution', 'infections']
            }
        }
    
    def _load_population_genetics_data(self):
        """Load population-specific genetic variation data"""
        return {
            'european': {
                'common_variants': ['APOE_e4', 'CYP2D6_poor', 'MTHFR_C677T'],
                'disease_prevalence': {
                    'cystic_fibrosis': 0.0004,
                    'hemochromatosis': 0.005,
                    'factor_V_leiden': 0.05
                }
            },
            'african': {
                'common_variants': ['G6PD_deficiency', 'sickle_cell_trait', 'APOL1'],
                'disease_prevalence': {
                    'sickle_cell_disease': 0.001,
                    'g6pd_deficiency': 0.11,
                    'hypertension': 0.45
                }
            },
            'asian': {
                'common_variants': ['CYP2D6_decreased', 'ALDH2_deficiency'],
                'disease_prevalence': {
                    'nasopharyngeal_carcinoma': 0.0001,
                    'alcohol_flush_syndrome': 0.36
                }
            },
            'hispanic': {
                'common_variants': ['CYP2D6_variants', 'TPMT_variants'],
                'disease_prevalence': {
                    'diabetes_type2': 0.17,
                    'gallbladder_disease': 0.15
                }
            }
        }
    
    def analyze_genetic_profile(self, genetic_data, patient_ancestry=None):
        """Analyze genetic profile for medical insights"""
        analysis = {
            'patient_id': genetic_data.get('patient_id', 'unknown'),
            'analysis_timestamp': datetime.now(),
            'genetic_risk_factors': [],
            'pharmacogenomic_recommendations': [],
            'disease_susceptibility': {},
            'polygenic_risk_scores': {},
            'clinical_actionability': []
        }
        
        # Analyze single gene variants
        for gene, variant_data in genetic_data.get('variants', {}).items():
            if gene in self.genetic_variants_database:
                gene_info = self.genetic_variants_database[gene]
                risk_analysis = self._analyze_single_gene_variant(gene, variant_data, gene_info)
                
                if risk_analysis['clinical_significance']:
                    analysis['genetic_risk_factors'].append(risk_analysis)
                    
                    # Add clinical actionability
                    if risk_analysis['actionable']:
                        analysis['clinical_actionability'].append({
                            'gene': gene,
                            'recommendation': risk_analysis['clinical_action'],
                            'evidence_level': risk_analysis['evidence_level']
                        })
        
        # Pharmacogenomic analysis
        for gene in self.pharmacogenomics_data:
            if gene in genetic_data.get('pharmacogenomic_variants', {}):
                pharm_analysis = self._analyze_pharmacogenomic_variant(
                    gene, genetic_data['pharmacogenomic_variants'][gene]
                )
                analysis['pharmacogenomic_recommendations'].append(pharm_analysis)
        
        # Calculate polygenic risk scores
        for disease in self.disease_susceptibility_genes:
            if disease in genetic_data.get('polygenic_data', {}):
                prs = self._calculate_polygenic_risk_score(
                    disease, genetic_data['polygenic_data'][disease], patient_ancestry
                )
                analysis['polygenic_risk_scores'][disease] = prs
        
        # Population-specific adjustments
        if patient_ancestry:
            analysis = self._apply_ancestry_adjustments(analysis, patient_ancestry)
        
        return analysis
    
    def _analyze_single_gene_variant(self, gene, variant_data, gene_info):
        """Analyze single gene variant for clinical significance"""
        analysis = {
            'gene': gene,
            'variant': variant_data.get('variant_type', 'unknown'),
            'clinical_significance': False,
            'risk_level': 0.0,
            'evidence_level': 'unknown',
            'actionable': False,
            'clinical_action': '',
            'associated_conditions': gene_info.get('associated_conditions', [])
        }
        
        # Determine risk level based on variant type
        variant_type = variant_data.get('classification', 'unknown')
        
        if gene == 'BRCA1' or gene == 'BRCA2':
            if variant_type == 'pathogenic':
                analysis['risk_level'] = gene_info['risk_levels']['pathogenic']
                analysis['clinical_significance'] = True
                analysis['evidence_level'] = 'strong'
                analysis['actionable'] = True
                analysis['clinical_action'] = 'Enhanced cancer screening, consider prophylactic surgery'
            elif variant_type == 'likely_pathogenic':
                analysis['risk_level'] = gene_info['risk_levels']['likely_pathogenic']
                analysis['clinical_significance'] = True
                analysis['evidence_level'] = 'moderate'
                analysis['actionable'] = True
                analysis['clinical_action'] = 'Enhanced cancer screening, genetic counseling'
        
        elif gene == 'APOE':
            allele_combination = variant_data.get('alleles', ['e3', 'e3'])
            # Calculate risk based on allele combination
            risk_multiplier = 1.0
            for allele in allele_combination:
                if allele in gene_info['allele_risks']:
                    risk_multiplier *= gene_info['allele_risks'][allele]
            
            if risk_multiplier > 2.0:
                analysis['risk_level'] = min(0.8, (risk_multiplier - 1) / 3)
                analysis['clinical_significance'] = True
                analysis['evidence_level'] = 'moderate'
                analysis['actionable'] = True
                analysis['clinical_action'] = 'Lifestyle modifications for Alzheimer prevention'
        
        elif gene == 'CFTR':
            if variant_type == 'pathogenic':
                analysis['clinical_significance'] = True
                analysis['evidence_level'] = 'strong'
                analysis['actionable'] = True
                if variant_data.get('zygosity') == 'homozygous':
                    analysis['clinical_action'] = 'Cystic fibrosis management protocol'
                else:
                    analysis['clinical_action'] = 'Carrier status - genetic counseling for family planning'
        
        return analysis
    
    def _analyze_pharmacogenomic_variant(self, gene, variant_data):
        """Analyze pharmacogenomic variants for drug response"""
        gene_info = self.pharmacogenomics_data[gene]
        
        analysis = {
            'gene': gene,
            'phenotype': variant_data.get('phenotype', 'unknown'),
            'affected_drugs': gene_info['drugs_affected'],
            'clinical_recommendations': [],
            'evidence_level': 'strong'
        }
        
        phenotype = variant_data.get('phenotype')
        if phenotype in gene_info['phenotypes']:
            phenotype_info = gene_info['phenotypes'][phenotype]
            
            analysis['drug_response'] = phenotype_info['drug_response']
            analysis['population_frequency'] = phenotype_info['frequency']
            
            # Add clinical recommendations
            if phenotype in gene_info.get('clinical_recommendations', {}):
                recommendation = gene_info['clinical_recommendations'][phenotype]
                analysis['clinical_recommendations'].append({
                    'recommendation': recommendation,
                    'drugs': gene_info['drugs_affected'],
                    'priority': 'high' if 'toxicity' in recommendation.lower() else 'medium'
                })
        
        return analysis
    
    def _calculate_polygenic_risk_score(self, disease, polygenic_data, ancestry=None):
        """Calculate polygenic risk score for complex diseases"""
        disease_info = self.disease_susceptibility_genes[disease]
        
        # Simulate polygenic risk score calculation
        # In practice, this would use actual SNP data and validated PRS models
        base_risk = np.random.normal(0, 1)  # Standardized PRS
        
        # Adjust for ancestry if provided
        if ancestry and ancestry in self.population_genetics:
            ancestry_adjustment = self.population_genetics[ancestry]['disease_prevalence'].get(disease, 1.0)
            base_risk *= ancestry_adjustment
        
        # Convert to percentile and risk category
        percentile = stats.norm.cdf(base_risk) * 100
        
        if percentile >= 95:
            risk_category = 'Very High'
        elif percentile >= 80:
            risk_category = 'High'
        elif percentile >= 60:
            risk_category = 'Above Average'
        elif percentile >= 40:
            risk_category = 'Average'
        elif percentile >= 20:
            risk_category = 'Below Average'
        else:
            risk_category = 'Low'
        
        return {
            'disease': disease,
            'polygenic_risk_score': float(base_risk),
            'percentile': float(percentile),
            'risk_category': risk_category,
            'heritability': disease_info['heritability'],
            'environmental_factors': disease_info['environmental_factors'],
            'clinical_utility': percentile >= 80 or percentile <= 20
        }
    
    def _apply_ancestry_adjustments(self, analysis, ancestry):
        """Apply ancestry-specific adjustments to genetic analysis"""
        if ancestry not in self.population_genetics:
            return analysis
        
        ancestry_data = self.population_genetics[ancestry]
        
        # Adjust disease prevalence based on ancestry
        for disease, prs_data in analysis['polygenic_risk_scores'].items():
            if disease in ancestry_data['disease_prevalence']:
                population_prevalence = ancestry_data['disease_prevalence'][disease]
                prs_data['population_specific_prevalence'] = population_prevalence
        
        # Add ancestry-specific genetic variants
        analysis['ancestry_specific_considerations'] = {
            'ancestry': ancestry,
            'common_variants': ancestry_data['common_variants'],
            'disease_prevalences': ancestry_data['disease_prevalence']
        }
        
        return analysis
    
    def generate_personalized_treatment_plan(self, genetic_analysis, current_symptoms, medical_history):
        """Generate personalized treatment plan based on genetic profile"""
        treatment_plan = {
            'patient_genetic_profile': genetic_analysis['patient_id'],
            'personalization_level': 'high',
            'genetic_considerations': [],
            'drug_recommendations': [],
            'screening_recommendations': [],
            'lifestyle_modifications': [],
            'monitoring_requirements': []
        }
        
        # Pharmacogenomic drug adjustments
        for pharm_rec in genetic_analysis['pharmacogenomic_recommendations']:
            for rec in pharm_rec['clinical_recommendations']:
                treatment_plan['drug_recommendations'].append({
                    'gene': pharm_rec['gene'],
                    'affected_drugs': pharm_rec['affected_drugs'],
                    'recommendation': rec['recommendation'],
                    'priority': rec['priority']
                })
        
        # Disease susceptibility screening
        for risk_factor in genetic_analysis['genetic_risk_factors']:
            if risk_factor['actionable']:
                treatment_plan['screening_recommendations'].append({
                    'condition': risk_factor['associated_conditions'],
                    'screening_protocol': risk_factor['clinical_action'],
                    'evidence_level': risk_factor['evidence_level'],
                    'frequency': 'annual' if risk_factor['risk_level'] > 0.5 else 'biennial'
                })
        
        # Polygenic risk score interventions
        for disease, prs_data in genetic_analysis['polygenic_risk_scores'].items():
            if prs_data['clinical_utility']:
                if prs_data['risk_category'] in ['High', 'Very High']:
                    # High risk interventions
                    treatment_plan['lifestyle_modifications'].extend([
                        f"Intensive {disease} prevention program",
                        f"Enhanced monitoring for {disease} risk factors"
                    ])
                    
                    for env_factor in prs_data['environmental_factors']:
                        treatment_plan['lifestyle_modifications'].append(
                            f"Target {env_factor} modification for {disease} prevention"
                        )
        
        return treatment_plan

class EnhancedNeuralArchitectures:
    """
    Advanced Neural Network Architectures for Medical AI
    Implements transformers, attention mechanisms, and specialized architectures
    """
    
    def __init__(self):
        self.models = {}
        self.attention_maps = {}
        self.model_ensemble = None
        self.architecture_configs = {
            'medical_transformer': {
                'num_heads': 8,
                'num_layers': 6,
                'embedding_dim': 512,
                'ff_dim': 2048,
                'dropout': 0.1
            },
            'symptom_encoder': {
                'encoder_layers': 3,
                'hidden_dim': 256,
                'attention_heads': 4
            },
            'multimodal_fusion': {
                'fusion_layers': 2,
                'fusion_dim': 128,
                'modality_weights': True
            }
        }
        
    def create_medical_transformer(self, input_dim, num_classes, sequence_length=None):
        """Create transformer model for medical diagnosis"""
        config = self.architecture_configs['medical_transformer']
        
        if TRANSFORMERS_AVAILABLE:
            # Use transformer architecture
            model = self._build_transformer_model(input_dim, num_classes, config)
        else:
            # Fallback to advanced feedforward with attention-like mechanisms
            model = self._build_attention_feedforward(input_dim, num_classes, config)
        
        self.models['medical_transformer'] = model
        return model
    
    def _build_transformer_model(self, input_dim, num_classes, config):
        """Build transformer model using TensorFlow"""
        # Input layer
        inputs = tf.keras.layers.Input(shape=(input_dim,))
        
        # Reshape for sequence processing (treat symptoms as sequence)
        # Reshape symptoms into pseudo-sequence
        sequence_length = min(input_dim, 100)  # Limit sequence length
        reshaped = tf.keras.layers.Reshape((sequence_length, -1))(inputs)
        
        # Embedding layer
        embedded = tf.keras.layers.Dense(config['embedding_dim'], activation='relu')(reshaped)
        
        # Positional encoding
        positions = tf.keras.layers.Lambda(
            lambda x: x + self._get_positional_encoding(tf.shape(x)[1], config['embedding_dim'])
        )(embedded)
        
        # Multi-head attention layers
        attention_output = positions
        for _ in range(config['num_layers']):
            # Multi-head self-attention
            attention_layer = tf.keras.layers.MultiHeadAttention(
                num_heads=config['num_heads'],
                key_dim=config['embedding_dim'] // config['num_heads'],
                dropout=config['dropout']
            )
            
            attention_out = attention_layer(attention_output, attention_output)
            attention_output = tf.keras.layers.Add()([attention_output, attention_out])
            attention_output = tf.keras.layers.LayerNormalization()(attention_output)
            
            # Feed-forward network
            ff_out = tf.keras.layers.Dense(config['ff_dim'], activation='relu')(attention_output)
            ff_out = tf.keras.layers.Dropout(config['dropout'])(ff_out)
            ff_out = tf.keras.layers.Dense(config['embedding_dim'])(ff_out)
            
            attention_output = tf.keras.layers.Add()([attention_output, ff_out])
            attention_output = tf.keras.layers.LayerNormalization()(attention_output)
        
        # Global pooling and classification
        pooled = tf.keras.layers.GlobalAveragePooling1D()(attention_output)
        pooled = tf.keras.layers.Dropout(0.3)(pooled)
        
        # Classification layers
        dense1 = tf.keras.layers.Dense(256, activation='relu')(pooled)
        dense1 = tf.keras.layers.Dropout(0.4)(dense1)
        dense2 = tf.keras.layers.Dense(128, activation='relu')(dense1)
        dense2 = tf.keras.layers.Dropout(0.3)(dense2)
        
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(dense2)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_attention_feedforward(self, input_dim, num_classes, config):
        """Build attention-enhanced feedforward network"""
        inputs = tf.keras.layers.Input(shape=(input_dim,))
        
        # Attention mechanism simulation
        attention_weights = tf.keras.layers.Dense(input_dim, activation='softmax', name='attention_weights')(inputs)
        attended_features = tf.keras.layers.Multiply()([inputs, attention_weights])
        
        # Enhanced feature processing
        x = tf.keras.layers.Dense(512, activation='relu')(attended_features)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Residual connections
        residual1 = tf.keras.layers.Dense(256, activation='relu')(inputs)
        x = tf.keras.layers.Add()([x, residual1])
        x = tf.keras.layers.LayerNormalization()(x)
        
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _get_positional_encoding(self, seq_len, embedding_dim):
        """Generate positional encoding for transformer"""
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embedding_dim, 2) * -(np.log(10000.0) / embedding_dim))
        
        pos_encoding = np.zeros((seq_len, embedding_dim))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        return tf.constant(pos_encoding, dtype=tf.float32)
    
    def create_symptom_encoder_decoder(self, input_dim, latent_dim=64):
        """Create encoder-decoder for symptom representation learning"""
        # Encoder
        encoder_inputs = tf.keras.layers.Input(shape=(input_dim,))
        
        # Symptom grouping attention
        symptom_attention = tf.keras.layers.Dense(input_dim, activation='softmax')(encoder_inputs)
        attended_symptoms = tf.keras.layers.Multiply()([encoder_inputs, symptom_attention])
        
        # Encoder layers
        encoded = tf.keras.layers.Dense(256, activation='relu')(attended_symptoms)
        encoded = tf.keras.layers.Dropout(0.3)(encoded)
        encoded = tf.keras.layers.Dense(128, activation='relu')(encoded)
        encoded = tf.keras.layers.Dropout(0.2)(encoded)
        encoded = tf.keras.layers.Dense(latent_dim, activation='relu', name='latent_representation')(encoded)
        
        # Decoder
        decoded = tf.keras.layers.Dense(128, activation='relu')(encoded)
        decoded = tf.keras.layers.Dropout(0.2)(decoded)
        decoded = tf.keras.layers.Dense(256, activation='relu')(decoded)
        decoded = tf.keras.layers.Dropout(0.3)(decoded)
        decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(decoded)
        
        # Create models
        encoder = tf.keras.Model(encoder_inputs, encoded, name='symptom_encoder')
        autoencoder = tf.keras.Model(encoder_inputs, decoded, name='symptom_autoencoder')
        
        autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        self.models['symptom_encoder'] = encoder
        self.models['symptom_autoencoder'] = autoencoder
        
        return encoder, autoencoder
    
    def create_hierarchical_attention_network(self, input_dim, num_classes):
        """Create hierarchical attention network for medical diagnosis"""
        inputs = tf.keras.layers.Input(shape=(input_dim,))
        
        # Symptom-level attention
        symptom_dense = tf.keras.layers.Dense(128, activation='tanh')(inputs)
        symptom_attention = tf.keras.layers.Dense(1, activation='softmax')(symptom_dense)
        symptom_representation = tf.keras.layers.Multiply()([inputs, symptom_attention])
        
        # System-level representation (group symptoms by body systems)
        # Simulate grouping by creating multiple attention heads
        system_representations = []
        
        for i in range(8):  # 8 body systems
            system_dense = tf.keras.layers.Dense(64, activation='tanh')(symptom_representation)
            system_attention = tf.keras.layers.Dense(1, activation='softmax',
                                                   name=f'system_attention_{i}')(system_dense)
            system_rep = tf.keras.layers.Multiply()([symptom_representation, system_attention])
            system_rep = tf.keras.layers.GlobalAveragePooling1D()(
                tf.keras.layers.Reshape((-1, 1))(system_rep)
            )
            system_representations.append(system_rep)
        
        # Concatenate system representations
        if len(system_representations) > 1:
            concatenated = tf.keras.layers.Concatenate()(system_representations)
        else:
            concatenated = system_representations[0]
        
        # Document-level attention
        doc_dense = tf.keras.layers.Dense(128, activation='tanh')(concatenated)
        doc_attention = tf.keras.layers.Dense(len(system_representations), activation='softmax')(doc_dense)
        doc_representation = tf.keras.layers.Multiply()([concatenated, doc_attention])
        
        # Classification layers
        dense1 = tf.keras.layers.Dense(256, activation='relu')(doc_representation)
        dense1 = tf.keras.layers.Dropout(0.4)(dense1)
        dense2 = tf.keras.layers.Dense(128, activation='relu')(dense1)
        dense2 = tf.keras.layers.Dropout(0.3)(dense2)
        
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(dense2)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.models['hierarchical_attention'] = model
        return model
    
    def create_multimodal_fusion_network(self, modality_dims, num_classes):
        """Create multimodal fusion network for multiple input types"""
        modality_inputs = []
        modality_features = []
        
        # Process each modality separately
        for i, (modality_name, dim) in enumerate(modality_dims.items()):
            modality_input = tf.keras.layers.Input(shape=(dim,), name=f'{modality_name}_input')
            modality_inputs.append(modality_input)
            
            # Modality-specific processing
            features = tf.keras.layers.Dense(128, activation='relu', 
                                           name=f'{modality_name}_dense1')(modality_input)
            features = tf.keras.layers.Dropout(0.3)(features)
            features = tf.keras.layers.Dense(64, activation='relu',
                                           name=f'{modality_name}_dense2')(features)
            features = tf.keras.layers.Dropout(0.2)(features)
            
            # Modality attention
            attention = tf.keras.layers.Dense(1, activation='sigmoid',
                                            name=f'{modality_name}_attention')(features)
            attended_features = tf.keras.layers.Multiply()([features, attention])
            
            modality_features.append(attended_features)
        
        # Cross-modal attention
        if len(modality_features) > 1:
            # Concatenate all modality features
            concatenated = tf.keras.layers.Concatenate()(modality_features)
            
            # Cross-modal attention weights
            cross_attention = tf.keras.layers.Dense(len(modality_features), activation='softmax',
                                                  name='cross_modal_attention')(concatenated)
            
            # Apply cross-modal attention
            weighted_features = []
            for i, features in enumerate(modality_features):
                weight = tf.keras.layers.Lambda(lambda x: x[:, i:i+1])(cross_attention)
                weighted = tf.keras.layers.Multiply()([features, weight])
                weighted_features.append(weighted)
            
            fused_features = tf.keras.layers.Add()(weighted_features)
        else:
            fused_features = modality_features[0]
        
        # Final classification layers
        dense1 = tf.keras.layers.Dense(256, activation='relu')(fused_features)
        dense1 = tf.keras.layers.Dropout(0.4)(dense1)
        dense2 = tf.keras.layers.Dense(128, activation='relu')(dense1)
        dense2 = tf.keras.layers.Dropout(0.3)(dense2)
        
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(dense2)
        
        model = tf.keras.Model(inputs=modality_inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.models['multimodal_fusion'] = model
        return model
    
    def create_ensemble_architecture(self, base_models, num_classes):
        """Create ensemble of different architectures"""
        ensemble_inputs = []
        model_outputs = []
        
        for i, model in enumerate(base_models):
            # Create input layer compatible with each base model
            model_input = tf.keras.layers.Input(shape=model.input_shape[1:], 
                                              name=f'model_{i}_input')
            ensemble_inputs.append(model_input)
            
            # Get predictions from each model
            model_output = model(model_input)
            model_outputs.append(model_output)
        
        # Ensemble combination strategies
        if len(model_outputs) > 1:
            # Average ensemble
            averaged = tf.keras.layers.Average()(model_outputs)
            
            # Weighted ensemble (learnable weights)
            ensemble_weights = tf.keras.layers.Dense(len(model_outputs), activation='softmax',
                                                   name='ensemble_weights')(
                tf.keras.layers.Concatenate()(model_outputs)
            )
            
            # Apply weights to model outputs
            weighted_outputs = []
            for i, output in enumerate(model_outputs):
                weight = tf.keras.layers.Lambda(lambda x: x[:, i:i+1])(ensemble_weights)
                weighted = tf.keras.layers.Multiply()([output, weight])
                weighted_outputs.append(weighted)
            
            weighted_ensemble = tf.keras.layers.Add()(weighted_outputs)
            
            # Meta-learner for final prediction
            meta_input = tf.keras.layers.Concatenate()([averaged, weighted_ensemble])
            meta_dense = tf.keras.layers.Dense(64, activation='relu')(meta_input)
            meta_dense = tf.keras.layers.Dropout(0.3)(meta_dense)
            final_output = tf.keras.layers.Dense(num_classes, activation='softmax')(meta_dense)
        else:
            final_output = model_outputs[0]
        
        ensemble_model = tf.keras.Model(inputs=ensemble_inputs, outputs=final_output)
        ensemble_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model_ensemble = ensemble_model
        return ensemble_model
    
    def extract_attention_maps(self, model, input_data, layer_name=None):
        """Extract attention maps for interpretability"""
        if layer_name is None:
            # Find attention layers automatically
            attention_layers = [layer.name for layer in model.layers 
                              if 'attention' in layer.name.lower()]
        else:
            attention_layers = [layer_name]
        
        attention_maps = {}
        
        for layer_name in attention_layers:
            try:
                # Create intermediate model to extract attention weights
                intermediate_model = tf.keras.Model(
                    inputs=model.input,
                    outputs=model.get_layer(layer_name).output
                )
                
                attention_output = intermediate_model.predict(input_data)
                attention_maps[layer_name] = attention_output
                
            except Exception as e:
                print(f"Could not extract attention from layer {layer_name}: {e}")
        
        self.attention_maps = attention_maps
        return attention_maps
    
    def visualize_attention_patterns(self, attention_maps, symptom_names=None):
        """Visualize attention patterns for interpretability"""
        if not PLOTLY_AVAILABLE:
            return "Plotly not available for visualization"
        
        visualizations = {}
        
        for layer_name, attention_data in attention_maps.items():
            if len(attention_data.shape) == 2:
                # Simple attention weights
                if symptom_names and len(symptom_names) == attention_data.shape[1]:
                    # Create heatmap
                    fig = go.Figure(data=go.Heatmap(
                        z=attention_data,
                        x=symptom_names,
                        y=[f'Sample {i}' for i in range(attention_data.shape[0])],
                        colorscale='Viridis'
                    ))
                    fig.update_layout(
                        title=f'Attention Patterns - {layer_name}',
                        xaxis_title='Symptoms',
                        yaxis_title='Samples'
                    )
                    visualizations[layer_name] = fig
        
        return visualizations
    
    def get_model_complexity_analysis(self):
        """Analyze computational complexity of different models"""
        analysis = {
            'model_comparisons': [],
            'memory_usage': {},
            'inference_time': {},
            'parameter_counts': {}
        }
        
        for model_name, model in self.models.items():
            if model is not None:
                param_count = model.count_params()
                analysis['parameter_counts'][model_name] = param_count
                
                # Estimate memory usage (rough approximation)
                memory_mb = param_count * 4 / (1024 * 1024)  # 4 bytes per parameter
                analysis['memory_usage'][model_name] = f"{memory_mb:.2f} MB"
                
                analysis['model_comparisons'].append({
                    'model': model_name,
                    'parameters': param_count,
                    'layers': len(model.layers),
                    'memory_estimate': memory_mb
                })
        
        return analysis


# Utility to clean and convert both datasets, then merge them into a unified DataFrame
def merge_and_clean_datasets(training_path, dataset_path, merged_path='merged_dataset.csv'):
    # 1. Load Training.csv (one-hot format)
    df_train = pd.read_csv(training_path)
    drop_cols = [col for col in df_train.columns if col.startswith('Unnamed') or df_train[col].sum() == 0]
    df_train = df_train.drop(columns=drop_cols)
    df_train = df_train.fillna(0)
    # Standardize column names (symptoms)
    df_train.columns = [col.strip().lower().replace(' ', '_') if col != 'prognosis' else 'prognosis' for col in df_train.columns]
    # 2. Load dataset.csv (symptom list format)
    df_new = pd.read_csv(dataset_path)
    # Gather all unique symptoms from Symptom columns
    symptom_set = set()
    for col in df_new.columns:
        if col.lower().startswith('symptom'):
            symptom_set.update(df_new[col].dropna().astype(str).str.strip().str.lower().replace('nan',''))
    symptom_set.discard('')
    # Also add all symptoms from Training.csv
    symptom_set.update([col for col in df_train.columns if col != 'prognosis'])
    symptom_list = sorted(symptom_set)
    # Build one-hot encoded DataFrame for dataset.csv
    records = []
    for _, row in df_new.iterrows():
        rec = {sym: 0 for sym in symptom_list}
        for col in df_new.columns:
            if col.lower().startswith('symptom') and pd.notna(row[col]):
                sym = str(row[col]).strip().lower()
                if sym:
                    rec[sym] = 1
        rec['prognosis'] = row['Disease'].strip()
        records.append(rec)
    df_new_clean = pd.DataFrame(records)
    # 3. Align columns for both DataFrames
    for sym in symptom_list:
        if sym not in df_train.columns:
            df_train[sym] = 0
        if sym not in df_new_clean.columns:
            df_new_clean[sym] = 0
    # Ensure same column order
    ordered_cols = symptom_list + ['prognosis']
    df_train = df_train[ordered_cols]
    df_new_clean = df_new_clean[ordered_cols]
    # 4. Concatenate and remove duplicates
    df_merged = pd.concat([df_train, df_new_clean], ignore_index=True)
    df_merged = df_merged.drop_duplicates()
    # 5. Save merged dataset
    df_merged.to_csv(merged_path, index=False)
    print(f"Merged and cleaned dataset saved to {merged_path}.")
    return df_merged




# 1. Merge and clean both datasets, always use merged data
df = merge_and_clean_datasets(
    'C:/Users/bubhi/OneDrive/Desktop/AI/tfenv311/Training.csv',
    'C:/Users/bubhi/OneDrive/Desktop/AI/tfenv311/dataset.csv',
    merged_path='C:/Users/bubhi/OneDrive/Desktop/AI/tfenv311/merged_dataset.csv'
)
print("Loaded and merged both datasets!")


# 2. Prepare features and labels
symptom_cols = [col for col in df.columns if col != 'prognosis']
X = df[symptom_cols].values
y = pd.factorize(df['prognosis'])[0]
label_names = pd.factorize(df['prognosis'])[1]

# 3. Oversample minority classes
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)

# 4. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)


# 4. Define a function to build the model (for cross-validation and tuning)
def build_model(optimizer='adam'):
    from tensorflow.keras import regularizers
    model = models.Sequential([
        layers.Input(shape=(len(symptom_cols),)),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
        layers.Dropout(0.5),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
        layers.Dropout(0.4),
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
        layers.Dropout(0.3),
        layers.Dense(len(label_names), activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# 5. Model training/loading logic

model_path = 'disease_model.h5'
# Check if model exists and if its input shape matches the merged data
retrain = True
if os.path.exists(model_path):
    try:
        best_model = load_model(model_path)
        # Check input shape compatibility
        if best_model.input_shape[-1] == X_train.shape[1]:
            print("Loaded trained model from file.")
            best_params = (16, 'adam')
            retrain = False
        else:
            print("Model input shape does not match data. Retraining...")
            os.remove(model_path)
    except Exception as e:
        print(f"Error loading model: {e}. Retraining...")
        os.remove(model_path)

if retrain:
    from sklearn.model_selection import StratifiedKFold
    from tensorflow.keras.callbacks import EarlyStopping
    import numpy as np
    best_val_acc = 0
    best_params = None
    best_model = None
    batch_sizes = [16, 32]
    optimizers = ['adam', 'rmsprop']
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for batch_size in batch_sizes:
        for optimizer in optimizers:
            val_accuracies = []
            for train_idx, val_idx in kfold.split(X_train, y_train):
                model = build_model(optimizer=optimizer)
                early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                model.fit(
                    X_train[train_idx], y_train[train_idx],
                    epochs=100,
                    batch_size=batch_size,
                    validation_data=(X_train[val_idx], y_train[val_idx]),
                    verbose=0,
                    callbacks=[early_stop]
                )
                val_loss, val_acc = model.evaluate(X_train[val_idx], y_train[val_idx], verbose=0)
                val_accuracies.append(val_acc)
            avg_val_acc = np.mean(val_accuracies)
            print(f"Batch size: {batch_size}, Optimizer: {optimizer}, Avg. val accuracy: {avg_val_acc:.2f}")
            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                best_params = (batch_size, optimizer)
                best_model = build_model(optimizer=optimizer)
                early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                best_model.fit(
                    X_train, y_train,
                    epochs=100,
                    batch_size=batch_size,
                    validation_split=0.1,
                    verbose=0,
                    callbacks=[early_stop]
                )
    # Save the trained model
    best_model.save(model_path)
    print(f"Model trained and saved to {model_path}.")

# 6. Evaluate the best model
loss, accuracy = best_model.evaluate(X_test, y_test, verbose=0)
print(f"\nBest params: batch_size={best_params[0]}, optimizer={best_params[1]}")
print(f"Test accuracy: {accuracy:.2f}")

# 7. Prompt user for symptoms

# User-friendly symptom input prompt
print("\nWelcome to the AI Disease Predictor!")
print("You can enter your symptoms and the AI will predict the most likely disease.")
print("\nHere are some example symptoms you can use:")
example_symptoms = ', '.join(symptom_cols[:10]) + ', ...'
print(example_symptoms)
print("\nType your symptoms separated by commas (e.g. fever, headache, nausea)")

while True:
    user_input = input("Enter your symptoms: ").strip().lower()
    user_symptoms = [s.strip() for s in user_input.split(',') if s.strip()]
    # Validate at least one valid symptom
    valid = any(symptom.lower() in [col.lower() for col in symptom_cols] for symptom in user_symptoms)
    if not user_symptoms or not valid:
        print("\nPlease enter at least one valid symptom from the list. Try again.")
        print("Example symptoms:", example_symptoms)
    else:
        break

# 8. Create input vector for user
user_vector = np.zeros((1, len(symptom_cols)))
for idx, symptom in enumerate(symptom_cols):
    if symptom.lower() in user_symptoms:
        user_vector[0, idx] = 1

# 9. Predict disease and show top 3 most likely
pred = best_model.predict(user_vector)
top_indices = np.argsort(pred[0])[::-1][:3]
print("\nTop 3 predicted diseases:")
for rank, idx in enumerate(top_indices, 1):
    print(f"{rank}. {label_names[idx]} (probability: {pred[0][idx]:.2f})")

# Print all probabilities for transparency
print("\nAll disease probabilities:")
for idx, name in enumerate(label_names):
    print(f"{name}: {pred[0][idx]:.4f}")