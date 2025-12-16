"""
Script completo para entrenar y evaluar modelos de credit risk.
Usa el PreprocessingPipeline y entrena múltiples modelos comparándolos.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import MODEL_FILE, MODEL_DIR
from src.data_utils import get_datasets, get_feature_target, get_train_val_sets
from src.preprocessing import PreprocessingPipeline, TARGET_COL

# Configuración
RANDOM_STATE = 42


def evaluate_model(
    model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, model_name: str
) -> Dict[str, Any]:
    """
    Evalúa un modelo con múltiples métricas.

    Args:
        model: Modelo entrenado
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        X_val: Features de validación
        y_val: Target de validación
        model_name: Nombre del modelo

    Returns:
        Diccionario con métricas del modelo
    """
    # Predicciones
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]

    # Métricas
    metrics = {
        "model_name": model_name,
        "train": {
            "roc_auc": roc_auc_score(y_train, y_train_proba),
            "f1": f1_score(y_train, y_train_pred),
            "precision": precision_score(y_train, y_train_pred),
            "recall": recall_score(y_train, y_train_pred),
        },
        "val": {
            "roc_auc": roc_auc_score(y_val, y_val_proba),
            "f1": f1_score(y_val, y_val_pred),
            "precision": precision_score(y_val, y_val_pred),
            "recall": recall_score(y_val, y_val_pred),
        },
    }

    return metrics


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    pipeline: PreprocessingPipeline,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Entrena múltiples modelos y los evalúa.

    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        X_val: Features de validación
        y_val: Target de validación
        X_test: Features de test
        y_test: Target de test
        pipeline: Pipeline de preprocessing

    Returns:
        Tupla con (mejor modelo, mejores métricas)
    """
    # Aplicar preprocessing
    print("\n" + "=" * 60)
    print("APPLYING PREPROCESSING PIPELINE")
    print("=" * 60)
    print(f"Train input: {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
    print(f"Validation input: {X_val.shape[0]:,} rows × {X_val.shape[1]} features")
    print(f"Test input: {X_test.shape[0]:,} rows × {X_test.shape[1]} features")
    print(f"\n[INFO] Applying ALL preprocessing steps: cleaning -> outliers -> feature engineering -> encoding -> scaling...")
    
    import time
    start_time = time.time()
    X_train_processed, X_val_processed, X_test_processed = pipeline.fit_transform(
        X_train, X_val, X_test
    )
    preprocessing_time = time.time() - start_time
    
    print(f"\n[OK] Preprocessing completed in {preprocessing_time:.2f} seconds ({preprocessing_time/60:.2f} minutes)")
    print(f"\nAfter preprocessing:")
    print(f"  Train: {X_train_processed.shape[0]:,} rows × {X_train_processed.shape[1]} features")
    print(f"  Validation: {X_val_processed.shape[0]:,} rows × {X_val_processed.shape[1]} features")
    print(f"  Test: {X_test_processed.shape[0]:,} rows × {X_test_processed.shape[1]} features")
    
    # Verificar calidad de datos procesados
    print(f"\nData quality check:")
    print(f"  Train stats - Min: {X_train_processed.min():.4f}, Max: {X_train_processed.max():.4f}")
    print(f"                Mean: {X_train_processed.mean():.4f}, Std: {X_train_processed.std():.4f}")
    print(f"  NaN count: {np.isnan(X_train_processed).sum()}")
    print(f"  Inf count: {np.isinf(X_train_processed).sum()}")
    
    if np.isnan(X_train_processed).sum() > 0 or np.isinf(X_train_processed).sum() > 0:
        print("  [WARNING] NaN or Inf values detected!")
    else:
        print("  [OK] No NaN or Inf values - data is clean")

    # Convertir targets a numpy
    y_train_np = y_train.values
    y_val_np = y_val.values
    y_test_np = y_test.values

    # Calcular sample_weight para Gradient Boosting (no tiene class_weight)
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight('balanced', y_train_np)
    
    # Modelos a entrenar (con mejores hiperparámetros para credit risk)
    models = {
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=3000,  # Más iteraciones
            class_weight="balanced",  # Maneja desbalanceo
            solver="lbfgs",
            C=1.0,  # Regularización moderada (mejor que 0.1)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,  # Más árboles
            max_depth=15,  # Profundidad moderada (evitar overfitting)
            min_samples_split=10,  # Más restrictivo
            min_samples_leaf=5,  # Más restrictivo
            random_state=RANDOM_STATE,
            class_weight="balanced",  # Maneja desbalanceo
            n_jobs=-1,
            max_features="sqrt",  # Mejor para credit risk
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300,  # Más árboles para mejor performance
            max_depth=5,  # Profundidad moderada para evitar overfitting
            learning_rate=0.03,  # Learning rate más bajo para mejor convergencia
            random_state=RANDOM_STATE,
            subsample=0.8,
            min_samples_split=20,  # Más restrictivo para evitar overfitting
            min_samples_leaf=10,  # Más restrictivo
            # NOTA: GradientBoosting no tiene class_weight, usamos sample_weight manualmente
        ),
    }

    results = {}
    best_model = None
    best_score = 0
    best_model_name = None

    print("\n" + "=" * 60)
    print("Training and evaluating models...")
    print("=" * 60)

    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        print(f"{'='*60}")
        print(f"  Training on: {X_train_processed.shape[0]:,} samples, {X_train_processed.shape[1]} features")
        print(f"  Target distribution: Class 0: {np.sum(y_train_np==0):,}, Class 1: {np.sum(y_train_np==1):,}")
        
        try:
            import time
            train_start = time.time()
            print(f"  Starting training...")
            
            # Para Gradient Boosting, usar sample_weight
            if name == "Gradient Boosting":
                model.fit(X_train_processed, y_train_np, sample_weight=sample_weights)
            else:
                model.fit(X_train_processed, y_train_np)
            
            train_time = time.time() - train_start
            print(f"  [OK] Training completed in {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
            metrics = evaluate_model(
                model, X_train_processed, y_train_np, X_val_processed, y_val_np, name
            )
            results[name] = {"model": model, "metrics": metrics}

            # Usar ROC-AUC como métrica principal para seleccionar el mejor modelo
            val_roc_auc = metrics["val"]["roc_auc"]
            print(f"  Validation ROC-AUC: {val_roc_auc:.4f}")
            print(f"  Validation F1: {metrics['val']['f1']:.4f}")
            print(f"  Validation Precision: {metrics['val']['precision']:.4f}")
            print(f"  Validation Recall: {metrics['val']['recall']:.4f}")

            if val_roc_auc > best_score:
                best_score = val_roc_auc
                best_model = model
                best_model_name = name

        except Exception as e:
            print(f"  Error training {name}: {str(e)}")
            continue

    # Evaluar mejor modelo en test y calcular threshold óptimo
    if best_model is not None:
        print("\n" + "=" * 60)
        print(f"Evaluating best model ({best_model_name}) on test set...")
        print("=" * 60)

        y_test_pred = best_model.predict(X_test_processed)
        y_test_proba = best_model.predict_proba(X_test_processed)[:, 1]
        
        # Calcular threshold óptimo usando ROC curve (Youden's J statistic)
        fpr, tpr, thresholds = roc_curve(y_test_np, y_test_proba)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]
        
        print(f"\n[INFO] Optimal Threshold Calculation:")
        print(f"   Optimal threshold (Youden's J): {optimal_threshold:.4f}")
        print(f"   At this threshold:")
        print(f"     TPR (Recall): {tpr[optimal_idx]:.4f}")
        print(f"     FPR: {fpr[optimal_idx]:.4f}")
        print(f"     Youden's J: {youden_j[optimal_idx]:.4f}")
        
        # Predicciones con threshold óptimo
        y_test_pred_optimal = (y_test_proba >= optimal_threshold).astype(int)
        
        test_metrics = {
            "roc_auc": roc_auc_score(y_test_np, y_test_proba),
            "f1": f1_score(y_test_np, y_test_pred),
            "precision": precision_score(y_test_np, y_test_pred),
            "recall": recall_score(y_test_np, y_test_pred),
        }
        
        # Métricas con threshold óptimo
        test_metrics_optimal = {
            "f1": f1_score(y_test_np, y_test_pred_optimal),
            "precision": precision_score(y_test_np, y_test_pred_optimal),
            "recall": recall_score(y_test_np, y_test_pred_optimal),
        }

        print("\nTest Set Metrics (threshold=0.5):")
        print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
        print(f"  F1-Score: {test_metrics['f1']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        
        print("\nTest Set Metrics (optimal threshold):")
        print(f"  F1-Score: {test_metrics_optimal['f1']:.4f}")
        print(f"  Precision: {test_metrics_optimal['precision']:.4f}")
        print(f"  Recall: {test_metrics_optimal['recall']:.4f}")

        print("\nClassification Report (threshold=0.5):")
        print(classification_report(y_test_np, y_test_pred))
        
        print("\nClassification Report (optimal threshold):")
        print(classification_report(y_test_np, y_test_pred_optimal))

        print("\nConfusion Matrix (threshold=0.5):")
        cm = confusion_matrix(y_test_np, y_test_pred)
        print(cm)
        
        print("\nConfusion Matrix (optimal threshold):")
        cm_optimal = confusion_matrix(y_test_np, y_test_pred_optimal)
        print(cm_optimal)

        best_metrics = {
            "model_name": best_model_name,
            "val_metrics": results[best_model_name]["metrics"]["val"],
            "test_metrics": test_metrics,
            "test_metrics_optimal": test_metrics_optimal,
            "optimal_threshold": float(optimal_threshold),
        }

        return best_model, best_metrics

    else:
        raise ValueError("No model was successfully trained")


def save_model_and_pipeline(
    model: Any, pipeline: PreprocessingPipeline, model_name: str, metrics: Dict[str, Any]
) -> None:
    """
    Guarda el modelo, pipeline y métricas.

    Args:
        model: Modelo entrenado
        pipeline: Pipeline de preprocessing
        model_name: Nombre del modelo
        metrics: Métricas del modelo
    """
    # Los directorios ya se crean en config.py

    # Guardar modelo
    model_path = MODEL_DIR / "model.joblib"
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

    # Guardar pipeline
    from src.config import PREPROCESSOR_FILE
    pipeline.save()
    print(f"Pipeline saved to: {PREPROCESSOR_FILE}")

    # Guardar métricas
    metrics_path = MODEL_DIR / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Best Model: {metrics['model_name']}\n\n")
        f.write("Validation Metrics:\n")
        for metric, value in metrics["val_metrics"].items():
            f.write(f"  {metric}: {value:.4f}\n")
        f.write("\nTest Metrics (threshold=0.5):\n")
        for metric, value in metrics["test_metrics"].items():
            f.write(f"  {metric}: {value:.4f}\n")
        if "test_metrics_optimal" in metrics:
            f.write("\nTest Metrics (optimal threshold):\n")
            for metric, value in metrics["test_metrics_optimal"].items():
                f.write(f"  {metric}: {value:.4f}\n")
        if "optimal_threshold" in metrics:
            f.write(f"\nOptimal Threshold: {metrics['optimal_threshold']:.4f}\n")
    print(f"Metrics saved to: {metrics_path}")
    
    # Guardar threshold óptimo en un archivo separado para la API
    if "optimal_threshold" in metrics:
        threshold_path = MODEL_DIR / "optimal_threshold.txt"
        with open(threshold_path, "w") as f:
            f.write(str(metrics["optimal_threshold"]))
        print(f"Optimal threshold saved to: {threshold_path}")


def main():
    """Función principal de entrenamiento."""
    print("=" * 60)
    print("Credit Risk Model Training")
    print("=" * 60)

    # 1. Cargar datos
    print("\n1. Loading datasets...")
    app_train, app_test, column_descriptions = get_datasets()
    print(f"   Train shape: {app_train.shape}")
    
    # Si no hay archivo de test, dividir el training en train/val/test
    if app_test is None:
        print("   Test file not found. Will split training data into train/val/test")
        from sklearn.model_selection import train_test_split
        
        # Separar features y target del dataset completo
        X_full = app_train.drop(columns=[TARGET_COL])
        y_full = app_train[TARGET_COL]
        
        # Primero dividir en train (70%) y temp (30%)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_full, y_full, test_size=0.3, random_state=RANDOM_STATE, shuffle=True, stratify=y_full
        )
        
        # Luego dividir temp en val (15%) y test (15%)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, shuffle=True, stratify=y_temp
        )
        
        print(f"   Split: Train ({X_train.shape[0]}), Val ({X_val.shape[0]}), Test ({X_test.shape[0]})")
        print(f"   y_train distribution:\n{y_train.value_counts(normalize=True)}")
    else:
        print(f"   Test shape: {app_test.shape}")
        # 2. Separar features y target
        print("\n2. Separating features and target...")
        X_train_full, y_train_full, X_test, y_test = get_feature_target(app_train, app_test)
        print(f"   X_train_full shape: {X_train_full.shape}")
        print(f"   y_train_full distribution:\n{y_train_full.value_counts(normalize=True)}")

        # 3. Separar train y validation
        print("\n3. Splitting train into train/validation...")
        X_train, X_val, y_train, y_val = get_train_val_sets(X_train_full, y_train_full)
        print(f"   Train shape: {X_train.shape}")
        print(f"   Validation shape: {X_val.shape}")

    # 4. Crear pipeline
    print("\n4. Creating preprocessing pipeline...")
    pipeline = PreprocessingPipeline(low_cardinality_threshold=20)

    # 5. Entrenar modelos
    print("\n5. Training models...")
    best_model, best_metrics = train_models(
        X_train, y_train, X_val, y_val, X_test, y_test, pipeline
    )

    # 6. Guardar mejor modelo y pipeline
    print("\n6. Saving best model and pipeline...")
    save_model_and_pipeline(best_model, pipeline, best_metrics["model_name"], best_metrics)

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
