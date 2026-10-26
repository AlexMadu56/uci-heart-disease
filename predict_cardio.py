#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = Path(__file__).with_name("modelo_cardio.joblib")
OPTIONAL_FEATURES = {"vasos_principales_observados", "resultado_prueba_talio"}
INTEGER_FEATURES = {
    "edad",
    "presion_arterial_reposo",
    "colesterol_serico",
    "frecuencia_cardiaca_maxima",
}
NONNEGATIVE_FEATURES = INTEGER_FEATURES
CATEGORICAL_VALUES = {
    "sexo": {0, 1},
    "tipo_dolor_toracico": {1, 2, 3, 4},
    "glucosa_ayunas_alta": {0, 1},
    "resultado_electrocardiograma_reposo": {0, 1, 2},
    "angina_inducida_ejercicio": {0, 1},
    "pendiente_segmento_st": {1, 2, 3},
    "vasos_principales_observados": {0, 1, 2, 3},
    "resultado_prueba_talio": {3, 6, 7},
}
CLASS_LABELS = {
    0: "Sin estrechamiento angiográfico significativo registrado",
    1: "Estrechamiento angiográfico significativo registrado",
}


def load_bundle(model_path: str | Path = MODEL_PATH) -> dict[str, Any]:
    """Carga y comprueba el bundle serializado del notebook."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
    bundle = joblib.load(model_path)
    required_keys = {
        "format_version",
        "model",
        "feature_columns",
        "numeric_ranges",
        "numeric_defaults",
    }
    if not isinstance(bundle, dict) or not required_keys.issubset(bundle):
        raise ValueError("El archivo no parece contener el bundle del modelo esperado.")
    if bundle["format_version"] != 1:
        raise ValueError("La versión del bundle del modelo no es compatible.")
    if not isinstance(bundle["feature_columns"], list):
        raise ValueError("El bundle no contiene una lista válida de variables.")
    model = bundle["model"]
    if (
        not hasattr(model, "predict")
        or not hasattr(model, "predict_proba")
        or not hasattr(model, "classes_")
    ):
        raise ValueError("El bundle no contiene un modelo con predict y predict_proba.")
    if set(model.classes_) != {0, 1}:
        raise ValueError("El modelo debe contener las clases 0 y 1.")
    return bundle


def prepare_input(row: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    if not isinstance(row, dict):
        raise TypeError("La entrada debe ser un objeto JSON con variables y valores.")

    unknown = set(row) - set(feature_columns)
    if unknown:
        raise ValueError(f"Variables no reconocidas: {', '.join(sorted(unknown))}.")

    missing = set(feature_columns) - set(row) - OPTIONAL_FEATURES
    if missing:
        raise ValueError(f"Faltan variables obligatorias: {', '.join(sorted(missing))}.")

    normalized: dict[str, float] = {}
    for column in feature_columns:
        value = row.get(column)
        if value is None:
            normalized[column] = np.nan
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ValueError(
                f"El valor de «{column}» debe ser numérico, no texto ni booleano."
            )
        numeric_value = float(value)
        if not np.isfinite(numeric_value):
            raise ValueError(f"El valor de «{column}» debe ser finito.")
        if column in INTEGER_FEATURES and not numeric_value.is_integer():
            raise ValueError(f"El valor de «{column}» debe ser un número entero.")
        if column in NONNEGATIVE_FEATURES and numeric_value < 0:
            raise ValueError(f"El valor de «{column}» no puede ser negativo.")
        if column in CATEGORICAL_VALUES and numeric_value not in CATEGORICAL_VALUES[column]:
            valid_values = ", ".join(map(str, sorted(CATEGORICAL_VALUES[column])))
            raise ValueError(
                f"Valor no válido para «{column}»: {numeric_value:g}. "
                f"Valores admitidos: {valid_values}."
            )
        normalized[column] = numeric_value
    return pd.DataFrame([normalized], columns=feature_columns)


def predict_from_dict(row: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    features = bundle["feature_columns"]
    X = prepare_input(row, features)
    model = bundle["model"]
    prediction = int(model.predict(X)[0])
    class_values = [int(value) for value in model.classes_]
    class_probabilities = model.predict_proba(X)[0]
    if len(class_probabilities) != len(class_values):
        raise ValueError("El modelo devolvió probabilidades incompatibles con sus clases.")
    if not np.isfinite(class_probabilities).all() or not np.isclose(
        class_probabilities.sum(), 1.0
    ):
        raise ValueError("El modelo devolvió una distribución de probabilidades no válida.")
    probabilities = {
        class_value: float(probability)
        for class_value, probability in zip(class_values, class_probabilities)
    }
    return {
        "prediction": prediction,
        "label": CLASS_LABELS[prediction],
        "probabilities": probabilities,
        "predicted_probability": probabilities[prediction],
        "features": X.iloc[0].to_dict(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Predicción de enfermedad cardíaca usando el modelo exportado desde el notebook.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--json", help="Objeto JSON con los valores de entrada.")
    source.add_argument("--file", type=Path, help="Ruta a un archivo JSON con los valores.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Ruta al bundle del modelo.")
    return parser.parse_args()


def main():
    args = parse_args()
    bundle = load_bundle(args.model)

    if args.json is not None:
        payload = json.loads(args.json)
    elif args.file:
        payload = json.loads(args.file.read_text(encoding="utf-8"))
    else:
        example_path = Path(__file__).with_name("paciente_ejemplo_cardio.json")
        payload = json.loads(example_path.read_text(encoding="utf-8"))

    result = predict_from_dict(payload, bundle)

    print("Clasificación experimental de la etiqueta UCI `num`:")
    print(f"  Etiqueta: {result['label']}")
    print("  Probabilidades estimadas por clase del conjunto:")
    for class_value, probability in result["probabilities"].items():
        print(f"    {CLASS_LABELS[class_value]}: {probability:.1%}")
    print("\nValores de entrada (los datos no conocidos quedan imputados):")
    for key in bundle["feature_columns"]:
        print(f"  {key}: {result['features'][key]}")
    print(
        "\nUso educativo únicamente: no es un diagnóstico, no descarta enfermedad "
        "y no predice eventos futuros."
    )


if __name__ == "__main__":
    main()
