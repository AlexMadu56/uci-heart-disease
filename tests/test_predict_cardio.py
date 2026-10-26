import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from predict_cardio import load_bundle, predict_from_dict, prepare_input


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bundle():
    return load_bundle(PROJECT_ROOT / "modelo_cardio.joblib")


@pytest.fixture
def example():
    return json.loads(
        (PROJECT_ROOT / "paciente_ejemplo_cardio.json").read_text(encoding="utf-8")
    )


def test_example_returns_a_valid_binary_classification(bundle, example):
    result = predict_from_dict(example, bundle)

    assert result["prediction"] in {0, 1}
    assert result["label"]
    assert set(result["features"]) == set(bundle["feature_columns"])
    assert set(result["probabilities"]) == set(bundle["model"].classes_)
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    assert result["predicted_probability"] == pytest.approx(
        result["probabilities"][result["prediction"]]
    )


def test_exported_model_contains_training_fitted_standard_scaler(bundle):
    numeric_pipeline = bundle["model"].named_steps["preprocesamiento"].named_transformers_[
        "numericas"
    ]
    scaler = numeric_pipeline.named_steps["escalador"]

    assert len(scaler.mean_) == 5
    assert len(scaler.scale_) == 5


def test_optional_missing_features_are_preserved_for_pipeline_imputation(bundle, example):
    example.pop("vasos_principales_observados")
    example.pop("resultado_prueba_talio")

    prepared = prepare_input(example, bundle["feature_columns"])

    assert prepared[["vasos_principales_observados", "resultado_prueba_talio"]].isna().all().all()


def test_unknown_values_for_all_features_are_imputed_by_the_model(bundle, example):
    unknown_values = dict.fromkeys(bundle["feature_columns"])

    prepared = prepare_input(unknown_values, bundle["feature_columns"])
    result = predict_from_dict(unknown_values, bundle)

    assert prepared.isna().all().all()
    assert result["prediction"] in {0, 1}
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)


def test_missing_required_feature_is_rejected(bundle, example):
    example.pop("edad")

    with pytest.raises(ValueError, match="Faltan variables obligatorias"):
        predict_from_dict(example, bundle)


def test_unknown_feature_is_rejected(bundle, example):
    example["nombre"] = "No corresponde"

    with pytest.raises(ValueError, match="Variables no reconocidas"):
        predict_from_dict(example, bundle)


def test_invalid_categorical_code_is_rejected(bundle, example):
    example["sexo"] = 2

    with pytest.raises(ValueError, match="Valor no válido"):
        predict_from_dict(example, bundle)


def test_non_finite_value_is_rejected(bundle, example):
    example["edad"] = float("nan")

    with pytest.raises(ValueError, match="debe ser finito"):
        predict_from_dict(example, bundle)


def test_numeric_text_is_rejected(bundle, example):
    example["edad"] = "52"

    with pytest.raises(ValueError, match="no texto ni booleano"):
        predict_from_dict(example, bundle)


def test_out_of_range_numeric_input_is_not_silently_clipped(bundle, example):
    example["edad"] = 200

    prepared = prepare_input(example, bundle["feature_columns"])

    assert prepared.loc[0, "edad"] == 200


def test_integer_measurement_rejects_fractional_values(bundle, example):
    example["edad"] = 52.5

    with pytest.raises(ValueError, match="debe ser un número entero"):
        predict_from_dict(example, bundle)


def test_nonnegative_measurement_rejects_negative_values(bundle, example):
    example["colesterol_serico"] = -1

    with pytest.raises(ValueError, match="no puede ser negativo"):
        predict_from_dict(example, bundle)


def test_streamlit_form_explains_all_fields_and_uses_descriptive_result(bundle):
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()

    assert not app.exception
    widgets = [*app.text_input, *app.selectbox]
    assert len(widgets) == 13
    assert all(widget.help for widget in widgets)
    assert all("Nombre técnico en español:" in widget.help for widget in widgets)
    assert all("Explicación técnica:" in widget.help for widget in widgets)
    assert {widget.label for widget in app.text_input} == {
        "Edad (años)",
        "Presión arterial en reposo (mm Hg)",
        "Nivel de colesterol (mg/dl)",
        "Frecuencia cardíaca máxima (latidos/min)",
        "Cambios en la señal cardíaca durante el esfuerzo (valor)",
    }
    assert {widget.label for widget in app.selectbox} == {
        "Sexo registrado",
        "Tipo de molestia en el pecho",
        "¿Glucosa elevada en ayunas?",
        "¿Molestias en el pecho durante el ejercicio?",
        "Resultado del electrocardiograma",
        "Vasos principales observados",
        "Forma de la señal cardíaca durante el esfuerzo",
        "Resultado de la prueba de perfusión cardíaca",
    }
    selects = {widget.key: widget for widget in app.selectbox}
    assert {
        key: widget.options for key, widget in selects.items()
    } == {
        "entrada_sexo": [
            "Femenino",
            "Masculino",
            "No lo sé",
        ],
        "entrada_tipo_dolor_toracico": [
            "Molestia típica",
            "Molestia atípica",
            "Otro tipo de dolor",
            "Sin molestias torácicas registradas",
            "No lo sé",
        ],
        "entrada_glucosa_ayunas_alta": [
            "No: 120 mg/dl o menos",
            "Sí: más de 120 mg/dl",
            "No lo sé",
        ],
        "entrada_angina_inducida_ejercicio": [
            "No",
            "Sí",
            "No lo sé",
        ],
        "entrada_resultado_electrocardiograma_reposo": [
            "Sin alteraciones registradas",
            "Cambios en el trazado eléctrico",
            "Posible aumento del tamaño del ventrículo izquierdo",
            "No lo sé",
        ],
        "entrada_vasos_principales_observados": [
            "Ningún vaso principal observado",
            "Un vaso principal",
            "Dos vasos principales",
            "Tres vasos principales",
            "No lo sé",
        ],
        "entrada_pendiente_segmento_st": [
            "Ascendente",
            "Plana",
            "Descendente",
            "No lo sé",
        ],
        "entrada_resultado_prueba_talio": [
            "Sin alteración registrada",
            "Cambio persistente",
            "Cambio reversible",
            "No lo sé",
        ],
    }
    assert selects["entrada_tipo_dolor_toracico"].options[:4] == [
        "Molestia típica",
        "Molestia atípica",
        "Otro tipo de dolor",
        "Sin molestias torácicas registradas",
    ]
    assert selects["entrada_tipo_dolor_toracico"].value == 1
    assert selects["entrada_angina_inducida_ejercicio"].options[:2] == ["No", "Sí"]
    assert selects["entrada_angina_inducida_ejercicio"].value == 0
    assert all(
        "código UCI" not in str(option)
        for widget in app.selectbox
        for option in widget.options
        if option is not None
    )

    submitted = app.button[0].click().run()

    assert not app.exception
    assert len(app.metric) == 2
    input_values = {
        widget.key.removeprefix("entrada_"): (
            float(widget.value)
            if widget.key.startswith("entrada_edad")
            or widget.key.startswith("entrada_presion_")
            or widget.key.startswith("entrada_colesterol_")
            or widget.key.startswith("entrada_frecuencia_")
            or widget.key.startswith("entrada_depresion_")
            else widget.value
        )
        for widget in [*submitted.text_input, *submitted.selectbox]
    }
    expected = predict_from_dict(input_values, bundle)
    assert expected["prediction"] == 0
    result_markup = "\n".join(item.value for item in app.markdown)
    assert "Resultado de la clasificación" in result_markup
    assert "Sin indicios de enfermedad cardíaca en la clasificación del modelo" in result_markup
    assert "estrechamiento mayor del 50 %" in result_markup
    assert "No confirma ni descarta una enfermedad" in result_markup
    assert "no representan riesgo clínico individual" in result_markup
    assert submitted.metric[0].value == f"{expected['probabilities'][1]:.1%}"
    assert submitted.metric[1].value == f"{expected['probabilities'][0]:.1%}"
    assert not any("📊" in item.value or "❤️" in item.value for item in app.markdown)


def test_form_can_mark_every_feature_unknown_and_apply_imputation(bundle):
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()

    for text_input in app.text_input:
        text_input.set_value("")
    for selectbox in app.selectbox:
        selectbox.select_index(len(selectbox.options) - 1)
    submitted = app.button[0].click().run()

    assert not submitted.exception
    assert len(submitted.metric) == 2
    assert any(
        "mediana para variables numéricas y categoría más frecuente "
        "para variables categóricas"
        in item.value
        for item in submitted.info
    )
    unknown_values = dict.fromkeys(bundle["feature_columns"])
    result = predict_from_dict(unknown_values, bundle)
    assert submitted.metric[0].value == f"{result['probabilities'][1]:.1%}"
    assert submitted.metric[1].value == f"{result['probabilities'][0]:.1%}"


def test_invalid_numeric_text_is_warned_and_imputed(bundle):
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()
    fields = {widget.key: widget for widget in app.text_input}
    fields["entrada_edad"].set_value("letras")
    fields["entrada_depresion_st_ejercicio"].set_value("3,9")
    submitted = app.button[0].click().run()

    assert not submitted.exception
    assert any(
        "Edad" in warning.value and "se imputaron" in warning.value
        for warning in submitted.warning
    )
    values = json.loads(
        (PROJECT_ROOT / "paciente_ejemplo_cardio.json").read_text(encoding="utf-8")
    )
    values["edad"] = None
    values["depresion_st_ejercicio"] = 3.9
    expected = predict_from_dict(values, bundle)
    assert submitted.metric[0].value == f"{expected['probabilities'][1]:.1%}"
    assert submitted.metric[1].value == f"{expected['probabilities'][0]:.1%}"


def test_positive_class_has_descriptive_non_diagnostic_result():
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()
    bundle = load_bundle(PROJECT_ROOT / "modelo_cardio.joblib")
    widget_by_key = {widget.key: widget for widget in app.selectbox}
    text_input_by_key = {widget.key: widget for widget in app.text_input}
    text_values = {
        "entrada_edad": 41,
        "entrada_presion_arterial_reposo": 115,
        "entrada_colesterol_serico": 289,
        "entrada_frecuencia_cardiaca_maxima": 145,
        "entrada_depresion_st_ejercicio": 3.9,
    }
    for key, value in text_values.items():
        text_input_by_key[key].set_value(str(value))
    widget_by_key["entrada_sexo"].set_value(1)
    widget_by_key["entrada_tipo_dolor_toracico"].set_value(3)
    widget_by_key["entrada_glucosa_ayunas_alta"].set_value(0)
    widget_by_key["entrada_resultado_electrocardiograma_reposo"].set_value(2)
    widget_by_key["entrada_angina_inducida_ejercicio"].set_value(1)
    widget_by_key["entrada_pendiente_segmento_st"].set_value(3)
    widget_by_key["entrada_vasos_principales_observados"].set_value(3)
    widget_by_key["entrada_resultado_prueba_talio"].set_value(7)
    submitted = app.button[0].click().run()
    inputs = {
        widget.key.removeprefix("entrada_"): (
            float(widget.value)
            if widget.key.startswith("entrada_edad")
            or widget.key.startswith("entrada_presion_")
            or widget.key.startswith("entrada_colesterol_")
            or widget.key.startswith("entrada_frecuencia_")
            or widget.key.startswith("entrada_depresion_")
            else widget.value
        )
        for widget in [*submitted.text_input, *submitted.selectbox]
    }
    expected = predict_from_dict(inputs, bundle)
    assert expected["prediction"] == 1

    result_markup = "\n".join(item.value for item in submitted.markdown)
    assert "Posibles indicios de enfermedad cardíaca registrada" in result_markup
    assert "No confirma ni descarta una enfermedad" in result_markup
    assert len(submitted.metric) == 2
    assert submitted.metric[0].value == f"{expected['probabilities'][1]:.1%}"
    assert submitted.metric[1].value == f"{expected['probabilities'][0]:.1%}"
    assert sum(expected["probabilities"].values()) == pytest.approx(1.0)
