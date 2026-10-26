import json
import math
from pathlib import Path

import streamlit as st

from predict_cardio import (
    CATEGORICAL_VALUES,
    INTEGER_FEATURES,
    load_bundle,
    predict_from_dict,
)


ARCHIVO_EJEMPLO = Path(__file__).with_name("paciente_ejemplo_cardio.json")

ETIQUETAS_CATEGORIAS = {
    "sexo": {0: "Femenino", 1: "Masculino"},
    "tipo_dolor_toracico": {
        1: "Molestia típica",
        2: "Molestia atípica",
        3: "Otro tipo de dolor",
        4: "Sin molestias torácicas registradas",
    },
    "glucosa_ayunas_alta": {
        0: "No: 120 mg/dl o menos",
        1: "Sí: más de 120 mg/dl",
    },
    "resultado_electrocardiograma_reposo": {
        0: "Sin alteraciones registradas",
        1: "Cambios en el trazado eléctrico",
        2: "Posible aumento del tamaño del ventrículo izquierdo",
    },
    "angina_inducida_ejercicio": {
        0: "No",
        1: "Sí",
    },
    "pendiente_segmento_st": {
        1: "Ascendente",
        2: "Plana",
        3: "Descendente",
    },
    "vasos_principales_observados": {
        0: "Ningún vaso principal observado",
        1: "Un vaso principal",
        2: "Dos vasos principales",
        3: "Tres vasos principales",
    },
    "resultado_prueba_talio": {
        3: "Sin alteración registrada",
        6: "Cambio persistente",
        7: "Cambio reversible",
    },
}

METADATOS_VARIABLES = {
    "edad": {
        "etiqueta": "Edad",
        "nombre_tecnico": "Edad",
        "original": "age",
        "sencillo": "Años de vida de la persona incluida en los datos.",
        "descripcion": "Edad de la persona registrada en el estudio.",
        "unidad": "años",
    },
    "sexo": {
        "etiqueta": "Sexo registrado",
        "nombre_tecnico": "Sexo",
        "original": "sex",
        "sencillo": "Categoría de sexo que figura en el registro original.",
        "descripcion": (
            "Categoría registrada como femenina o masculina en la variante "
            "Cleveland; no representa la identidad de género."
        ),
        "unidad": "Categoría",
    },
    "tipo_dolor_toracico": {
        "etiqueta": "Tipo de molestia en el pecho",
        "nombre_tecnico": "Tipo de dolor torácico",
        "original": "cp",
        "sencillo": "Cómo se clasificó el dolor o la molestia en el pecho.",
        "descripcion": (
            "Categorías de la evaluación original: molestia típica y atípica "
            "corresponden a patrones clínicos de angina; otro tipo de dolor "
            "agrupa el dolor no anginoso; la última opción indica que no se "
            "registraron síntomas torácicos. Son categorías históricas, no "
            "una valoración personal ni un diagnóstico."
        ),
        "unidad": "No aplica",
    },
    "presion_arterial_reposo": {
        "etiqueta": "Presión arterial en reposo",
        "nombre_tecnico": "Presión arterial en reposo",
        "original": "trestbps",
        "sencillo": "Medición de la presión arterial tomada al ingresar al hospital.",
        "descripcion": (
            "Presión arterial medida en reposo al ingresar al hospital, tal como "
            "se define en la ficha de UCI."
        ),
        "unidad": "mm Hg",
    },
    "colesterol_serico": {
        "etiqueta": "Nivel de colesterol",
        "nombre_tecnico": "Colesterol sérico",
        "original": "chol",
        "sencillo": "Cantidad de colesterol medida en el análisis de sangre.",
        "descripcion": "Valor de colesterol sérico registrado en el estudio.",
        "unidad": "mg/dl",
    },
    "glucosa_ayunas_alta": {
        "etiqueta": "¿Glucosa elevada en ayunas?",
        "nombre_tecnico": "Glucemia en ayunas",
        "original": "fbs",
        "sencillo": "Indica si el análisis en ayunas superó el umbral de los datos.",
        "descripcion": (
            "Indica si la glucosa en ayunas superaba 120 mg/dl en los datos "
            "originales."
        ),
        "unidad": "Categoría definida por el umbral de 120 mg/dl",
    },
    "resultado_electrocardiograma_reposo": {
        "etiqueta": "Resultado del electrocardiograma",
        "nombre_tecnico": "Resultado electrocardiográfico en reposo",
        "original": "restecg",
        "sencillo": "Categoría del trazado eléctrico del corazón en reposo.",
        "descripcion": (
            "Categoría del electrocardiograma en reposo: sin alteración "
            "registrada, cambios ST-T, o posible/probable hipertrofia "
            "ventricular izquierda. No constituye por sí sola un diagnóstico."
        ),
        "unidad": "No aplica",
    },
    "frecuencia_cardiaca_maxima": {
        "etiqueta": "Frecuencia cardíaca máxima",
        "nombre_tecnico": "Frecuencia cardíaca máxima alcanzada",
        "original": "thalach",
        "sencillo": "El pulso más alto anotado durante la prueba de esfuerzo.",
        "descripcion": (
            "Frecuencia máxima alcanzada durante la prueba de esfuerzo. La ficha "
            "de UCI no especifica la unidad."
        ),
        "unidad": "No especificada por UCI; habitualmente latidos/min",
        "unidad_campo": "latidos/min",
    },
    "angina_inducida_ejercicio": {
        "etiqueta": "¿Molestias en el pecho durante el ejercicio?",
        "nombre_tecnico": "Angina inducida por ejercicio",
        "original": "exang",
        "sencillo": (
            "Indica si se registraron molestias en el pecho durante el ejercicio."
        ),
        "descripcion": (
            "La variable original registra la presencia o ausencia de angina "
            "inducida por el ejercicio. Las molestias no permiten determinar "
            "por sí solas su causa."
        ),
        "unidad": "No aplica",
    },
    "depresion_st_ejercicio": {
        "etiqueta": "Cambios en la señal cardíaca durante el esfuerzo",
        "nombre_tecnico": (
            "Depresión del segmento ST inducida por ejercicio respecto al reposo"
        ),
        "original": "oldpeak",
        "sencillo": (
            "Cambio medido en una parte del trazado eléctrico durante el esfuerzo."
        ),
        "descripcion": (
            "Depresión del segmento ST inducida por el ejercicio en relación "
            "con el reposo. Es una medida del trazado, no una unidad de presión."
        ),
        "unidad": "No especificada por UCI",
        "unidad_campo": "valor",
    },
    "pendiente_segmento_st": {
        "etiqueta": "Forma de la señal cardíaca durante el esfuerzo",
        "nombre_tecnico": "Pendiente del segmento ST en el pico del ejercicio",
        "original": "slope",
        "sencillo": "Indica si esa parte del trazado subía, permanecía plana o bajaba.",
        "descripcion": (
            "Forma de la pendiente del segmento ST en el punto máximo de la "
            "prueba de ejercicio."
        ),
        "unidad": "No aplica",
    },
    "vasos_principales_observados": {
        "etiqueta": "Vasos principales observados",
        "nombre_tecnico": (
            "Número de vasos principales coloreados mediante fluoroscopia"
        ),
        "original": "ca",
        "sencillo": "Número de vasos principales que aparecen en la prueba de imagen.",
        "descripcion": (
            "Número de vasos principales coloreados mediante fluoroscopia, "
            "según la definición de UCI."
        ),
        "unidad": "Recuento (0–3)",
    },
    "resultado_prueba_talio": {
        "etiqueta": "Resultado de la prueba de perfusión cardíaca",
        "nombre_tecnico": "Resultado de la prueba de perfusión con talio",
        "original": "thal",
        "sencillo": (
            "Categoría anotada para el resultado de una prueba que observa el "
            "paso de sangre por el corazón."
        ),
        "descripcion": (
            "Resultado categórico de una prueba de perfusión con talio: normal, "
            "defecto fijo (cambio persistente) o reversible (cambio que varía "
            "entre fases de la prueba). La etiqueta no determina por sí sola "
            "la causa ni un diagnóstico."
        ),
        "unidad": "Categoría codificada por UCI",
    },
}

SECCIONES_FORMULARIO = (
    (
        "Datos generales",
        ("edad", "sexo"),
    ),
    (
        "Síntomas y mediciones",
        (
            "tipo_dolor_toracico",
            "presion_arterial_reposo",
            "colesterol_serico",
            "glucosa_ayunas_alta",
            "frecuencia_cardiaca_maxima",
            "angina_inducida_ejercicio",
        ),
    ),
    (
        "Pruebas cardíacas",
        (
            "resultado_electrocardiograma_reposo",
            "depresion_st_ejercicio",
            "pendiente_segmento_st",
            "vasos_principales_observados",
            "resultado_prueba_talio",
        ),
    ),
)


def ayuda_variable(nombre: str, bundle: dict) -> str:
    metadata = METADATOS_VARIABLES[nombre]
    lineas = [
        f"En palabras sencillas: {metadata['sencillo']}",
        f"Nombre técnico en español: {metadata['nombre_tecnico']}.",
        f"Explicación técnica: {metadata['descripcion']}",
        f"Unidad documentada: {metadata['unidad']}.",
    ]
    if nombre in CATEGORICAL_VALUES:
        opciones = [
            ETIQUETAS_CATEGORIAS[nombre][valor]
            for valor in sorted(CATEGORICAL_VALUES[nombre])
        ]
        lineas.append("Opciones admitidas: " + "; ".join(opciones) + ".")
        if nombre == "tipo_dolor_toracico":
            lineas.append(
                "Molestia típica y molestia atípica corresponden a patrones "
                "clínicos de angina; «otro tipo de dolor» agrupa dolor no "
                "anginoso; «sin molestias torácicas registradas» indica que "
                "no constan síntomas torácicos en el registro."
            )
        if nombre == "angina_inducida_ejercicio":
            lineas.append(
                "En palabras sencillas: indica si se registraron molestias "
                "en el pecho durante el ejercicio. Explicación técnica: la "
                "variable original registra presencia o ausencia de angina "
                "inducida por el ejercicio; las molestias no determinan por "
                "sí solas su causa."
            )
        lineas.append(
            "Si no conoces el dato, selecciona «No lo sé»; el pipeline lo "
            "sustituirá por la categoría más frecuente aprendida en entrenamiento."
        )
    else:
        limites = bundle["numeric_ranges"][nombre]
        lineas.append(
            "Intervalo observado en las filas de entrenamiento: "
            f"{limites['min']:g}–{limites['max']:g} {metadata['unidad']}. "
            "Es una referencia de la muestra, no un límite clínico."
        )
        if nombre in INTEGER_FEATURES:
            lineas.append(
                "El formulario admite enteros finitos no negativos, sin fijar "
                "un máximo clínico."
            )
        else:
            lineas.append(
                "El formulario admite valores decimales finitos y no impone "
                "límites clínicos."
            )
        lineas.append(
            "Si no conoces el dato, deja el campo vacío; el pipeline lo sustituirá "
            "por la mediana aprendida en entrenamiento."
        )
    return "\n\n".join(lineas)


def crear_formulario(bundle: dict, ejemplo: dict) -> tuple[dict, list[str]]:
    columnas_modelo = set(bundle["feature_columns"])
    columnas_interfaz = {
        nombre for _, nombres in SECCIONES_FORMULARIO for nombre in nombres
    }
    if columnas_modelo != columnas_interfaz:
        raise ValueError(
            "Las variables del formulario no coinciden con las variables del modelo."
        )

    valores = {}
    valores_invalidos = []
    for titulo, nombres in SECCIONES_FORMULARIO:
        st.markdown(f"#### {titulo}")
        columnas = st.columns(2)
        for indice, nombre in enumerate(nombres):
            metadata = METADATOS_VARIABLES[nombre]
            with columnas[indice % 2]:
                if nombre in CATEGORICAL_VALUES:
                    opciones = sorted(CATEGORICAL_VALUES[nombre])
                    opciones.append(None)
                    valor_inicial = ejemplo.get(nombre, opciones[0])
                    indice_inicial = (
                        opciones.index(valor_inicial) if valor_inicial in opciones else 0
                    )

                    def etiqueta_opcion(valor, variable=nombre):
                        if valor is None:
                            return "No lo sé"
                        return ETIQUETAS_CATEGORIAS[variable][valor]

                    valores[nombre] = st.selectbox(
                        metadata["etiqueta"],
                        options=opciones,
                        index=indice_inicial,
                        format_func=etiqueta_opcion,
                        help=ayuda_variable(nombre, bundle),
                        key=f"entrada_{nombre}",
                    )
                    continue

                valor_inicial = ejemplo.get(
                    nombre, bundle["numeric_defaults"][nombre]
                )
                texto_inicial = (
                    str(int(valor_inicial))
                    if nombre in INTEGER_FEATURES
                    else f"{float(valor_inicial):.1f}"
                )
                texto = st.text_input(
                    f"{metadata['etiqueta']} "
                    f"({metadata.get('unidad_campo', metadata['unidad'])})",
                    value=texto_inicial,
                    help=ayuda_variable(nombre, bundle),
                    key=f"entrada_{nombre}",
                )
                texto = texto.strip()
                if not texto:
                    valores[nombre] = None
                    continue
                try:
                    valor = float(texto.replace(",", "."))
                    if not math.isfinite(valor):
                        raise ValueError
                    if nombre in INTEGER_FEATURES and (
                        not valor.is_integer() or valor < 0
                    ):
                        raise ValueError
                    valores[nombre] = int(valor) if nombre in INTEGER_FEATURES else valor
                except ValueError:
                    valores[nombre] = None
                    valores_invalidos.append(metadata["etiqueta"])
    return valores, valores_invalidos


st.set_page_config(page_title="Salud cardiovascular | Proyecto educativo", layout="wide")
st.markdown(
    """
    <style>
    :root {
        --cardio-ink: #17324d;
        --cardio-muted: #5d7185;
        --cardio-primary: #147d78;
        --cardio-primary-light: #e9f5f3;
        --cardio-border: #dce7eb;
        --cardio-background: #f5f8fa;
        --cardio-alert: #9b5b12;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 85% 0%, rgba(20, 125, 120, .08), transparent 34%),
            var(--cardio-background);
        color: var(--cardio-ink);
    }
    .block-container {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3, h4 {
        color: var(--cardio-ink);
        letter-spacing: -.025em;
    }
    .hero {
        border: 1px solid var(--cardio-border);
        border-radius: 24px;
        padding: clamp(1.5rem, 4vw, 2.6rem);
        margin: .4rem 0 1.25rem;
        background: linear-gradient(120deg, #fff 10%, #edf7f5 100%);
        box-shadow: 0 14px 40px rgba(23, 50, 77, .06);
    }
    .hero-eyebrow {
        color: var(--cardio-primary);
        font-size: .76rem;
        font-weight: 750;
        letter-spacing: .13em;
        text-transform: uppercase;
        margin-bottom: .65rem;
    }
    .hero-title {
        color: var(--cardio-ink);
        font-size: clamp(2rem, 5vw, 3.2rem);
        font-weight: 760;
        line-height: 1.06;
        margin: 0;
    }
    .hero-copy {
        color: var(--cardio-muted);
        font-size: 1.04rem;
        line-height: 1.65;
        margin: .9rem 0 0;
        max-width: 760px;
    }
    [data-testid="stForm"] {
        background: rgba(255, 255, 255, .92);
        border: 1px solid var(--cardio-border);
        border-radius: 20px;
        padding: clamp(1rem, 2.8vw, 2rem);
        box-shadow: 0 10px 32px rgba(23, 50, 77, .045);
    }
    [data-testid="stForm"] h4 {
        border-bottom: 1px solid var(--cardio-border);
        padding-bottom: .7rem;
        margin-top: 1.25rem;
    }
    [data-testid="stWidgetLabel"] p {
        color: var(--cardio-ink);
        font-weight: 620;
        line-height: 1.4;
    }
    [data-testid="stTooltipIcon"] {
        color: var(--cardio-primary);
    }
    [data-testid="stTooltipContent"] {
        max-width: min(420px, 86vw);
    }
    [data-testid="stFormSubmitButton"] button {
        min-height: 3.1rem;
        border: 0;
        border-radius: 12px;
        background: var(--cardio-primary);
        color: #fff;
        font-weight: 700;
        box-shadow: 0 7px 18px rgba(20, 125, 120, .19);
    }
    [data-testid="stFormSubmitButton"] button:hover {
        background: #0f6864;
        color: #fff;
    }
    [data-testid="stTabs"] [role="tab"] {
        color: var(--cardio-muted);
        font-weight: 650;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        color: var(--cardio-primary);
    }
    [data-testid="stAlert"] {
        border-radius: 14px;
    }
    .result-card {
        background: #fff;
        border: 1px solid var(--cardio-border);
        border-left: 7px solid var(--cardio-primary);
        border-radius: 18px;
        padding: clamp(1.25rem, 3vw, 2rem);
        margin: 1.2rem 0 .8rem;
        box-shadow: 0 12px 32px rgba(23, 50, 77, .07);
    }
    .result-card.positive {
        border-left-color: #b26521;
    }
    .result-kicker {
        color: var(--cardio-primary);
        font-size: .75rem;
        font-weight: 750;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin: 0 0 .65rem;
    }
    .result-card.positive .result-kicker {
        color: #995316;
    }
    .result-title {
        color: var(--cardio-ink);
        font-size: clamp(1.45rem, 3vw, 2rem);
        font-weight: 740;
        line-height: 1.22;
        margin: 0;
    }
    .result-copy {
        color: var(--cardio-muted);
        font-size: 1rem;
        line-height: 1.65;
        margin: .85rem 0 0;
        max-width: 850px;
    }
    .medical-note {
        background: #f3f6f8;
        border: 1px solid var(--cardio-border);
        border-radius: 14px;
        color: var(--cardio-ink);
        line-height: 1.6;
        padding: 1rem 1.1rem;
        margin-top: .9rem;
    }
    @media (max-width: 700px) {
        .block-container { padding: 1.1rem 1rem 2rem; }
        .hero { border-radius: 18px; }
        [data-testid="stForm"] { border-radius: 16px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <header class="hero">
      <p class="hero-eyebrow">Proyecto divulgativo de salud cardiovascular</p>
      <h1 class="hero-title">Explora una clasificación del conjunto UCI Heart Disease</h1>
      <p class="hero-copy">Introduce los datos disponibles y consulta qué clase asigna el modelo a los registros. La herramienta muestra información educativa; no ofrece una valoración médica.</p>
    </header>
    """,
    unsafe_allow_html=True,
)
st.warning(
    "Aviso médico: esta clasificación no es un diagnóstico. Un resultado "
    "positivo no confirma enfermedad y uno negativo no la descarta.",
)

try:
    bundle = load_bundle()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

ejemplo = json.loads(ARCHIVO_EJEMPLO.read_text(encoding="utf-8"))
tab_prediccion, tab_datos = st.tabs(["Probar el modelo", "Datos y metodología"])

with tab_prediccion:
    st.markdown(
        "Completa los campos con la información que tengas. Usa el icono de "
        "ayuda junto a cada campo para consultar una explicación sencilla y "
        "otra técnica. Si desconoces un dato numérico, deja su campo vacío; "
        "si desconoces uno categórico, elige «No lo sé». Se imputará "
        "con la mediana (variable numérica) o la categoría más frecuente "
        "(variable categórica) aprendida durante el entrenamiento. Los valores "
        "de ejemplo se pueden editar."
    )
    with st.form("entrada_modelo"):
        valores, valores_invalidos = crear_formulario(bundle, ejemplo)
        enviar = st.form_submit_button("Obtener clasificación experimental")

    if enviar:
        resultado = predict_from_dict(valores, bundle)
        clase = resultado["prediction"]
        titulo_resultado = (
            "Posibles indicios de enfermedad cardíaca registrada"
            if clase == 1
            else "Sin indicios de enfermedad cardíaca en la clasificación del modelo"
        )
        detalle_resultado = (
            "En este conjunto, la categoría positiva corresponde a registros "
            "con estrechamiento mayor del 50 % en al menos una arteria "
            "principal. Esta clasificación no confirma enfermedad en una "
            "persona."
            if clase == 1
            else "En este conjunto, la categoría negativa corresponde a "
            "registros sin estrechamiento mayor del 50 % en una arteria "
            "principal. No descarta otras enfermedades cardíacas."
        )
        tipo_resultado = "positive" if clase == 1 else "negative"
        st.markdown(
            f"""
            <section class="result-card {tipo_resultado}" role="status" aria-live="polite">
              <p class="result-kicker">Resultado de la clasificación</p>
              <h2 class="result-title">{titulo_resultado}</h2>
              <p class="result-copy">{detalle_resultado}</p>
            </section>
            <div class="medical-note"><strong>Importante:</strong> es una clasificación estadística de datos históricos, no una conclusión clínica. No confirma ni descarta una enfermedad y no predice infartos futuros.</div>
            """,
            unsafe_allow_html=True,
        )
        columnas_probabilidad = st.columns(2)
        with columnas_probabilidad[0]:
            st.metric(
                "Probabilidad estimada de presencia de enfermedad cardíaca registrada",
                f"{resultado['probabilities'][1]:.1%}",
            )
        with columnas_probabilidad[1]:
            st.metric(
                "Probabilidad estimada de ausencia de enfermedad cardíaca registrada",
                f"{resultado['probabilities'][0]:.1%}",
            )
        st.caption(
            "Son estimaciones del modelo para las clases del conjunto (presencia "
            "o ausencia de estrechamiento mayor del 50 % registrado). No se ha "
            "evaluado su calibración: no representan riesgo clínico individual "
            "ni predicen infartos o eventos futuros."
        )

        if any(value is None for value in valores.values()):
            st.info(
                "Los datos vacíos o marcados como «No lo sé» se imputaron "
                "dentro del pipeline: mediana para variables numéricas y "
                "categoría más frecuente para variables categóricas."
            )
        if valores_invalidos:
            st.warning(
                "No se reconoció un valor numérico válido en "
                + ", ".join(valores_invalidos)
                + ". Se trataron como desconocidos y se imputaron con los "
                "valores aprendidos durante el entrenamiento."
            )

        fuera_de_rango = [
            nombre
            for nombre, limites in bundle["numeric_ranges"].items()
            if valores[nombre] is not None
            and (
                valores[nombre] < limites["min"]
                or valores[nombre] > limites["max"]
            )
        ]
        if fuera_de_rango:
            st.warning(
                "Estos valores quedan fuera del intervalo observado en las filas "
                "de entrenamiento: "
                + ", ".join(METADATOS_VARIABLES[nombre]["etiqueta"] for nombre in fuera_de_rango)
                + ". La muestra no establece límites clínicos y la clasificación "
                "puede no ser fiable fuera de sus datos."
            )

    st.caption("Los datos introducidos no se guardan.")

with tab_datos:
    st.markdown(
        """
        **Qué muestra.** Esta sección documenta la fuente, las variables y el
        método para quienes quieran conocer el detalle del ejercicio.

        **Fuente y objetivo.** Se usa la selección Cleveland de UCI Heart
        Disease: 303 registros de pacientes evaluados en un contexto clínico.
        La etiqueta original `num` va de 0 a 4. En esta demostración se agrupa
        `num = 0` como ausencia de estrechamiento mayor del 50 % registrado y
        `num = 1–4` como presencia de ese estrechamiento en al menos una arteria
        principal, conforme a la definición de UCI. Hay 164 registros negativos
        y 139 positivos. No es una predicción de infartos futuros.

        **Interpretación.** El titular de la clasificación resume las dos
        categorías del conjunto; no diagnostica a la persona que introduce los
        datos. La aplicación muestra las salidas de `predict_proba` vinculadas
        explícitamente a las clases 1 y 0. Su calibración no se ha evaluado:
        no representan riesgo clínico individual.

        **Variables utilizadas.** Cada entrada incluye una ayuda breve en
        lenguaje sencillo y una explicación técnica. Los intervalos numéricos
        que se muestran corresponden al entrenamiento, no a recomendaciones
        médicas. El formulario no impone máximos clínicos arbitrarios.
        """
    )
    for titulo, nombres in SECCIONES_FORMULARIO:
        st.markdown(f"#### {titulo}")
        for nombre in nombres:
            metadata = METADATOS_VARIABLES[nombre]
            linea = (
                f"**{metadata['etiqueta']}** — nombre técnico: "
                f"{metadata['nombre_tecnico']}; código original en UCI: "
                f"`{metadata['original']}`. {metadata['sencillo']} "
                f"**Qué mide exactamente:** "
                f"{metadata['descripcion']} Unidad: {metadata['unidad']}."
            )
            if nombre in CATEGORICAL_VALUES:
                opciones = "; ".join(
                    ETIQUETAS_CATEGORIAS[nombre][valor]
                    for valor in sorted(CATEGORICAL_VALUES[nombre])
                )
                linea += f" Categorías admitidas: {opciones}."
            else:
                limites = bundle["numeric_ranges"][nombre]
                linea += (
                    f" En entrenamiento se observó {limites['min']:g}–"
                    f"{limites['max']:g} {metadata['unidad']}; no es un límite "
                    "clínico ni el intervalo admitido por la aplicación."
                )
            linea += (
                " Si se desconoce el dato, se imputa con la mediana numérica "
                "o la categoría más frecuente, aprendidas en entrenamiento."
            )
            st.markdown(linea)

    with st.expander("Metodología, evaluación y limitaciones"):
        st.markdown(
        """
        **Preprocesamiento.** Las cinco variables numéricas (`age`, `trestbps`,
        `chol`, `thalach` y `oldpeak`) se imputan por mediana y se estandarizan
        con `StandardScaler`. El escalador aprende medias y desviaciones solo
        del subconjunto de ajuste. Las ocho variables categóricas (incluido el
        recuento `ca`) se imputan por moda y se codifican con one-hot. Todo el
        preprocesamiento está dentro del pipeline; la validación cruzada ajusta
        una copia en cada fold y el test reservado no interviene. El formulario
        conserva las unidades originales: la transformación ocurre
        automáticamente antes del clasificador.

        El modelo desplegado es un **Random Forest de 200 árboles**. Los árboles
        no suelen necesitar escalado; se mantiene el escalador existente porque
        forma parte del pipeline compartido con la regresión logística baseline.
        No se duplicó el preprocesamiento ni se cambió el modelo.

        **Evaluación registrada.** Separación estratificada 80/20 (242 filas de
        entrenamiento y 61 de test), más validación cruzada de 4 folds solo en
        entrenamiento para ajustar hiperparámetros. El test guardado para el
        Random Forest registra TN=30, FP=3, FN=6 y TP=22; precisión=0,88,
        sensibilidad=0,79, especificidad=0,91, exactitud balanceada=0,85 y
        AUC ROC=0,92. La calibración de `predict_proba` no se evaluó.

        **Interpretación y límites.** La aplicación muestra las salidas de
        `predict_proba` para las dos clases, asociadas a la clase correcta
        mediante `classes_`. La calibración no se ha evaluado, por lo que no
        son probabilidades clínicas ni riesgos futuros. Solo hay 61
        observaciones de test y no existe validación externa o temporal. `ca`
        y `thal` proceden de pruebas
        diagnósticas y pueden aportar información cercana al proceso que define
        la etiqueta. La generalización a otras poblaciones no está establecida.
        """
        )
