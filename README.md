# UCI Heart Disease — clasificación experimental

Proyecto de portfolio de Data Science sobre el conjunto **Heart Disease** de
UCI. Incluye un notebook de análisis, un pipeline de Random Forest serializado,
una aplicación Streamlit y una interfaz de línea de comandos para probar una
clasificación experimental.

> **Aviso importante:** el modelo no es una herramienta diagnóstica. Un
> resultado positivo no confirma una enfermedad y uno negativo no la descarta.
> No predice infartos ni otros eventos futuros, y no debe utilizarse para tomar
> decisiones médicas.

## Estructura

```text
.
├── .github/
│   └── workflows/
│       └── tests.yml
├── .gitignore
├── .python-version
├── app.py
├── README.md
├── predict_cardio.py
├── modelo_cardio.joblib
├── paciente_ejemplo_cardio.json
├── ProjectoUCI.ipynb
├── tests/
│   └── test_predict_cardio.py
├── requirements.txt
├── requirements-notebook.txt
└── requirements-test.txt
```

`app.py` contiene la aplicación principal. `predict_cardio.py` valida los datos
y centraliza la carga del modelo y la clasificación; también puede ejecutarse
desde terminal. El notebook conserva la exploración, el preprocesamiento, la
validación y la exportación del pipeline. El JSON incluido es un ejemplo
didáctico, no datos de un paciente real.

## Datos y variable objetivo

Se usa la selección **Cleveland** del conjunto UCI Heart Disease (303
observaciones, 13 variables predictoras). El conjunto original incluye otras
bases de datos y representa pacientes evaluados en un contexto clínico
concreto; no es una muestra representativa de la población general.

La variable UCI `num` codifica de 0 a 4 el estado angiográfico. Para la tarea
binaria se transforma en clase negativa si `num == 0` y positiva si `num` está
entre 1 y 4. UCI describe 0 como estrechamiento menor del 50 % y los valores
positivos como enfermedad registrada (>50 % en al menos una arteria principal).
Por tanto, la clase estima enfermedad registrada en esos datos, no un evento
cardiovascular futuro. En la selección usada hay 164 observaciones de clase
negativa y 139 de clase positiva.

## Variables de entrada

La app ofrece ayuda contextual junto a cada campo, con una descripción
divulgativa y su definición técnica. Los valores originales esperados por el
modelo se conservan internamente.

| Nombre mostrado | Nombre técnico en español | Código original en UCI | En palabras sencillas | Definición técnica y unidad |
|---|---|---|---|---|
| Edad | Edad | `age` | Edad de la persona. | Años. |
| Sexo registrado | Sexo | `sex` | Categoría de sexo registrada. | Categoría del conjunto original. |
| Tipo de molestia en el pecho | Tipo de dolor torácico | `cp` | Clasificación de la molestia en el pecho. | Categoría del dolor torácico registrado. |
| Presión arterial en reposo | Presión arterial en reposo | `trestbps` | Medición de presión al ingresar al hospital. | Presión en reposo al ingreso, mm Hg. |
| Nivel de colesterol | Colesterol sérico | `chol` | Cantidad de colesterol medida en sangre. | Colesterol sérico, mg/dl. |
| ¿Glucosa elevada en ayunas? | Glucemia en ayunas | `fbs` | Indica si se superó el umbral del conjunto. | Glucosa en ayunas >120 mg/dl, sí/no. |
| Resultado del electrocardiograma | Resultado electrocardiográfico en reposo | `restecg` | Categoría del trazado eléctrico en reposo. | Resultado electrocardiográfico categórico. |
| Frecuencia cardíaca máxima | Frecuencia cardíaca máxima alcanzada | `thalach` | Pulso máximo anotado durante la prueba de esfuerzo. | Frecuencia cardíaca máxima alcanzada; la ficha de UCI no especifica unidad (habitualmente latidos/min). |
| ¿Molestias en el pecho durante el ejercicio? | Angina inducida por ejercicio | `exang` | Indica si se registraron molestias en el pecho durante el ejercicio. | Presencia o ausencia de angina inducida por ejercicio; no determina por sí sola la causa de las molestias. |
| Cambios en la señal cardíaca durante el esfuerzo | Depresión del segmento ST inducida por ejercicio respecto al reposo | `oldpeak` | Cambio del trazado ST asociado al ejercicio. | Depresión del segmento ST inducida por ejercicio respecto al reposo; unidad no especificada por UCI. |
| Forma de la señal cardíaca durante el esfuerzo | Pendiente del segmento ST en el pico del ejercicio | `slope` | Indica si el tramo observado sube, es plano o baja. | Forma de la pendiente del segmento ST en el ejercicio. |
| Vasos principales observados | Número de vasos principales coloreados mediante fluoroscopia | `ca` | Número de vasos principales vistos en la prueba. | Vasos coloreados mediante fluoroscopia, recuento 0–3; hay valores ausentes. |
| Resultado de la prueba de perfusión cardíaca | Resultado de la prueba de perfusión con talio | `thal` | Categoría de una prueba que observa el paso de sangre por el corazón. | Resultado categórico de la prueba de perfusión; hay valores ausentes. |

Las etiquetas del selector mantienen los códigos originales sin exponerlos en
el formulario. Las categorías confirmadas para la variante Cleveland son:

| Código original | Etiqueta visible |
|---|---|
| `sex`: 0 / 1 | Femenino / Masculino |
| `cp`: 1 / 2 / 3 / 4 | Molestia típica / Molestia atípica / Otro tipo de dolor / Sin molestias torácicas registradas |
| `fbs`: 0 / 1 | No: 120 mg/dl o menos / Sí: más de 120 mg/dl |
| `restecg`: 0 / 1 / 2 | Sin alteraciones registradas / Cambios en el trazado eléctrico / Posible aumento del tamaño del ventrículo izquierdo |
| `exang`: 0 / 1 | No / Sí |
| `slope`: 1 / 2 / 3 | Ascendente / Plana / Descendente |
| `ca`: 0 / 1 / 2 / 3 | Ningún vaso / Un vaso / Dos vasos / Tres vasos principales |
| `thal`: 3 / 6 / 7 | Sin alteración registrada / Cambio persistente / Cambio reversible |

En `cp`, las dos primeras categorías corresponden a patrones clínicos de
angina; “molestia típica” no significa dolor normal ni constituye un
autodiagnóstico. La variable `exang` registra angina inducida por el ejercicio;
la etiqueta sencilla no permite determinar la causa de cualquier molestia.
En `thal`, las etiquetas sencillas resumen las categorías originales de
resultado normal, defecto fijo y defecto reversible, sin interpretarlas como
diagnósticos individuales.

La selección contiene 4 valores ausentes de `ca` y 2 de `thal`; no se eliminan
filas por esos datos. En la app, los campos numéricos se pueden dejar vacíos y
las variables categóricas tienen la opción «No lo sé». Si se introduce texto no
numérico o un valor incompatible con el formato esperado en un campo numérico,
se avisa y se trata como desconocido. En todos esos casos, los valores ausentes
se pasan al pipeline ya entrenado: se imputan con la mediana para las cinco
variables numéricas y con la categoría más frecuente para las ocho categóricas.
Esos valores de imputación se aprendieron solo durante el entrenamiento; no
se calculan a partir de lo que introduce la persona usuaria.

En los campos numéricos, la interfaz puede mostrar el intervalo observado en
las filas de entrenamiento como referencia de la muestra. No es un límite
clínico ni restringe las entradas. Los campos enteros requieren enteros no
negativos por su tipo de medida; el formulario no impone máximos clínicos.

## Metodología y resultados registrados

El notebook contiene una partición estratificada 80/20 (242 observaciones de
entrenamiento y 61 de test), validación cruzada de 4 folds en entrenamiento y
búsqueda de hiperparámetros. El preprocesamiento está integrado en el pipeline: imputación por mediana y
`StandardScaler` para las cinco variables numéricas, imputación por moda y
codificación one-hot para las ocho categóricas. El escalador aprende sus
parámetros solo del conjunto de ajuste. Cada fold clona y ajusta el pipeline
sin incluir su partición de validación, y el test se mantiene separado. Los
árboles del Random Forest no suelen necesitar escalado; se conserva el paso
porque el pipeline es compartido con la regresión logística baseline. La
aplicación recibe siempre unidades originales y aplica las transformaciones
internamente a través del mismo pipeline serializado.

En las salidas guardadas del notebook, el Random Forest exportado obtuvo en el
test (n=61): TN=30, FP=3, FN=6, TP=22; precisión=0,88, sensibilidad=0,79,
especificidad=0,91, F1=0,83, exactitud balanceada=0,85 y AUC ROC=0,92. Son
resultados de una única partición pequeña y deben leerse como exploratorios,
no como rendimiento esperado en uso real. El notebook compara varios modelos
en el mismo test; cualquier elección posterior motivada por esas métricas
introduciría sesgo de selección.

El Random Forest proporciona `predict_proba`, con las salidas vinculadas al
orden de `classes_`. Tras enviar el formulario, la aplicación muestra las dos
estimaciones como porcentajes: clase positiva (estrechamiento mayor del 50 %
registrado) y clase negativa. La calibración no se ha evaluado, por lo que no
deben interpretarse como probabilidades clínicas calibradas, riesgo individual
o probabilidad de sufrir un evento futuro.

## Ejecución

Se recomienda Python 3.12, que coincide con el entorno probado y está indicado
en `.python-version`. Desde la raíz del proyecto:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

`requirements.txt` contiene solo las dependencias de la aplicación desplegada.
La app y este archivo están en la raíz del repositorio, como espera la
configuración actual. Al crear la app en Streamlit Community Cloud, selecciona
Python 3.12 en «Advanced settings»: `.python-version` documenta la versión
local, pero Community Cloud pide elegir su versión en la configuración de la
app.
Para trabajar con el notebook, instala además sus dependencias:

```bash
python -m pip install -r requirements-notebook.txt
jupyter lab ProjectoUCI.ipynb
```

Para probar el JSON de ejemplo desde terminal:

```bash
python predict_cardio.py
```

Para proporcionar una entrada JSON propia:

```bash
python predict_cardio.py --file paciente_ejemplo_cardio.json
```

Ejecuta las pruebas con:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
```

Abre y ejecuta el notebook desde la raíz del proyecto para que la celda de
exportación escriba `modelo_cardio.joblib` en la ubicación esperada. Algunas
celdas descargan datos de UCI y otras usan bibliotecas para comparar modelos e
interpretar resultados; consulta el propio notebook antes de ejecutar todo el
análisis. El archivo del modelo está serializado con joblib y debe cargarse
únicamente desde una fuente de confianza.


## Fuentes y licencias

- Datos: [UCI Machine Learning Repository — Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease), de Andras Janosi, William Steinbrunn, Matthias Pfisterer y Robert Detrano; DOI
  [10.24432/C52P4X](https://doi.org/10.24432/C52P4X). La ficha/API oficial
  consultada identifica a sus autores y el DOI, pero no muestra una licencia
  explícita; verifica las condiciones vigentes con UCI antes de redistribuir
  los datos o sus derivados. Este repositorio no incluye una copia
  independiente del dataset.
- Las dependencias mantienen sus propias licencias, detalladas en sus
  respectivos proyectos. El código original de este repositorio no declara
  todavía una licencia propia.

## Limitaciones

- 303 registros de una selección clínica pequeña y específica; los datos no
  representan a la población general ni necesariamente a pacientes actuales.
- `ca` y `thal` provienen de pruebas diagnósticas y pueden contener información
  muy próxima al proceso usado para establecer la etiqueta. No debe entenderse
  como predicción previa al diagnóstico.
- El test solo contiene 61 observaciones; las estimaciones de métricas tienen
  alta incertidumbre. No hay validación externa ni temporal.
- No se ha demostrado calibración ni utilidad clínica, y la aplicación no
  calcula riesgo individual.
- Los resultados dependen del preprocesamiento y de las versiones de las
  bibliotecas usadas para crear `modelo_cardio.joblib`.

Este proyecto es una demostración educativa de análisis y despliegue de un
modelo de clasificación, no un dispositivo médico.
