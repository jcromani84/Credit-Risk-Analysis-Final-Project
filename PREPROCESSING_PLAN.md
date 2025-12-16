# 📊 Preprocessing Implementado - Documentación Completa

## 🎯 Objetivo

Este documento describe **exactamente cómo está implementado** el pipeline de preprocessing en el código actual (`src/preprocessing.py`).

El pipeline procesa los datos en **6 pasos secuenciales**, transformando 53 columnas originales en ~117 features finales normalizadas.

---

## 📋 Estructura del Dataset (54 columnas)

### Variables con Descripción Completa:

#### **Identificadores:**

- **`ID_CLIENT`** (Var_Id: 1)
  - **Descripción:** Número secuencial para el solicitante (usar como clave)
  - **Valores:** 1-50000 (train), 50001-70000 (test), 70001-90000 (prediction)
  - **Acción:** Remover antes del preprocessing

#### **Variables de Aplicación:**

- **`CLERK_TYPE`** (Var_Id: 2)

  - **Descripción:** Tipo de empleado/clerk (no informado)
  - **Valores:** C
  - **Tipo:** Categórica

- **`PAYMENT_DAY`** (Var_Id: 3)

  - **Descripción:** Día del mes elegido por el solicitante para el pago de la factura
  - **Valores:** 1, 5, 10, 15, 20, 25
  - **Tipo:** Numérica discreta

- **`APPLICATION_SUBMISSION_TYPE`** (Var_Id: 4)

  - **Descripción:** Indica si la aplicación fue enviada vía internet o en persona/por correo
  - **Valores:** Web, Carga
  - **Tipo:** Categórica binaria

- **`QUANT_ADDITIONAL_CARDS`** (Var_Id: 5)

  - **Descripción:** Cantidad de tarjetas adicionales solicitadas en el mismo formulario
  - **Valores:** 1, 2, NULL
  - **Tipo:** Numérica discreta

- **`POSTAL_ADDRESS_TYPE`** (Var_Id: 6)
  - **Descripción:** Indica si la dirección postal es la del hogar u otra. Encoding no informado
  - **Valores:** 1, 2
  - **Tipo:** Numérica/categórica

#### **Variables Demográficas:**

- **`SEX`** (Var_Id: 7)

  - **Descripción:** Sexo del solicitante
  - **Valores:** M=Male, F=Female
  - **Tipo:** Categórica binaria

- **`MARITAL_STATUS`** (Var_Id: 8)

  - **Descripción:** Estado civil. Encoding no informado
  - **Valores:** 1, 2, 3, 4, 5, 6, 7
  - **Tipo:** Numérica/categórica ordinal

- **`QUANT_DEPENDANTS`** (Var_Id: 9)

  - **Descripción:** Cantidad de dependientes
  - **Valores:** 0, 1, 2, ...
  - **Tipo:** Numérica discreta

- **`EDUCATION_LEVEL`** (Var_Id: 10)

  - **Descripción:** Nivel educativo en orden gradual. Encoding no informado
  - **Valores:** 1, 2, 3, 4, 5
  - **Tipo:** Numérica/categórica ordinal

- **`STATE_OF_BIRTH`** (Var_Id: 11)

  - **Descripción:** Estado de nacimiento
  - **Valores:** Estados brasileños, XX, missing
  - **Tipo:** Categórica

- **`CITY_OF_BIRTH`** (Var_Id: 12)

  - **Descripción:** Ciudad de nacimiento
  - **Valores:** Varios
  - **Tipo:** Categórica (alta cardinalidad)

- **`NACIONALITY`** (Var_Id: 13)
  - **Descripción:** País de nacimiento. Encoding no informado pero Brasil probablemente es 1
  - **Valores:** 0, 1, 2
  - **Tipo:** Numérica/categórica

#### **Variables de Residencia:**

- **`RESIDENCIAL_STATE`** (Var_Id: 14)

  - **Descripción:** Estado de residencia
  - **Valores:** Estados brasileños
  - **Tipo:** Categórica

- **`RESIDENCIAL_CITY`** (Var_Id: 15)

  - **Descripción:** Ciudad de residencia
  - **Valores:** Varios
  - **Tipo:** Categórica (alta cardinalidad)

- **`RESIDENCIAL_BOROUGH`** (Var_Id: 16)

  - **Descripción:** Barrio de residencia
  - **Valores:** Varios
  - **Tipo:** Categórica (alta cardinalidad)

- **`FLAG_RESIDENCIAL_PHONE`** (Var_Id: 17)

  - **Descripción:** Indica si el solicitante posee teléfono residencial
  - **Valores:** Y, N
  - **Tipo:** Categórica binaria

- **`RESIDENCIAL_PHONE_AREA_CODE`** (Var_Id: 18)

  - **Descripción:** Código de área de tres dígitos (pseudo-código)
  - **Valores:** Códigos de área
  - **Tipo:** Categórica

- **`RESIDENCE_TYPE`** (Var_Id: 19)

  - **Descripción:** Tipo de residencia. Encoding no informado. Generalmente: propia, hipoteca, alquilada, padres, familia, etc.
  - **Valores:** 1, 2, 3, 4, 5, NULL
  - **Tipo:** Numérica/categórica

- **`MONTHS_IN_RESIDENCE`** (Var_Id: 20)

  - **Descripción:** Tiempo en la residencia actual en meses
  - **Valores:** 1, 2, ..., NULL
  - **Tipo:** Numérica continua

- **`RESIDENCIAL_ZIP_3`** (Var_Id: 52)
  - **Descripción:** Tres dígitos más significativos del código postal real del hogar
  - **Valores:** Códigos postales
  - **Tipo:** Numérica/categórica

#### **Variables Financieras:**

- **`PERSONAL_MONTHLY_INCOME`** (Var_Id: 23)

  - **Descripción:** Ingreso mensual regular personal del solicitante en moneda brasileña (R$)
  - **Valores:** Valores numéricos
  - **Tipo:** Numérica continua
  - **Nota:** Variable crítica, puede tener outliers

- **`OTHER_INCOMES`** (Var_Id: 24)

  - **Descripción:** Otros ingresos del solicitante promediados mensualmente en moneda brasileña (R$)
  - **Valores:** Valores numéricos
  - **Tipo:** Numérica continua

- **`PERSONAL_ASSETS_VALUE`** (Var_Id: 32)

  - **Descripción:** Valor total de posesiones personales como casas, autos, etc. en moneda brasileña (R$)
  - **Valores:** Valores numéricos
  - **Tipo:** Numérica continua
  - **Nota:** Puede tener outliers extremos

- **`QUANT_BANKING_ACCOUNTS`** (Var_Id: 30)

  - **Descripción:** Cantidad de cuentas bancarias
  - **Valores:** 0, 1, 2
  - **Tipo:** Numérica discreta

- **`QUANT_SPECIAL_BANKING_ACCOUNTS`** (Var_Id: 31)
  - **Descripción:** Cantidad de cuentas bancarias especiales
  - **Valores:** 0, 1, 2
  - **Tipo:** Numérica discreta

#### **Variables de Tarjetas:**

- **`FLAG_VISA`** (Var_Id: 25)

  - **Descripción:** Flag indicando si el solicitante es titular de tarjeta VISA
  - **Valores:** 0, 1
  - **Tipo:** Numérica binaria

- **`FLAG_MASTERCARD`** (Var_Id: 26)

  - **Descripción:** Flag indicando si el solicitante es titular de tarjeta MASTERCARD
  - **Valores:** 0, 1
  - **Tipo:** Numérica binaria

- **`FLAG_DINERS`** (Var_Id: 27)

  - **Descripción:** Flag indicando si el solicitante es titular de tarjeta DINERS
  - **Valores:** 0, 1
  - **Tipo:** Numérica binaria

- **`FLAG_AMERICAN_EXPRESS`** (Var_Id: 28)

  - **Descripción:** Flag indicando si el solicitante es titular de tarjeta AMERICAN EXPRESS
  - **Valores:** 0, 1
  - **Tipo:** Numérica binaria

- **`FLAG_OTHER_CARDS`** (Var_Id: 29)
  - **Descripción:** A pesar de ser "FLAG", este campo presenta tres valores no explicados
  - **Valores:** 0, 1, NULL
  - **Tipo:** Numérica/categórica

#### **Variables de Empleo:**

- **`COMPANY`** (Var_Id: 34)

  - **Descripción:** Si el solicitante ha proporcionado el nombre de la compañía donde trabaja formalmente
  - **Valores:** Y, N
  - **Tipo:** Categórica binaria

- **`PROFESSIONAL_STATE`** (Var_Id: 35)

  - **Descripción:** Estado donde trabaja el solicitante
  - **Valores:** Estados brasileños
  - **Tipo:** Categórica

- **`PROFESSIONAL_CITY`** (Var_Id: 36)

  - **Descripción:** Ciudad donde trabaja el solicitante
  - **Valores:** Varios
  - **Tipo:** Categórica (alta cardinalidad, muchos missing)

- **`PROFESSIONAL_BOROUGH`** (Var_Id: 37)

  - **Descripción:** Barrio donde trabaja el solicitante
  - **Valores:** Varios
  - **Tipo:** Categórica (alta cardinalidad, muchos missing)

- **`FLAG_PROFESSIONAL_PHONE`** (Var_Id: 38)

  - **Descripción:** Indica si se proporcionó el número de teléfono profesional
  - **Valores:** Y, N
  - **Tipo:** Categórica binaria

- **`PROFESSIONAL_PHONE_AREA_CODE`** (Var_Id: 39)

  - **Descripción:** Código de área de tres dígitos (pseudo-código)
  - **Valores:** Códigos de área
  - **Tipo:** Categórica

- **`MONTHS_IN_THE_JOB`** (Var_Id: 40)

  - **Descripción:** Tiempo en el trabajo actual en meses
  - **Valores:** Valores numéricos
  - **Tipo:** Numérica continua

- **`PROFESSION_CODE`** (Var_Id: 41)

  - **Descripción:** Código de profesión del solicitante. Encoding no informado
  - **Valores:** 1, 2, 3, ...
  - **Tipo:** Numérica/categórica

- **`OCCUPATION_TYPE`** (Var_Id: 42)

  - **Descripción:** Tipo de ocupación. Encoding no informado
  - **Valores:** 1, 2, 3, 4, 5, NULL
  - **Tipo:** Numérica/categórica

- **`MATE_PROFESSION_CODE`** (Var_Id: 43)

  - **Descripción:** Código de profesión del cónyuge. Encoding no informado
  - **Valores:** 1, 2, 3, ..., NULL
  - **Tipo:** Numérica/categórica (muchos missing)

- **`PROFESSIONAL_ZIP_3`** (Var_Id: 53)
  - **Descripción:** Tres dígitos más significativos del código postal real del trabajo
  - **Valores:** Códigos postales
  - **Tipo:** Numérica/categórica

#### **Variables de Contacto:**

- **`FLAG_MOBILE_PHONE`** (Var_Id: 21)

  - **Descripción:** Indica si el solicitante posee teléfono móvil
  - **Valores:** Y, N
  - **Tipo:** Categórica binaria

- **`FLAG_EMAIL`** (Var_Id: 22)
  - **Descripción:** Indica si el solicitante posee dirección de email
  - **Valores:** 0, 1
  - **Tipo:** Numérica binaria

#### **Variables de Documentación:**

- **`FLAG_HOME_ADDRESS_DOCUMENT`** (Var_Id: 45)

  - **Descripción:** Flag indicando confirmación documental de dirección del hogar
  - **Valores:** 0, 1
  - **Tipo:** Numérica binaria

- **`FLAG_RG`** (Var_Id: 46)

  - **Descripción:** Flag indicando confirmación documental del número de cédula de ciudadanía
  - **Valores:** 0, 1
  - **Tipo:** Numérica binaria

- **`FLAG_CPF`** (Var_Id: 47)

  - **Descripción:** Flag indicando confirmación documental del estado de contribuyente
  - **Valores:** 0, 1
  - **Tipo:** Numérica binaria

- **`FLAG_INCOME_PROOF`** (Var_Id: 48)
  - **Descripción:** Flag indicando confirmación documental de ingresos
  - **Valores:** 0, 1
  - **Tipo:** Numérica binaria

#### **Otras Variables:**

- **`QUANT_CARS`** (Var_Id: 33)

  - **Descripción:** Cantidad de autos que posee el solicitante
  - **Valores:** Valores numéricos
  - **Tipo:** Numérica discreta

- **`EDUCATION_LEVEL_1`** (Var_Id: 44)

  - **Descripción:** Nivel educativo del cónyuge en orden gradual. Encoding no informado
  - **Valores:** 1, 2, 3, 4, 5, NULL
  - **Tipo:** Numérica/categórica ordinal (muchos missing)

- **`PRODUCT`** (Var_Id: 49)

  - **Descripción:** Tipo de producto de crédito solicitado. Encoding no informado
  - **Valores:** 1, 2, 7
  - **Tipo:** Numérica/categórica

- **`FLAG_ACSP_RECORD`** (Var_Id: 50)

  - **Descripción:** Flag indicando si el solicitante tiene algún registro previo de morosidad crediticia
  - **Valores:** Y, N
  - **Tipo:** Categórica binaria
  - **Nota:** Variable muy importante para riesgo crediticio

- **`AGE`** (Var_Id: 51)
  - **Descripción:** Edad del solicitante al momento de la solicitud
  - **Valores:** Valores numéricos
  - **Tipo:** Numérica continua
  - **Nota:** Variable importante, puede tener outliers (edades muy altas o muy bajas)

#### **Target:**

- **`TARGET_LABEL_BAD=1`** (Var_Id: 54)
  - **Descripción:** Variable objetivo: BAD=1 (default), GOOD=0 (no default)
  - **Valores:** 0, 1
  - **Tipo:** Numérica binaria
  - **Distribución:** ~74% NO (0), ~26% YES (1) - **Desbalanceado**

---

## 🔧 Feature Engineering Implementado

El pipeline crea **19 nuevas features** agrupadas en 8 categorías:

### 1. **Features Financieras Combinadas** (5 features)

```python
# Ingreso total mensual
TOTAL_MONTHLY_INCOME = PERSONAL_MONTHLY_INCOME + OTHER_INCOMES

# Ratio ingreso/activos
INCOME_TO_ASSETS_RATIO = PERSONAL_MONTHLY_INCOME / (PERSONAL_ASSETS_VALUE + 1)

# Ingreso por dependiente
INCOME_PER_DEPENDANT = TOTAL_MONTHLY_INCOME / (QUANT_DEPENDANTS + 1)

# Ratio de otros ingresos sobre ingreso principal
INCOME_RATIO = OTHER_INCOMES / (PERSONAL_MONTHLY_INCOME + 1e-6)

# Activos por dependiente
ASSETS_PER_DEPENDANT = PERSONAL_ASSETS_VALUE / (QUANT_DEPENDANTS + 1)
```

**Nota:** Se usa `+1` o `+1e-6` para evitar división por cero.

### 2. **Features de Estabilidad** (3 features)

```python
# Años en residencia (conversión de meses)
YEARS_IN_RESIDENCE = MONTHS_IN_RESIDENCE / 12

# Años en trabajo (conversión de meses)
YEARS_IN_JOB = MONTHS_IN_THE_JOB / 12

# Score de estabilidad general (promedio normalizado)
STABILITY_SCORE = (MONTHS_IN_RESIDENCE + MONTHS_IN_THE_JOB) / 24
```

**Nota:** `STABILITY_SCORE` usa `.fillna(0)` antes de sumar para manejar missing values.

### 3. **Features de Contacto/Documentación** (2 features)

```python
# Total de métodos de contacto disponibles
CONTACT_METHODS_COUNT = (
    FLAG_RESIDENCIAL_PHONE.fillna(0) +
    FLAG_MOBILE_PHONE.fillna(0) +
    FLAG_EMAIL.fillna(0)
)

# Total de documentos proporcionados
DOCUMENTS_COUNT = (
    FLAG_HOME_ADDRESS_DOCUMENT.fillna(0) +
    FLAG_RG.fillna(0) +
    FLAG_CPF.fillna(0) +
    FLAG_INCOME_PROOF.fillna(0)
)
```

### 4. **Features de Tarjetas** (2 features)

```python
# Total de tarjetas (principales + adicionales)
TOTAL_CARDS = (
    FLAG_VISA.fillna(0) +
    FLAG_MASTERCARD.fillna(0) +
    FLAG_DINERS.fillna(0) +
    FLAG_AMERICAN_EXPRESS.fillna(0) +
    FLAG_OTHER_CARDS.fillna(0) +
    QUANT_ADDITIONAL_CARDS.fillna(0)
)

# Tiene tarjetas principales (Visa o Mastercard)
HAS_MAJOR_CARDS = (FLAG_VISA.fillna(0) + FLAG_MASTERCARD.fillna(0) > 0).astype(int)
```

### 5. **Features Geográficas** (4 features)

```python
# Mismo estado residencia y trabajo
SAME_STATE_RES_PROF = (RESIDENCIAL_STATE == PROFESSIONAL_STATE).astype(int)

# Misma ciudad residencia y trabajo
SAME_CITY_RES_PROF = (RESIDENCIAL_CITY == PROFESSIONAL_CITY).astype(int)

# Mismo código postal residencia y trabajo
SAME_ZIP_RES_PROF = (RESIDENCIAL_ZIP_3 == PROFESSIONAL_ZIP_3).astype(int)

# Nació en el mismo estado donde reside
BORN_IN_RESIDENCE_STATE = (STATE_OF_BIRTH == RESIDENCIAL_STATE).astype(int)
```

### 6. **Features de Cuentas Bancarias** (2 features)

```python
# Total de cuentas bancarias
TOTAL_BANKING_ACCOUNTS = (
    QUANT_BANKING_ACCOUNTS.fillna(0) +
    QUANT_SPECIAL_BANKING_ACCOUNTS.fillna(0)
)

# Tiene cuentas bancarias especiales
HAS_SPECIAL_ACCOUNTS = (QUANT_SPECIAL_BANKING_ACCOUNTS > 0).astype(int)
```

### 7. **Features de Edad** (2 features)

```python
# Edad al cuadrado (para capturar relaciones no lineales)
AGE_SQUARED = AGE ** 2

# Grupos de edad (se crea después de imputar AGE en paso 4)
AGE_GROUP = pd.cut(
    AGE,
    bins=[0, 30, 40, 50, 60, 100],
    labels=["<30", "30-40", "40-50", "50-60", "60+"]
)
# Se convierte a string para encoding
```

**Nota:** `AGE_GROUP` se crea en el **Paso 4** (después de imputar missing values de AGE), pero se documenta aquí porque es parte del feature engineering.

### 8. **Features de Missing Values (Indicadores)** (8 features)

Se crean **binarias** (0/1) indicando si la variable original tiene missing:

```python
# Indicadores de missing para variables importantes
MISSING_PROFESSIONAL_CITY
MISSING_PROFESSIONAL_BOROUGH
MISSING_PROFESSION_CODE
MISSING_MONTHS_IN_RESIDENCE
MISSING_MATE_PROFESSION_CODE
MISSING_EDUCATION_LEVEL_1
MISSING_RESIDENCE_TYPE
MISSING_OCCUPATION_TYPE
```

**Total de features creadas:** 19 nuevas features + 8 indicadores de missing = **27 nuevas columnas**

---

## 🔄 Pipeline de Preprocessing - Implementación Actual

El pipeline se ejecuta en **6 pasos secuenciales**:

### **Paso 1: Limpieza Inicial** (`_step1_initial_cleaning`)

#### **1.1. Remover ID_CLIENT**

```python
if ID_COL in df.columns:
    df = df.drop(columns=[ID_COL])
```

- Se remueve la columna `ID_CLIENT` (identificador único, no útil para modelado)

#### **1.2. Convertir Flags Y/N a 0/1**

**Antes** de detectar columnas constantes, se convierten estas columnas:

- `FLAG_RESIDENCIAL_PHONE`: Y→1, N→0
- `FLAG_MOBILE_PHONE`: Y→1, N→0
- `COMPANY`: Y→1, N→0
- `FLAG_PROFESSIONAL_PHONE`: Y→1, N→0
- `FLAG_ACSP_RECORD`: Y→1, N→0

```python
df[col] = df[col].map({"Y": 1, "N": 0, "y": 1, "n": 0, 1: 1, 0: 0}).fillna(df[col])
df[col] = pd.to_numeric(df[col], errors="coerce")
```

**Razón:** Convertir antes de detectar constantes asegura que Y/N no se consideren constantes incorrectamente.

#### **1.3. Identificar y Remover Columnas Constantes**

**Solo en entrenamiento** (cuando `self.is_fitted == False`):

```python
# Detectar constantes:
# - Columnas con nunique(dropna=True) == 0 (todas NaN)
# - Columnas con nunique(dropna=True) == 1 (un solo valor único)
# - Columnas numéricas con std() == 0 (sin varianza)
constant_cols = [col for col in df.columns if ...]
self.constant_columns_removed = constant_cols  # Guardar para aplicar después
```

**Resultado típico:** Se remueven **9 columnas constantes** identificadas en el EDA:

- `CLERK_TYPE` (todos "C")
- Varias columnas numéricas con todos ceros
- Varias columnas categóricas con todos "N"

**En producción:** Se usa la lista guardada `self.constant_columns_removed` para remover las mismas columnas.

**Resultado:** De 53 columnas → **43 columnas** (después de remover 9 constantes + 1 ID)

---

### **Paso 2: Manejo de Outliers** (`_step2_handle_outliers`)

#### **Método: Winsorization con Percentiles 1%-99%**

**Variables procesadas** (definidas en `OUTLIER_COLS`):

1. `PERSONAL_MONTHLY_INCOME` (2% outliers según EDA)
2. `PERSONAL_ASSETS_VALUE` (0.96% outliers)
3. `OTHER_INCOMES` (0.92% outliers)
4. `AGE` (0.88% outliers)
5. `MONTHS_IN_RESIDENCE` (0.85% outliers)
6. `PROFESSION_CODE` (0.85% outliers)
7. `MATE_PROFESSION_CODE` (0.43% outliers)
8. `MARITAL_STATUS` (0.45% outliers)
9. `QUANT_DEPENDANTS` (0.61% outliers)
10. `MONTHS_IN_THE_JOB` (0.19% outliers)

**Proceso:**

```python
# En entrenamiento: calcular límites
lower = df[col].quantile(0.01)  # Percentil 1%
upper = df[col].quantile(0.99)  # Percentil 99%
self.outlier_limits[col] = {"lower": lower, "upper": upper}

# Aplicar capping (clip)
df[col] = df[col].clip(lower=limits["lower"], upper=limits["upper"])
```

**Resultado:**

- Valores < percentil 1% → reemplazados por percentil 1%
- Valores > percentil 99% → reemplazados por percentil 99%
- Límites se guardan en `self.outlier_limits` para aplicar en producción

**Ejemplo de límites típicos:**

- `PERSONAL_MONTHLY_INCOME`: 207.99 - 3,734.03 R$
- `PERSONAL_ASSETS_VALUE`: 0.00 - 50,000.00 R$
- `AGE`: 18.00 - 79.00 años

---

### **Paso 3: Feature Engineering** (`_step3_feature_engineering`)

#### **Crear 19 nuevas features** (descritas arriba en sección "Feature Engineering Implementado")

**Orden de creación:**

1. Features financieras (5)
2. Features de estabilidad (3)
3. Features de contacto/documentación (2)
4. Features de tarjetas (2)
5. Features geográficas (4)
6. Features de cuentas bancarias (2)
7. Features de edad (1: AGE_SQUARED; AGE_GROUP se crea en Paso 4)

**Resultado:** De 43 columnas → **62 columnas** (43 originales + 19 nuevas)

**Nota:** Los indicadores de missing (8) se crean en el Paso 4, no aquí.

---

### **Paso 4: Manejo de Missing Values** (`_step4_missing_values`)

#### **4.1. Crear Indicadores de Missing**

**Antes** de imputar, se crean 8 indicadores binarios (0/1):

```python
for col in MISSING_INDICATOR_COLS:
    indicator_col = f"MISSING_{col}"
    df[indicator_col] = df[col].isna().astype(int)
```

**Variables con indicadores:**

- `MISSING_PROFESSIONAL_CITY`
- `MISSING_PROFESSIONAL_BOROUGH`
- `MISSING_PROFESSION_CODE`
- `MISSING_MONTHS_IN_RESIDENCE`
- `MISSING_MATE_PROFESSION_CODE`
- `MISSING_EDUCATION_LEVEL_1`
- `MISSING_RESIDENCE_TYPE`
- `MISSING_OCCUPATION_TYPE`

**Resultado:** De 62 columnas → **70 columnas** (62 + 8 indicadores)

#### **4.2. Separar Columnas Categóricas y Numéricas**

**Solo en entrenamiento:**

```python
self.categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
self.numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
```

#### **4.3. Imputar Categóricas con Moda**

```python
self.categorical_imputer = SimpleImputer(strategy="most_frequent")
# Se ajusta solo con datos de entrenamiento
df[self.categorical_columns] = self.categorical_imputer.transform(df[self.categorical_columns])
```

**Estrategia:** `most_frequent` (moda) - valor más común para cada columna.

#### **4.4. Imputar Numéricas con Mediana**

```python
self.numeric_imputer = SimpleImputer(strategy="median")
# Se ajusta solo con datos de entrenamiento
df[self.numeric_columns] = self.numeric_imputer.transform(df[self.numeric_columns])
```

**Estrategia:** `median` (mediana) - valor central para cada columna numérica.

#### **4.5. Crear AGE_GROUP**

**Después** de imputar AGE:

```python
df["AGE_GROUP"] = pd.cut(
    df["AGE"],
    bins=[0, 30, 40, 50, 60, 100],
    labels=["<30", "30-40", "40-50", "50-60", "60+"]
)
df["AGE_GROUP"] = df["AGE_GROUP"].astype(str)  # Convertir a string para encoding
```

**Resultado:** De 70 columnas → **71 columnas** (70 + 1 AGE_GROUP)

---

### **Paso 5: Encoding** (`_step5_encoding`)

#### **5.1. Identificar Tipos de Columnas Categóricas**

**Solo en entrenamiento**, se clasifican las categóricas:

```python
# Binarias: exactamente 2 valores únicos
self.binary_cat_columns = [col for col in cat_cols if df[col].nunique(dropna=True) == 2]

# Múltiples categorías: separar por cardinalidad
multi_cat_columns = [col for col in cat_cols if col not in self.binary_cat_columns]

# Baja cardinalidad: ≤20 categorías (umbral configurable, default=20)
self.ohe_cat_columns = [col for col in multi_cat_columns if df[col].nunique(dropna=True) <= self.low_cardinality_threshold]

# Alta cardinalidad: >20 categorías
self.ordinal_cat_columns = [col for col in multi_cat_columns if col not in self.ohe_cat_columns]
```

#### **5.2. Encoding de Binarias: OrdinalEncoder**

```python
self.binary_encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1  # Si aparece valor nuevo en producción, se codifica como -1
)
df[binary_cols] = self.binary_encoder.transform(df[binary_cols])
```

**Resultado:** Binarias se convierten a 0/1 numéricos (1 columna → 1 columna)

**Ejemplos:** `SEX` (M/F) → 0/1, `APPLICATION_SUBMISSION_TYPE` (Web/Carga) → 0/1

#### **5.3. Encoding de Baja Cardinalidad: OneHotEncoder**

```python
self.ohe_encoder = OneHotEncoder(
    handle_unknown="ignore",  # Si aparece categoría nueva, se ignora (todas las columnas = 0)
    sparse_output=False
)
ohe_array = self.ohe_encoder.transform(df[ohe_cols])
ohe_df = pd.DataFrame(ohe_array, columns=self.ohe_encoder.get_feature_names_out(ohe_cols))
df = df.drop(columns=ohe_cols)  # Remover columnas originales
df = pd.concat([df, ohe_df], axis=1)  # Agregar columnas one-hot
```

**Resultado:** 1 columna categórica → **N columnas binarias** (una por categoría)

**Ejemplos:**

- `SEX` (M, F) → `SEX_M` (0/1), `SEX_F` (0/1) = **2 columnas**
- `RESIDENCE_TYPE` (1, 2, 3, 4, 5) → `RESIDENCE_TYPE_1`, `RESIDENCE_TYPE_2`, ..., `RESIDENCE_TYPE_5` = **5 columnas**

#### **5.4. Encoding de Alta Cardinalidad: OrdinalEncoder**

```python
self.ordinal_encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1  # Si aparece categoría nueva, se codifica como -1
)
df[ordinal_cols] = self.ordinal_encoder.transform(df[ordinal_cols])
```

**Resultado:** Alta cardinalidad se convierte a números ordinales (1 columna → 1 columna)

**Ejemplos:** `RESIDENCIAL_CITY`, `PROFESSIONAL_CITY`, `CITY_OF_BIRTH` (muchas categorías) → números 0, 1, 2, ...

**Resultado final:** Aproximadamente **117 features** (varía según categorías únicas en cada columna)

---

### **Paso 6: Escalado** (`_step6_scaling`)

#### **MinMaxScaler para Todas las Columnas Numéricas**

```python
self.scaler = MinMaxScaler()
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
# Remover TARGET_COL si existe
df[numeric_cols] = self.scaler.transform(df[numeric_cols])
```

**Resultado:**

- Todas las features numéricas se normalizan al rango **[0, 1]**
- Fórmula: `(x - min) / (max - min)`
- Se guardan `min_` y `max_` de cada columna para aplicar en producción

**NO cambia el número de columnas**, solo normaliza los valores.

**Resultado final:** **~117 features normalizadas** (todas en rango 0-1)

---

## 📊 Resumen de Transformaciones

### **Transformación de Columnas:**

```
53 columnas originales
    ↓ Paso 1: Limpieza
    - Remueve ID_CLIENT (1 columna)
    - Remueve 9 columnas constantes
    = 43 columnas
    ↓ Paso 2: Outliers
    - Winsorization (no cambia número de columnas)
    = 43 columnas
    ↓ Paso 3: Feature Engineering
    - Crea 19 nuevas features
    = 62 columnas
    ↓ Paso 4: Missing Values
    - Crea 8 indicadores de missing
    - Crea AGE_GROUP (1 columna)
    = 71 columnas
    ↓ Paso 5: Encoding
    - OneHotEncoder expande columnas (1 → múltiples)
    - OrdinalEncoder mantiene (1 → 1)
    = ~117 features
    ↓ Paso 6: Scaling
    - MinMaxScaler normaliza (no cambia número)
    = ~117 features finales (todas normalizadas 0-1)
```

---

## 💾 Guardado del Pipeline

### **Archivo:** `preprocessor.joblib`

**Contiene:**

- `PreprocessingPipeline` completo con:
  - `constant_columns_removed`: Lista de 9 columnas constantes
  - `outlier_limits`: Diccionario con límites (lower/upper) de 10 variables
  - `categorical_columns`: Lista de columnas categóricas identificadas
  - `numeric_columns`: Lista de columnas numéricas identificadas
  - `binary_cat_columns`: Lista de binarias
  - `ohe_cat_columns`: Lista de baja cardinalidad (OneHot)
  - `ordinal_cat_columns`: Lista de alta cardinalidad (Ordinal)
  - `feature_engineering_features`: Lista de 19 features creadas
  - `categorical_imputer`: SimpleImputer con modas aprendidas
  - `numeric_imputer`: SimpleImputer con medianas aprendidas
  - `binary_encoder`: OrdinalEncoder para binarias
  - `ohe_encoder`: OneHotEncoder para baja cardinalidad
  - `ordinal_encoder`: OrdinalEncoder para alta cardinalidad
  - `scaler`: MinMaxScaler con min/max aprendidos
  - `is_fitted`: Flag indicando que el pipeline está entrenado

**Tamaño típico:** ~1-2 MB

---

## 🔄 Uso en Producción

### **Entrenamiento:**

```python
pipeline = PreprocessingPipeline(low_cardinality_threshold=20)
X_train_processed = pipeline.fit_transform(X_train, X_val, X_test)
pipeline.save()  # Guarda preprocessor.joblib
```

### **Producción (nuevos datos):**

```python
pipeline = PreprocessingPipeline.load()  # Carga preprocessor.joblib
X_new_processed = pipeline.transform(X_new)  # Aplica transformaciones guardadas
```

**Garantías:**

- Mismas columnas constantes removidas
- Mismos límites de outliers aplicados
- Mismas modas/medianas para imputación
- Mismas categorías aprendidas para encoding
- Mismos min/max para escalado

---

## ⚠️ Consideraciones Importantes

1. **Desbalanceo de target:** 74% NO vs 26% YES

   - Considerar técnicas de balanceo (SMOTE, undersampling, class_weight)
   - Usar métricas apropiadas (ROC-AUC, Precision-Recall, F1-score)

2. **Missing Values:**

   - Variables con muchos missing:
     - `PROFESSIONAL_CITY`, `PROFESSIONAL_BOROUGH` - Muchos missing
     - `MATE_PROFESSION_CODE`, `EDUCATION_LEVEL_1` - Muchos missing
   - Usar indicadores de missing como features
   - Considerar que missing puede ser informativo (ej: no tiene trabajo formal)

3. **Variables de Alta Cardinalidad:**

   - `RESIDENCIAL_CITY` - Muchas categorías
   - `RESIDENCIAL_BOROUGH` - Muchas categorías
   - `PROFESSIONAL_CITY` - Muchas categorías + muchos missing
   - `CITY_OF_BIRTH` - Muchas categorías
   - **Estrategia:** Agrupar categorías poco frecuentes o usar Target Encoding

4. **Variables Geográficas:**

   - Pueden tener información útil sobre riesgo por región
   - Considerar codificar estados/ciudades por riesgo promedio (Target Encoding)
   - `RESIDENCIAL_ZIP_3` y `PROFESSIONAL_ZIP_3` pueden ser útiles para agrupar

5. **Outliers:**

   - Variables financieras (`PERSONAL_MONTHLY_INCOME`, `PERSONAL_ASSETS_VALUE`) pueden tener valores extremos
   - `AGE` puede tener valores anómalos
   - **Estrategia:** Capping con IQR o Winsorization

6. **Variables Constantes:**

   - Verificar si hay columnas con todos los valores iguales
   - Remover antes del encoding para evitar problemas

7. **Variables con Encoding Desconocido:**
   - `MARITAL_STATUS`, `EDUCATION_LEVEL`, `RESIDENCE_TYPE`, `OCCUPATION_TYPE` tienen encoding no informado
   - Tratar como categóricas ordinales si tienen orden lógico, sino como categóricas nominales

---

---

## ⚙️ Configuración

### **Parámetros del Pipeline:**

```python
PreprocessingPipeline(low_cardinality_threshold=20)
```

- `low_cardinality_threshold`: Umbral para separar baja vs alta cardinalidad (default: 20)
  - ≤20 categorías → OneHotEncoder
  - > 20 categorías → OrdinalEncoder

### **Constantes Configuradas:**

```python
# Columnas Y/N a convertir
YN_COLUMNS = [
    "FLAG_RESIDENCIAL_PHONE",
    "FLAG_MOBILE_PHONE",
    "COMPANY",
    "FLAG_PROFESSIONAL_PHONE",
    "FLAG_ACSP_RECORD",
]

# Variables para indicadores de missing
MISSING_INDICATOR_COLS = [
    "PROFESSIONAL_CITY",
    "PROFESSIONAL_BOROUGH",
    "PROFESSION_CODE",
    "MONTHS_IN_RESIDENCE",
    "MATE_PROFESSION_CODE",
    "EDUCATION_LEVEL_1",
    "RESIDENCE_TYPE",
    "OCCUPATION_TYPE",
]

# Variables para Winsorization
OUTLIER_COLS = [
    "PERSONAL_MONTHLY_INCOME",
    "PERSONAL_ASSETS_VALUE",
    "OTHER_INCOMES",
    "AGE",
    "MONTHS_IN_RESIDENCE",
    "PROFESSION_CODE",
    "MATE_PROFESSION_CODE",
    "MARITAL_STATUS",
    "QUANT_DEPENDANTS",
    "MONTHS_IN_THE_JOB",
]
```

---

## ⚠️ Consideraciones Importantes

1. **Orden de Pasos es Crítico:**

   - Feature engineering debe ir **antes** de encoding
   - Missing indicators deben crearse **antes** de imputar
   - Encoding debe ir **después** de imputar (para tener valores completos)

2. **Manejo de Valores Desconocidos:**

   - OneHotEncoder: `handle_unknown="ignore"` → categorías nuevas = todas columnas en 0
   - OrdinalEncoder: `unknown_value=-1` → categorías nuevas = -1

3. **Winsorization Limita Valores:**

   - Valores extremos se recortan a percentiles 1%-99%
   - Esto puede afectar predicciones si hay valores muy altos/bajos fuera del rango de entrenamiento

4. **Missing Values Informativos:**

   - Los indicadores de missing capturan información útil (ej: missing en variables profesionales puede indicar desempleo)
   - Los missing se imputan pero también se crean indicadores

5. **Escalado Final:**
   - Todas las features se normalizan a [0, 1]
   - Esto ayuda a modelos que usan distancias (KNN) o regularización
   - No cambia relaciones entre features, solo escala

---

## 📚 Referencias

- **Hallazgos del EDA:** Ver `EDA_FINDINGS.md` para detalles completos
- **Columnas constantes:** 9 columnas identificadas y removidas automáticamente
- **Outliers:** Proporciones específicas por variable documentadas en EDA
- **Feature Engineering:** `INCOME_RATIO` y otras features implementadas según hallazgos del EDA

---

**Estado:** ✅ Implementado y funcionando. Pipeline guardado en `data/processed/preprocessor.joblib`.
