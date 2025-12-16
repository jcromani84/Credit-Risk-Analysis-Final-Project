"""
Pipeline completo de preprocessing para Credit Risk Analysis.
Incluye: limpieza, outliers, feature engineering, missing values, encoding y escalado.
"""

from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

from src.config import PREPROCESSOR_FILE

# Constantes
ID_COL = "ID_CLIENT"
TARGET_COL = "TARGET_LABEL_BAD=1"

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

# Variables numéricas continuas para Winsorization (basado en EDA)
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


class PreprocessingPipeline:
    """
    Pipeline completo de preprocessing reutilizable.
    Guarda todos los transformadores para aplicar en nuevos datos.
    """

    def __init__(self, low_cardinality_threshold: int = 20):
        """
        Inicializa el pipeline.

        Args:
            low_cardinality_threshold: Umbral para considerar baja cardinalidad (default: 20)
        """
        self.low_cardinality_threshold = low_cardinality_threshold
        self.is_fitted = False

        # Almacenar transformadores y configuraciones
        self.constant_columns_removed: List[str] = []
        self.outlier_limits: Dict[str, Dict[str, float]] = {}  # {col: {lower: x, upper: y}}
        self.categorical_columns: List[str] = []
        self.numeric_columns: List[str] = []
        self.binary_cat_columns: List[str] = []
        self.ohe_cat_columns: List[str] = []
        self.ordinal_cat_columns: List[str] = []
        self.feature_engineering_features: List[str] = []

        # Transformadores
        self.categorical_imputer: Optional[SimpleImputer] = None
        self.numeric_imputer: Optional[SimpleImputer] = None
        self.binary_encoder: Optional[OrdinalEncoder] = None
        self.ohe_encoder: Optional[OneHotEncoder] = None
        self.ordinal_encoder: Optional[OrdinalEncoder] = None
        self.scaler: Optional[MinMaxScaler] = None

    def _step1_initial_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Paso 1: Limpieza inicial
        - Remover ID_CLIENT
        - Convertir flags Y/N a 0/1
        - Remover columnas constantes
        """
        df = df.copy()

        # Remover ID_CLIENT si existe
        if ID_COL in df.columns:
            df = df.drop(columns=[ID_COL])

        # Convertir flags Y/N a 0/1 PRIMERO (antes de detectar constantes)
        for col in YN_COLUMNS:
            if col in df.columns:
                df[col] = df[col].map({"Y": 1, "N": 0, "y": 1, "n": 0, 1: 1, 0: 0}).fillna(df[col])
                # Asegurarse de que sea numérico
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Identificar y remover columnas constantes DESPUÉS de la conversión
        if not self.is_fitted:
            # En entrenamiento: identificar constantes
            constant_cols = []
            for col in df.columns:
                # Skip ID column si existe
                if col == ID_COL:
                    continue
                # Detectar constantes: solo un valor único (ignorando NaN)
                unique_count = df[col].nunique(dropna=True)
                if unique_count == 0:  # Todas son NaN
                    constant_cols.append(col)
                elif unique_count == 1:  # Solo un valor único
                    constant_cols.append(col)
                elif df[col].dtype in ["int64", "float64", "int32", "float32"]:
                    # Para numéricas, verificar varianza
                    if df[col].std() == 0 or pd.isna(df[col].std()):
                        constant_cols.append(col)

            self.constant_columns_removed = constant_cols
            if constant_cols:
                print(f"Removing {len(constant_cols)} constant columns: {constant_cols}")

        # Remover columnas constantes
        if self.constant_columns_removed:
            df = df.drop(columns=[col for col in self.constant_columns_removed if col in df.columns])

        return df

    def _step2_handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Paso 2: Manejo de outliers usando Winsorization (percentiles 1%-99%)
        """
        df = df.copy()

        # Variables numéricas continuas para Winsorization
        cols_to_winsorize = [col for col in OUTLIER_COLS if col in df.columns]

        if not self.is_fitted:
            # En entrenamiento: calcular límites
            for col in cols_to_winsorize:
                if df[col].dtype in ["int64", "float64"]:
                    lower = df[col].quantile(0.01)
                    upper = df[col].quantile(0.99)
                    self.outlier_limits[col] = {"lower": lower, "upper": upper}

        # Aplicar Winsorization
        for col in cols_to_winsorize:
            if col in self.outlier_limits:
                limits = self.outlier_limits[col]
                df[col] = df[col].clip(lower=limits["lower"], upper=limits["upper"])

        return df

    def _step3_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Paso 3: Feature Engineering
        Crea 8 tipos de features combinadas según el plan.
        """
        df = df.copy()

        # 1. Features Financieras Combinadas
        if "PERSONAL_MONTHLY_INCOME" in df.columns and "OTHER_INCOMES" in df.columns:
            df["TOTAL_MONTHLY_INCOME"] = (
                df["PERSONAL_MONTHLY_INCOME"].fillna(0) + df["OTHER_INCOMES"].fillna(0)
            )

        if "PERSONAL_MONTHLY_INCOME" in df.columns and "PERSONAL_ASSETS_VALUE" in df.columns:
            df["INCOME_TO_ASSETS_RATIO"] = (
                df["PERSONAL_MONTHLY_INCOME"] / (df["PERSONAL_ASSETS_VALUE"] + 1)
            )

        if "TOTAL_MONTHLY_INCOME" in df.columns and "QUANT_DEPENDANTS" in df.columns:
            df["INCOME_PER_DEPENDANT"] = (
                df["TOTAL_MONTHLY_INCOME"] / (df["QUANT_DEPENDANTS"] + 1)
            )

        if "OTHER_INCOMES" in df.columns and "PERSONAL_MONTHLY_INCOME" in df.columns:
            df["INCOME_RATIO"] = (
                df["OTHER_INCOMES"] / (df["PERSONAL_MONTHLY_INCOME"] + 1e-6)
            )

        if "PERSONAL_ASSETS_VALUE" in df.columns and "QUANT_DEPENDANTS" in df.columns:
            df["ASSETS_PER_DEPENDANT"] = (
                df["PERSONAL_ASSETS_VALUE"] / (df["QUANT_DEPENDANTS"] + 1)
            )

        # 2. Features de Estabilidad
        if "MONTHS_IN_RESIDENCE" in df.columns:
            df["YEARS_IN_RESIDENCE"] = df["MONTHS_IN_RESIDENCE"] / 12

        if "MONTHS_IN_THE_JOB" in df.columns:
            df["YEARS_IN_JOB"] = df["MONTHS_IN_THE_JOB"] / 12

        if "MONTHS_IN_RESIDENCE" in df.columns and "MONTHS_IN_THE_JOB" in df.columns:
            df["STABILITY_SCORE"] = (
                df["MONTHS_IN_RESIDENCE"].fillna(0) + df["MONTHS_IN_THE_JOB"].fillna(0)
            ) / 24

        # 3. Features de Contacto/Documentación
        contact_cols = ["FLAG_RESIDENCIAL_PHONE", "FLAG_MOBILE_PHONE", "FLAG_EMAIL"]
        if all(col in df.columns for col in contact_cols):
            df["CONTACT_METHODS_COUNT"] = (
                df["FLAG_RESIDENCIAL_PHONE"].fillna(0)
                + df["FLAG_MOBILE_PHONE"].fillna(0)
                + df["FLAG_EMAIL"].fillna(0)
            )

        doc_cols = [
            "FLAG_HOME_ADDRESS_DOCUMENT",
            "FLAG_RG",
            "FLAG_CPF",
            "FLAG_INCOME_PROOF",
        ]
        if all(col in df.columns for col in doc_cols):
            df["DOCUMENTS_COUNT"] = (
                df["FLAG_HOME_ADDRESS_DOCUMENT"].fillna(0)
                + df["FLAG_RG"].fillna(0)
                + df["FLAG_CPF"].fillna(0)
                + df["FLAG_INCOME_PROOF"].fillna(0)
            )

        # 4. Features de Tarjetas
        card_cols = [
            "FLAG_VISA",
            "FLAG_MASTERCARD",
            "FLAG_DINERS",
            "FLAG_AMERICAN_EXPRESS",
            "FLAG_OTHER_CARDS",
        ]
        if all(col in df.columns for col in card_cols):
            df["TOTAL_CARDS"] = (
                df["FLAG_VISA"].fillna(0)
                + df["FLAG_MASTERCARD"].fillna(0)
                + df["FLAG_DINERS"].fillna(0)
                + df["FLAG_AMERICAN_EXPRESS"].fillna(0)
                + df["FLAG_OTHER_CARDS"].fillna(0)
            )
            if "QUANT_ADDITIONAL_CARDS" in df.columns:
                df["TOTAL_CARDS"] = df["TOTAL_CARDS"] + df["QUANT_ADDITIONAL_CARDS"].fillna(0)

            df["HAS_MAJOR_CARDS"] = (
                (df["FLAG_VISA"].fillna(0) + df["FLAG_MASTERCARD"].fillna(0)) > 0
            ).astype(int)

        # 5. Features Geográficas
        if "RESIDENCIAL_STATE" in df.columns and "PROFESSIONAL_STATE" in df.columns:
            df["SAME_STATE_RES_PROF"] = (
                df["RESIDENCIAL_STATE"] == df["PROFESSIONAL_STATE"]
            ).astype(int)

        if "RESIDENCIAL_CITY" in df.columns and "PROFESSIONAL_CITY" in df.columns:
            df["SAME_CITY_RES_PROF"] = (
                df["RESIDENCIAL_CITY"] == df["PROFESSIONAL_CITY"]
            ).astype(int)

        if "RESIDENCIAL_ZIP_3" in df.columns and "PROFESSIONAL_ZIP_3" in df.columns:
            df["SAME_ZIP_RES_PROF"] = (
                df["RESIDENCIAL_ZIP_3"] == df["PROFESSIONAL_ZIP_3"]
            ).astype(int)

        if "STATE_OF_BIRTH" in df.columns and "RESIDENCIAL_STATE" in df.columns:
            df["BORN_IN_RESIDENCE_STATE"] = (
                df["STATE_OF_BIRTH"] == df["RESIDENCIAL_STATE"]
            ).astype(int)

        # 6. Features de Cuentas Bancarias
        if "QUANT_BANKING_ACCOUNTS" in df.columns and "QUANT_SPECIAL_BANKING_ACCOUNTS" in df.columns:
            df["TOTAL_BANKING_ACCOUNTS"] = (
                df["QUANT_BANKING_ACCOUNTS"].fillna(0)
                + df["QUANT_SPECIAL_BANKING_ACCOUNTS"].fillna(0)
            )

        if "QUANT_SPECIAL_BANKING_ACCOUNTS" in df.columns:
            df["HAS_SPECIAL_ACCOUNTS"] = (df["QUANT_SPECIAL_BANKING_ACCOUNTS"] > 0).astype(int)

        # 7. Features de Edad
        if "AGE" in df.columns:
            df["AGE_SQUARED"] = df["AGE"] ** 2
            # Grupos de edad (después de imputación si es necesario)
            # Se hará después de manejar missing values

        # 8. Features de Missing Values (indicadores) - se crean en paso 4

        # Guardar lista de features creadas (para referencia)
        if not self.is_fitted:
            self.feature_engineering_features = [
                "TOTAL_MONTHLY_INCOME",
                "INCOME_TO_ASSETS_RATIO",
                "INCOME_PER_DEPENDANT",
                "INCOME_RATIO",
                "ASSETS_PER_DEPENDANT",
                "YEARS_IN_RESIDENCE",
                "YEARS_IN_JOB",
                "STABILITY_SCORE",
                "CONTACT_METHODS_COUNT",
                "DOCUMENTS_COUNT",
                "TOTAL_CARDS",
                "HAS_MAJOR_CARDS",
                "SAME_STATE_RES_PROF",
                "SAME_CITY_RES_PROF",
                "SAME_ZIP_RES_PROF",
                "BORN_IN_RESIDENCE_STATE",
                "TOTAL_BANKING_ACCOUNTS",
                "HAS_SPECIAL_ACCOUNTS",
                "AGE_SQUARED",
            ]

        return df

    def _step4_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Paso 4: Manejo de Missing Values
        - Crear indicadores de missing
        - Imputar categóricas con moda
        - Imputar numéricas con mediana
        """
        df = df.copy()

        # Crear indicadores de missing ANTES de imputar
        for col in MISSING_INDICATOR_COLS:
            if col in df.columns:
                indicator_col = f"MISSING_{col}"
                df[indicator_col] = df[col].isna().astype(int)

        # Separar categóricas y numéricas (antes de imputar)
        if not self.is_fitted:
            self.categorical_columns = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
            self.numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

            # Remover target de numéricas si existe
            if TARGET_COL in self.numeric_columns:
                self.numeric_columns.remove(TARGET_COL)

        # Imputar categóricas con moda
        if self.categorical_columns:
            if not self.is_fitted:
                self.categorical_imputer = SimpleImputer(strategy="most_frequent")
                self.categorical_imputer.fit(df[self.categorical_columns])
            df[self.categorical_columns] = self.categorical_imputer.transform(
                df[self.categorical_columns]
            )

        # Imputar numéricas con mediana
        numeric_cols_to_impute = [
            col for col in self.numeric_columns if col in df.columns
        ]
        if numeric_cols_to_impute:
            if not self.is_fitted:
                self.numeric_imputer = SimpleImputer(strategy="median")
                self.numeric_imputer.fit(df[numeric_cols_to_impute])
            df[numeric_cols_to_impute] = self.numeric_imputer.transform(
                df[numeric_cols_to_impute]
            )

        # Crear grupos de edad después de imputar AGE
        if "AGE" in df.columns and "AGE_GROUP" not in df.columns:
            df["AGE_GROUP"] = pd.cut(
                df["AGE"],
                bins=[0, 30, 40, 50, 60, 100],
                labels=["<30", "30-40", "40-50", "50-60", "60+"],
            )
            # Convertir a string para encoding
            df["AGE_GROUP"] = df["AGE_GROUP"].astype(str)

        return df

    def _step5_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Paso 5: Encoding
        - Binarias: OrdinalEncoder
        - Baja cardinalidad (≤20): OneHotEncoder
        - Alta cardinalidad (>20): OrdinalEncoder
        """
        df = df.copy()

        # Identificar columnas categóricas (después de feature engineering)
        cat_cols = [
            col
            for col in df.columns
            if col in self.categorical_columns or df[col].dtype == "object"
        ]

        if not self.is_fitted:
            # Separar binarias vs multi-categoría
            self.binary_cat_columns = [
                col
                for col in cat_cols
                if df[col].nunique(dropna=True) == 2
            ]
            multi_cat_columns = [col for col in cat_cols if col not in self.binary_cat_columns]

            # Separar baja vs alta cardinalidad
            self.ohe_cat_columns = [
                col
                for col in multi_cat_columns
                if df[col].nunique(dropna=True) <= self.low_cardinality_threshold
            ]
            self.ordinal_cat_columns = [
                col for col in multi_cat_columns if col not in self.ohe_cat_columns
            ]

        # Encoding binarias
        if self.binary_cat_columns:
            binary_cols = [col for col in self.binary_cat_columns if col in df.columns]
            if binary_cols:
                if not self.is_fitted:
                    self.binary_encoder = OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    )
                    self.binary_encoder.fit(df[binary_cols])
                df[binary_cols] = self.binary_encoder.transform(df[binary_cols])

        # Encoding OneHot para baja cardinalidad
        ohe_cols = [col for col in self.ohe_cat_columns if col in df.columns]
        if ohe_cols:
            if not self.is_fitted:
                self.ohe_encoder = OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False
                )
                self.ohe_encoder.fit(df[ohe_cols])

            ohe_array = self.ohe_encoder.transform(df[ohe_cols])
            ohe_df = pd.DataFrame(
                ohe_array,
                columns=self.ohe_encoder.get_feature_names_out(ohe_cols),
                index=df.index,
            )
            df = df.drop(columns=ohe_cols)
            df = pd.concat([df, ohe_df], axis=1)

        # Encoding Ordinal para alta cardinalidad
        ordinal_cols = [col for col in self.ordinal_cat_columns if col in df.columns]
        if ordinal_cols:
            if not self.is_fitted:
                self.ordinal_encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
                self.ordinal_encoder.fit(df[ordinal_cols])
            df[ordinal_cols] = self.ordinal_encoder.transform(df[ordinal_cols])

        return df

    def _step6_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Paso 6: Escalado con MinMaxScaler
        """
        df = df.copy()

        # Obtener columnas numéricas finales (después de encoding)
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if TARGET_COL in numeric_cols:
            numeric_cols.remove(TARGET_COL)

        if numeric_cols:
            if not self.is_fitted:
                self.scaler = MinMaxScaler()
                self.scaler.fit(df[numeric_cols])

            df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        return df

    def fit_transform(
        self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None, test_df: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Ajusta el pipeline con datos de entrenamiento y transforma train/val/test.

        Args:
            train_df: DataFrame de entrenamiento
            val_df: DataFrame de validación (opcional)
            test_df: DataFrame de test (opcional)

        Returns:
            Tupla con arrays numpy transformados (train, val, test)
        """
        self.is_fitted = False  # Permitir fitting

        # Procesar train
        train_processed = self._step1_initial_cleaning(train_df)
        train_processed = self._step2_handle_outliers(train_processed)
        train_processed = self._step3_feature_engineering(train_processed)
        train_processed = self._step4_missing_values(train_processed)
        train_processed = self._step5_encoding(train_processed)
        train_processed = self._step6_scaling(train_processed)

        self.is_fitted = True  # Marcar como ajustado

        # Procesar val y test si existen
        val_processed = None
        test_processed = None

        if val_df is not None:
            val_processed = self._step1_initial_cleaning(val_df)
            val_processed = self._step2_handle_outliers(val_processed)
            val_processed = self._step3_feature_engineering(val_processed)
            val_processed = self._step4_missing_values(val_processed)
            val_processed = self._step5_encoding(val_processed)
            val_processed = self._step6_scaling(val_processed)

        if test_df is not None:
            test_processed = self._step1_initial_cleaning(test_df)
            test_processed = self._step2_handle_outliers(test_processed)
            test_processed = self._step3_feature_engineering(test_processed)
            test_processed = self._step4_missing_values(test_processed)
            test_processed = self._step5_encoding(test_processed)
            test_processed = self._step6_scaling(test_processed)

        # Convertir a numpy arrays
        train_array = train_processed.values
        val_array = val_processed.values if val_processed is not None else None
        test_array = test_processed.values if test_processed is not None else None

        return train_array, val_array, test_array

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforma nuevos datos usando el pipeline ajustado.

        Args:
            df: DataFrame a transformar

        Returns:
            Array numpy transformado
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before calling transform()")

        df_processed = self._step1_initial_cleaning(df)
        df_processed = self._step2_handle_outliers(df_processed)
        df_processed = self._step3_feature_engineering(df_processed)
        df_processed = self._step4_missing_values(df_processed)
        df_processed = self._step5_encoding(df_processed)
        df_processed = self._step6_scaling(df_processed)

        return df_processed.values

    def save(self, filepath: Optional[str] = None) -> None:
        """
        Guarda el pipeline completo usando joblib.

        Args:
            filepath: Ruta donde guardar. Si None, usa PREPROCESSOR_FILE de config.
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving")

        filepath = filepath or PREPROCESSOR_FILE
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self, filepath)
        print(f"Pipeline saved to {filepath}")

    @classmethod
    def load(cls, filepath: Optional[str] = None) -> "PreprocessingPipeline":
        """
        Carga un pipeline guardado.

        Args:
            filepath: Ruta del pipeline. Si None, usa PREPROCESSOR_FILE de config.

        Returns:
            Pipeline cargado
        """
        filepath = filepath or PREPROCESSOR_FILE
        pipeline = joblib.load(filepath)
        return pipeline


# Función de compatibilidad con código existente
def preprocess_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    low_cardinality_threshold: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Función de compatibilidad con código existente.
    Crea un pipeline, lo ajusta y transforma los datos.
    """
    pipeline = PreprocessingPipeline(low_cardinality_threshold=low_cardinality_threshold)
    train, val, test = pipeline.fit_transform(train_df, val_df, test_df)
    return train, val, test
