
import re
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

# Pre-compiled regex patterns for extracting numbers from text
_RAM_RE = re.compile(r"(\d+)GB", re.IGNORECASE)
_FREQ_RE = re.compile(r"([0-9]*\.?[0-9]+)GHz", re.IGNORECASE)


def parse_ram(value: str) -> float:
    """Extract RAM size in GB."""
    if not isinstance(value, str):
        return np.nan
    m = _RAM_RE.search(value)
    return float(m.group(1)) if m else np.nan


def parse_cpu(value: str) -> Dict[str, Any]:
    """Parse CPU string into brand, family, frequency (GHz)."""
    if not isinstance(value, str):
        return {"cpu_brand": "UNK", "cpu_family": "UNK", "cpu_freq_ghz": np.nan}

    parts = value.split()  # Split CPU string into components
    brand = parts[0] if parts else "UNK"  # First part is usually the brand

    # Find CPU family by checking common patterns
    family = "UNK"
    for token in parts:
        if re.match(r"i[3579]-?\d*", token, re.IGNORECASE):  # Intel Core series
            family = token
            break
        if token.lower() in {"celeron", "pentium", "ryzen", "atom"}:  # Other families
            family = token
            break

    # Extract clock frequency if present
    freq = np.nan
    fm = _FREQ_RE.search(value)
    if fm:
        freq = float(fm.group(1))

    return {"cpu_brand": brand, "cpu_family": family, "cpu_freq_ghz": freq}


def parse_memory(value: str) -> Dict[str, float]:
    """Decompose memory string into SSD/HDD/Flash GB values."""
    if not isinstance(value, str):
        return {"mem_ssd_gb": 0.0, "mem_hdd_gb": 0.0, "mem_flash_gb": 0.0}

    parts = [p.strip() for p in value.split("+")]  # Handle multiple storage units
    ssd = hdd = flash = 0.0  # Initialize storage counters

    for p in parts:
        size = None
        # Convert TB to GB if needed
        if match := re.search(r"([0-9]*\.?[0-9]+)TB", p, re.IGNORECASE):
            size = float(match.group(1)) * 1024.0
        elif match := re.search(r"([0-9]*\.?[0-9]+)GB", p, re.IGNORECASE):
            size = float(match.group(1))

        if size is None:
            continue

        p_low = p.lower()
        # Categorize storage type and add to appropriate counter
        if "ssd" in p_low:
            ssd += size
        elif "hdd" in p_low:
            hdd += size
        elif "flash" in p_low:
            flash += size
        else:
            ssd += size  # Default to SSD if type unclear

    return {"mem_ssd_gb": ssd, "mem_hdd_gb": hdd, "mem_flash_gb": flash}


def parse_gpu(value: str) -> Dict[str, Any]:
    """Extract GPU brand."""
    if not isinstance(value, str) or not value.strip():
        return {"gpu_brand": "UNK"}
    return {"gpu_brand": value.split()[0]}  # First word is usually the brand


def parse_screen(resolution: str, inches: float) -> Dict[str, Any]:
    """Parse screen resolution, detect touch/IPS, compute PPI."""
    if not isinstance(resolution, str):
        return {"res_width": np.nan, "res_height": np.nan, "is_touch": 0.0, "is_ips": 0.0, "ppi": np.nan}

    # Extract resolution dimensions (e.g., "1920x1080")
    width = height = None
    for token in resolution.split():
        if "x" in token and re.match(r"^\d+x\d+$", token):
            try:
                w, h = token.split("x")
                width, height = int(w), int(h)
            except Exception:
                width = height = None
            break

    # Check for special features
    is_touch = 1.0 if "touch" in resolution.lower() else 0.0
    is_ips = 1.0 if "ips" in resolution.lower() else 0.0

    # Calculate pixels per inch if we have all required values
    ppi = np.nan
    if width and height and inches and inches > 0:
        ppi = ((width**2 + height**2) ** 0.5) / inches

    return {
        "res_width": width if width else np.nan,
        "res_height": height if height else np.nan,
        "is_touch": is_touch,
        "is_ips": is_ips,
        "ppi": ppi,
    }


# --- Simple custom transformers (no sklearn allowed) ---

class SimpleImputer:
    """Custom imputer that fills missing values with column means."""
    def __init__(self):
        self.means: Dict[str, float] = {}  # Store mean values for each column

    def fit(self, df: pd.DataFrame, numeric_cols: List[str]):
        """Learn the mean values for each numeric column."""
        for c in numeric_cols:
            self.means[c] = float(df[c].mean()) if len(df[c].dropna()) else 0.0
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with learned means."""
        out = df.copy()
        for c, m in self.means.items():
            out[c] = out[c].fillna(m)
        return out


class OneHotEncoder:
    """Custom one-hot encoder that converts categorical variables to binary vectors."""
    def __init__(self):
        self.categories: Dict[str, List[str]] = {}  # Store unique categories for each column

    def fit(self, df: pd.DataFrame, cat_cols: List[str]):
        """Learn the unique categories for each categorical column."""
        for c in cat_cols:
            seen = []
            for v in df[c].fillna("__NA__"):  # Handle NaN values
                if v not in seen:
                    seen.append(v)
            self.categories[c] = seen
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical columns to one-hot encoded binary columns."""
        out = df.copy()
        all_new = []
        for c, cats in self.categories.items():
            vals = out[c].fillna("__NA__")
            arr = np.zeros((len(vals), len(cats)))  # Create binary matrix
            cat_index = {cat: i for i, cat in enumerate(cats)}  # Map categories to indices
            for i, v in enumerate(vals):
                if v in cat_index:
                    arr[i, cat_index[v]] = 1.0  # Set 1 for matching category
            col_names = [f"{c}__{cat}" for cat in cats]  # Create new column names
            all_new.append(pd.DataFrame(arr, columns=col_names, index=out.index))
            out = out.drop(columns=[c])  # Remove original categorical column
        if all_new:
            out = pd.concat([out] + all_new, axis=1)  # Add new one-hot columns
        return out


class StandardScaler:
    """Custom standard scaler that normalizes features to zero mean and unit variance."""
    def __init__(self):
        self.means: Optional[np.ndarray] = None  # Store column means
        self.stds: Optional[np.ndarray] = None   # Store column standard deviations
        self.columns: List[str] = []             # Store column names

    def fit(self, df: pd.DataFrame):
        """Learn the mean and standard deviation for each column."""
        self.columns = list(df.columns)
        self.means = df.mean(axis=0).to_numpy(dtype=float)
        stds = df.std(axis=0, ddof=0).to_numpy(dtype=float)
        stds[stds == 0] = 1.0  # Avoid division by zero for constant columns
        self.stds = stds
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale features using learned parameters: (x - mean) / std."""
        arr = (df[self.columns].to_numpy(dtype=float) - self.means) / self.stds
        return pd.DataFrame(arr, columns=self.columns, index=df.index)


# --- Core pipeline ---

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply domain-specific feature engineering for laptops."""
    df = df.copy()

    # Convert screen size and weight to numeric values
    df["Inches"] = pd.to_numeric(df["Inches"], errors="coerce")
    if "Weight" in df.columns:
        df["Weight"] = df["Weight"].astype(str).str.lower().str.replace("kg", "", regex=False)
        df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")

    # Parse hardware specifications into structured features
    df["RAM_GB"] = df["Ram"].apply(parse_ram)  # Extract RAM size

    mem_parts = df["Memory"].apply(parse_memory)  # Parse storage specs
    df = pd.concat([df, pd.DataFrame(list(mem_parts))], axis=1)

    cpu_parts = df["Cpu"].apply(parse_cpu)  # Parse CPU specs
    df = pd.concat([df, pd.DataFrame(list(cpu_parts))], axis=1)

    gpu_parts = df["Gpu"].apply(parse_gpu)  # Extract GPU brand
    df = pd.concat([df, pd.DataFrame(list(gpu_parts))], axis=1)

    # Parse screen specifications
    screen_parts = [parse_screen(r, inc) for r, inc in zip(df["ScreenResolution"], df["Inches"])]
    df = pd.concat([df, pd.DataFrame(screen_parts)], axis=1)

    # Remove original text columns that have been parsed
    for c in ["Ram", "Memory", "Cpu", "Gpu", "ScreenResolution"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    return df


def preprocess_training(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """Preprocess training data -> (X, y, processors)."""
    df_feat = engineer_features(df)  # Create engineered features
    y = df_feat[target].to_numpy(dtype=float)  # Extract target variable
    X = df_feat.drop(columns=[target])  # Remove target from features

    # Identify column types for preprocessing
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    # Apply preprocessing pipeline
    imputer = SimpleImputer().fit(X, numeric_cols)  # Handle missing values
    X_imp = imputer.transform(X)

    encoder = OneHotEncoder().fit(X_imp, cat_cols)  # Encode categorical variables
    X_enc = encoder.transform(X_imp)

    scaler = StandardScaler().fit(X_enc)  # Scale features
    X_scaled = scaler.transform(X_enc)

    # Store preprocessing objects for later use
    processors = {
        "imputer": imputer,
        "encoder": encoder,
        "scaler": scaler,
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
        "feature_names": list(X_scaled.columns),
        "target": target,
    }
    return X_scaled, y, processors


def preprocess_inference(df: pd.DataFrame, processors: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """Apply preprocessing pipeline to new data using fitted processors."""
    target = processors.get("target")
    df_feat = engineer_features(df)  # Apply same feature engineering

    # Handle target variable if present (for evaluation)
    y = None
    if target in df_feat.columns:
        y = df_feat[target].to_numpy(dtype=float)
        X = df_feat.drop(columns=[target])
    else:
        X = df_feat

    # Apply same preprocessing steps as training
    # Fill missing values with training means
    for c in processors["numeric_cols"]:
        if c in X.columns:
            X[c] = X[c].fillna(processors["imputer"].means[c])
        else:
            X[c] = processors["imputer"].means[c]  # Handle missing columns

    # Apply one-hot encoding
    X_enc = processors["encoder"].transform(X)

    # Ensure all expected columns are present (add missing ones as zeros)
    for col in processors["feature_names"]:
        if col not in X_enc.columns:
            X_enc[col] = 0.0

    # Keep only expected feature columns
    X_enc = X_enc[processors["feature_names"]]

    # Apply feature scaling
    X_scaled = processors["scaler"].transform(X_enc)
    return X_scaled, y


__all__ = ["preprocess_training", "preprocess_inference"]
# === END FILE ===
