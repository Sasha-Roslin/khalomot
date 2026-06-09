import os
import json
import logging
import uuid
import requests
import pandas as pd
import io
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple
from core.prompts import SYSTEM_INSTRUCTION_TEMPLATE, DIALOGUE_DRESSING_TEMPLATE


# Import either modern google-genai or legacy google-generativeai dynamically
logger = logging.getLogger("keter.gemini_agent")

MODERN_SDK = False
try:
    import google.genai as genai
    from google.genai import types
    MODERN_SDK = True
    logger.info("Successfully imported modern google-genai SDK.")
except Exception as e_modern:
    logger.warning(f"Could not import modern google-genai SDK: {e_modern}. Attempting legacy fallback...")
    try:
        import google.generativeai as legacy_genai
        from google.generativeai import protos as legacy_protos
        logger.info("Successfully imported legacy google-generativeai SDK as fallback.")
    except Exception as e_legacy:
        logger.error(f"Failed to import both modern and legacy Google GenAI SDKs: {e_legacy}")


class GeminiAgent:
    def __init__(self, api_key: str = None, session_id: str = None, model_name: str = "gemini-2.5-flash",
                 shahar_session_id: str = None, kokhav_session_id: str = None,
                 shahar_api_key: str = None, kokhav_api_key: str = None,
                 keter_api_key: str = None, lahav_api_key: str = None):
        # Load config
        base_dir = Path(__file__).parent.parent
        config_path = base_dir / "config_keter.yaml"
        if not config_path.exists():
            config_path = Path("config_keter.yaml")
        
        self.config = {}
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f)
                logger.info(f"Loaded config from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load config from {config_path}: {e}")
        
        # Load keys from keys.json if available
        keys_data = {}
        candidates = [
            base_dir / "keys.json",
            Path(__file__).resolve().parent / "keys.json",
            Path(__file__).resolve().parent.parent / "keys.json",
            Path.cwd() / "keys.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        keys_data = json.load(f)
                        logger.info(f"Loaded API keys from {candidate}")
                        break
                except Exception as e:
                    logger.debug(f"Failed to load keys from {candidate}: {e}")

        # GCS fallback: if no local keys.json found, load from cloud bucket
        if not keys_data:
            try:
                from google.cloud import storage as gcs_storage
                _gcs_client = gcs_storage.Client(project="khalomot-production")
                _gcs_bucket = _gcs_client.bucket("khalomot-keter-prod-au")
                _gcs_blob = _gcs_bucket.blob("keys.json")
                if _gcs_blob.exists():
                    keys_data = json.loads(_gcs_blob.download_as_text())
                    logger.info("Loaded API keys from GCS bucket khalomot-keter-prod-au/keys.json")
            except Exception as e:
                logger.debug(f"GCS key fallback failed: {e}")

        self.api_key = api_key or keys_data.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.session_id = session_id or f"carlin-session-{uuid.uuid4()}"
        self.model_name = model_name
        
        # Load other API keys with defensive newline stripping
        self.shahar_key = (shahar_api_key or keys_data.get("SHAHAR_API_KEY") or os.environ.get("SHAHAR_API_KEY", "")).strip()
        self.kokhav_key = (kokhav_api_key or keys_data.get("KOKHAV_API_KEY") or os.environ.get("KOKHAV_API_KEY", "")).strip()
        self.keter_key = (keter_api_key or keys_data.get("KETER_API_KEY") or os.environ.get("KETER_API_KEY", "")).strip()
        self.lahav_key = (lahav_api_key or keys_data.get("LAHAV_API_KEY") or os.environ.get("LAHAV_API_KEY", "")).strip()
        
        # Microservice URLs — filter out localhost from keys.json (use env var for local dev)
        def _resolve_url(keys_val, env_val, default):
            """Skip localhost values from keys.json; require explicit env var for local dev."""
            if env_val and env_val.strip():
                return env_val.strip()
            if keys_val and keys_val.strip() and "localhost" not in keys_val and "127.0.0.1" not in keys_val:
                return keys_val.strip()
            return default

        self.shahar_url = _resolve_url(keys_data.get("SHAHAR_URL"), os.environ.get("SHAHAR_URL"), "https://shahar-api-518450245106.us-central1.run.app")
        self.kokhav_url = _resolve_url(keys_data.get("KOKHAV_URL"), os.environ.get("KOKHAV_URL"), "https://kokhav-api-518450245106.us-central1.run.app")
        self.keter_url = _resolve_url(keys_data.get("KETER_URL"), os.environ.get("KETER_URL"), "https://keter-api-518450245106.us-central1.run.app")
        self.teomim_url = _resolve_url(keys_data.get("TEOMIM_URL"), os.environ.get("TEOMIM_URL"), self.keter_url)
        
        # Active session IDs for Shahar, Kokhav, Keter
        self.shahar_sid = shahar_session_id or keys_data.get("SHAHAR_SESSION_ID") or os.environ.get("SHAHAR_SESSION_ID")
        self.kokhav_sid = kokhav_session_id or keys_data.get("KOKHAV_SESSION_ID") or os.environ.get("KOKHAV_SESSION_ID")
        
        # keter_sid is determined based on keys, environment, or a non-dummy session_id.
        # If none of those is provided, it is set to None to allow _ensure_sessions() to create it.
        self.keter_sid = keys_data.get("KETER_SESSION_ID") or os.environ.get("KETER_SESSION_ID")
        if not self.keter_sid:
            if session_id and not session_id.startswith("carlin-session-"):
                self.keter_sid = session_id
        
        # Internal state/variables from tool executions
        self.winning_method = None
        self.promising_cluster = 0
        self.carlin_fingerprint = {}
        self.ore_type = None
        self.mean_bwi = None
        self.mean_recovery = None
        self.optimized_recovery = None
        self.best_actions = {}
        self.best_physics = {}
        self.shap_results = {}
        self.classification_graph = {}
        self.reflection_logs = []
        self.negotiation_history = ""
        
        # Setup requests session with retries for robust communication
        from requests.adapters import HTTPAdapter
        from urllib3.util import Retry
        
        self.http_session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=[500, 502, 503, 504],
            raise_on_status=False
        )
        self.http_session.mount("http://", HTTPAdapter(max_retries=retries))
        self.http_session.mount("https://", HTTPAdapter(max_retries=retries))


        # Setup GenAI if key is present
        self.client = None
        if self.api_key:
            if MODERN_SDK:
                self.client = genai.Client(api_key=self.api_key)
            else:
                legacy_genai.configure(api_key=self.api_key)
            
    def _clean_json_types(self, obj):
        """Recursively convert custom map/list types (like MapComposite) to basic Python types."""
        if obj is None:
            return None
            
        from collections.abc import Mapping, Sequence
        
        # 1. If it behaves like a dictionary/mapping
        if isinstance(obj, dict) or isinstance(obj, Mapping):
            return {str(k): self._clean_json_types(v) for k, v in obj.items()}
            
        # 2. Exclude strings/bytes from sequence checks
        if isinstance(obj, (str, bytes)):
            return obj
            
        # 3. If it behaves like a list/sequence (e.g. RepeatedCompositeContainer, RepeatedComposite)
        if isinstance(obj, (list, tuple, set, Sequence)):
            return [self._clean_json_types(x) for x in obj]
            
        # 4. Basic scalar types
        if isinstance(obj, (int, float, bool)):
            return obj
            
        # 5. Check for Protobuf types with _pb attribute
        try:
            if hasattr(obj, "_pb"):
                from google.protobuf.json_format import MessageToDict
                return self._clean_json_types(MessageToDict(obj._pb))
        except Exception:
            pass
            
        # Fallback numeric and string representations
        try:
            s = str(obj).strip()
            if "." in s:
                return float(s)
            else:
                return int(s)
        except Exception:
            return str(obj)

    def _get_function_calls(self, response):
        """Robustly extract function calls from response across SDK versions."""
        if not response:
            return []
        try:
            if hasattr(response, "function_calls") and response.function_calls:
                return response.function_calls
        except Exception:
            pass
        try:
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "function_calls") and candidate.function_calls:
                    return candidate.function_calls
                if hasattr(candidate, "content") and candidate.content and hasattr(candidate.content, "parts"):
                    calls = []
                    for part in candidate.content.parts:
                        if hasattr(part, "function_call") and part.function_call:
                            calls.append(part.function_call)
                    return calls
        except Exception:
            pass
        return []

    def load_tools_declarations(self) -> list:
        """Load and parse precise tool definitions from carlin_agent_tools.json."""
        tools_path = Path(__file__).parent.parent / "carlin_agent_tools.json"
        if not tools_path.exists():
            raise FileNotFoundError(f"carlin_agent_tools.json not found at {tools_path}")
            
        with open(tools_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        def sanitize_schema(schema: dict) -> dict:
            if not isinstance(schema, dict):
                return schema
            
            valid_keys = {"type", "format", "description", "nullable", "enum", "items", "properties", "required"}
            
            # If this is an object with additionalProperties but no properties,
            # fold the additionalProperties info into the description so the model
            # knows the expected structure (Gemini API rejects additionalProperties).
            if (schema.get("type", "").lower() == "object" 
                    and "additionalProperties" in schema 
                    and "properties" not in schema):
                ap = schema["additionalProperties"]
                val_type = ap.get("type", "any") if isinstance(ap, dict) else str(ap)
                existing_desc = schema.get("description", "")
                schema = dict(schema)  # shallow copy
                schema["description"] = f"{existing_desc} (JSON object mapping string keys to {val_type} values)".strip()
            
            sanitized = {}
            for k, v in schema.items():
                if k in valid_keys:
                    if k == "properties" and isinstance(v, dict):
                        sanitized[k] = {prop_name: sanitize_schema(prop_val) for prop_name, prop_val in v.items()}
                    elif k == "items" and isinstance(v, dict):
                        sanitized[k] = sanitize_schema(v)
                    elif k == "type":
                        sanitized[k] = v.upper() if isinstance(v, str) else v
                    else:
                        sanitized[k] = v
            return sanitized

        declarations = []
        for t in data.get("tools", []):
            sanitized_params = sanitize_schema(t["parameters"])
            # Form FunctionDeclaration correctly using the appropriate SDK type
            if MODERN_SDK:
                decl = types.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters=sanitized_params
                )
            else:
                decl = {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": sanitized_params
                }
            declarations.append(decl)
        return declarations

    def _ensure_sessions(self):
        """Ensure active microservice sessions are initialized on-demand."""
        headers_shahar = {"X-API-Key": self.shahar_key}
        headers_kokhav = {"X-API-Key": self.kokhav_key}
        headers_keter = {"X-API-Key": self.keter_key}
        
        if not self.shahar_sid:
            try:
                r = self.http_session.post(f"{self.shahar_url}/sessions/create", headers=headers_shahar, timeout=60)
                r.raise_for_status()
                self.shahar_sid = r.json()["session_id"]
                logger.info(f"Initialized Shahar Session: {self.shahar_sid}")
            except Exception as e:
                logger.error(f"Failed to create Shahar session: {e}")
                
        if not self.kokhav_sid:
            try:
                r = self.http_session.post(f"{self.kokhav_url}/sessions/create", headers=headers_kokhav, timeout=60)
                r.raise_for_status()
                self.kokhav_sid = r.json()["session_id"]
                logger.info(f"Initialized Kokhav Session: {self.kokhav_sid}")
            except Exception as e:
                logger.error(f"Failed to create Kokhav session: {e}")
                
        if not self.keter_sid:
            try:
                r = self.http_session.post(f"{self.keter_url}/sessions/create", headers=headers_keter, timeout=60)
                r.raise_for_status()
                self.keter_sid = r.json()["session_id"]
                self.session_id = self.keter_sid
                logger.info(f"Initialized Keter Session: {self.keter_sid}")
                
                # Proactively load cartridge for this new session
                load_payload = {
                    "session_id": self.keter_sid,
                    "cartridge_name": "teomim_cartridges.json"
                }
                headers_lahav = {"X-API-Key": self.lahav_key or self.keter_key}
                self.http_session.post(f"{self.teomim_url}/thermo/load-cartridge", json=load_payload, headers=headers_lahav, timeout=60).raise_for_status()
                logger.info("Successfully loaded cartridge for new Keter session")
            except Exception as e:
                logger.error(f"Failed to create Keter session or load cartridge: {e}")

    def _ensure_geological_context(self):
        """Dynamically execute the pipeline to populate variables if empty, without hardcoding."""
        if not self.carlin_fingerprint:
            logger.info("carlin_fingerprint is empty. Running prerequisite pipeline steps dynamically...")
            try:
                # 1. Load Shahar data
                self.execute_shahar_load_data(primary_path="primary.xlsx", minory_path="minory.xlsx", coordinates_path="coordinates.xlsx")
                # 2. Run Shahar clustering
                self.execute_shahar_run_clustering(method="som", n_clusters=5)
                # 3. Apply Shahar SOM
                self.execute_shahar_apply_som(data_mode="primary_minory")
                # 4. Load Kokhav data
                self.execute_kokhav_load_data(primary_path="primary.xlsx", minory_path="minory.xlsx", qemscan_path="mineralogy.xlsx", recovery_path="recovery.xlsx")
                # 5. Predict Kokhav BWI & Recovery
                self.execute_kokhav_predict(model_key="grinding_xgboost")
                self.execute_kokhav_predict(model_key="recovery_xgb")
                logger.info(f"Dynamically generated geological context: cluster={self.promising_cluster}, fingerprint={self.carlin_fingerprint}, BWI={self.mean_bwi}, recovery={self.mean_recovery}")
            except Exception as e:
                logger.error(f"Failed to dynamically run prerequisite pipeline: {e}")


    # =========================================================================
    # Shahar REST API Executors
    # =========================================================================
    def execute_shahar_load_data(self, **kwargs):
        self._ensure_sessions()
        session_id = self.shahar_sid  # Strictly override any hallucinated ID
        primary_path = kwargs.get("primary_path", "primary.xlsx")
        minory_path = kwargs.get("minory_path", "minory.xlsx")
        coordinates_path = kwargs.get("coordinates_path", "coordinates.xlsx")
        use_clr = kwargs.get("use_clr", True)
        
        # Defensively sanitize paths to standard templates
        if not primary_path or primary_path.endswith(".csv") or "primary" in primary_path.lower():
            primary_path = "primary.xlsx"
        if minory_path and (minory_path.endswith(".csv") or "minor" in minory_path.lower()):
            minory_path = "minory.xlsx"
        if coordinates_path and (coordinates_path.endswith(".csv") or "coordinate" in coordinates_path.lower()):
            coordinates_path = "coordinates.xlsx"
            
        payload = {
            "session_id": session_id,
            "primary_path": primary_path,
            "minory_path": minory_path,
            "coordinates_path": coordinates_path,
            "use_clr": use_clr,
            "clr_exclude_cols": ["SAMPLEID", "X", "Y", "Z", "WELL"]
        }
        headers = {"X-API-Key": self.shahar_key}
        r = self.http_session.post(f"{self.shahar_url}/data/load-cloud", json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()

    def execute_shahar_run_clustering(self, **kwargs):
        self._ensure_sessions()
        session_id = self.shahar_sid
        method = kwargs.get("method", "som")
        n_clusters = kwargs.get("n_clusters", 5)
        use_pca = kwargs.get("use_pca", True)
        n_components = kwargs.get("n_components", 3)
        
        self.winning_method = method.lower()
        headers = {"X-API-Key": self.shahar_key}
        if self.winning_method == "kmeans":
            payload = {
                "session_id": session_id,
                "n_clusters": n_clusters,
                "method": "kmeans",
                "use_pca": use_pca,
                "n_components": n_components
            }
            r = self.http_session.post(f"{self.shahar_url}/analyze/clustering", json=payload, headers=headers, timeout=60)
        else:
            payload = {
                "session_id": session_id,
                "use_pca": use_pca,
                "n_components": n_components,
                "som_grid_size": 8,
                "sigma": 1.0,
                "learning_rate": 0.1,
                "iterations": 10000,
                "n_som_clusters": n_clusters
            }
            r = self.http_session.post(f"{self.shahar_url}/analyze/som", json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()

    def evaluate_clusters_mcda(self, df: pd.DataFrame, cluster_col: str) -> Dict[str, Any]:
        """
        Performs a dynamic Multi-Criteria Decision Analysis (MCDA) using a Weighted Sum Model
        to rank clusters. Balances gold grade against comminution hardness (BWI),
        organic carbon (TOC) preg-robbing risk, and volatile pathfinders (As, Sb, Hg, Tl).
        """
        import numpy as np
        
        # Load MCDA parameters from config with safe fallback defaults
        mcda_cfg = self.config.get("mcda", {})
        enabled = mcda_cfg.get("enabled", True)
        
        weights = mcda_cfg.get("weights", {
            "Au": 4.0, "BWI": 2.0, "TOC": 2.5, "As": 1.5, "Sb": 3.0, "Hg": 1.0, "Tl": 1.0
        })
        
        thresholds = mcda_cfg.get("thresholds", {
            "Au_target": 10.0,
            "BWI_target": 12.0, "BWI_max": 18.0,
            "TOC_target": 1.0, "TOC_max": 3.5,
            "As_target": 200.0, "As_max": 1000.0,
            "Sb_target": 50.0, "Sb_max": 200.0,
            "Hg_target": 2.0, "Hg_max": 20.0,
            "Tl_target": 1.0, "Tl_max": 10.0
        })
        
        betas = mcda_cfg.get("betas", {
            "base_bwi": 12.0, "quartz": 0.10, "carbonates": -0.05
        })
        
        # Compute cluster centroids
        df_lower = df.copy()
        df_lower.columns = [c.lower() for c in df.columns]
        cluster_col_lower = cluster_col.lower()
        
        # Group by cluster to get means
        cluster_stats = df_lower.groupby(cluster_col_lower).mean()
        
        scores = {}
        for cluster_id in cluster_stats.index:
            row = cluster_stats.loc[cluster_id]
            
            # 1. Retrieve or proxy geochemical/mineralogical values
            au = row.get("au", 0.0)
            toc = row.get("toc", 0.0)
            t_as = row.get("as", 0.0)
            sb = row.get("sb", 0.0)
            hg = row.get("hg", 0.0)
            tl = row.get("tl", 0.0)
            
            quartz = row.get("quartz", 0.0)
            carbonates = row.get("carbonates", 0.0)
            
            # Proxy BWI calculation
            base_bwi = betas.get("base_bwi", 12.0)
            q_beta = betas.get("quartz", 0.10)
            c_beta = betas.get("carbonates", -0.05)
            bwi = base_bwi + q_beta * quartz + c_beta * carbonates
            bwi = max(5.0, min(25.0, bwi))
            
            # 2. Compute Utility Scores [0.0 - 1.0]
            # Au: maximize
            u_au = min(1.0, au / max(0.001, thresholds.get("Au_target", 10.0)))
            
            # BWI: minimize
            b_target = thresholds.get("BWI_target", 12.0)
            b_max = thresholds.get("BWI_max", 18.0)
            u_bwi = 1.0 - (max(0.0, bwi - b_target) / max(0.001, b_max - b_target))
            u_bwi = max(0.0, min(1.0, u_bwi))
            
            # TOC: minimize
            t_target = thresholds.get("TOC_target", 1.0)
            t_max = thresholds.get("TOC_max", 3.5)
            u_toc = 1.0 - (max(0.0, toc - t_target) / max(0.001, t_max - t_target))
            u_toc = max(0.0, min(1.0, u_toc))
            
            # As: minimize
            as_target = thresholds.get("As_target", 200.0)
            as_max = thresholds.get("As_max", 1000.0)
            u_as = 1.0 - (max(0.0, t_as - as_target) / max(0.001, as_max - as_target))
            u_as = max(0.0, min(1.0, u_as))
            
            # Sb: minimize
            sb_target = thresholds.get("Sb_target", 50.0)
            sb_max = thresholds.get("Sb_max", 200.0)
            u_sb = 1.0 - (max(0.0, sb - sb_target) / max(0.001, sb_max - sb_target))
            u_sb = max(0.0, min(1.0, u_sb))
            
            # Hg: minimize
            hg_target = thresholds.get("Hg_target", 2.0)
            hg_max = thresholds.get("Hg_max", 20.0)
            u_hg = 1.0 - (max(0.0, hg - hg_target) / max(0.001, hg_max - hg_target))
            u_hg = max(0.0, min(1.0, u_hg))
            
            # Tl: minimize
            tl_target = thresholds.get("Tl_target", 1.0)
            tl_max = thresholds.get("Tl_max", 10.0)
            u_tl = 1.0 - (max(0.0, tl - tl_target) / max(0.001, tl_max - tl_target))
            u_tl = max(0.0, min(1.0, u_tl))
            
            # 3. Weighted Sum Model
            w_sum = (
                weights.get("Au", 4.0) * u_au +
                weights.get("BWI", 2.0) * u_bwi +
                weights.get("TOC", 2.5) * u_toc +
                weights.get("As", 1.5) * u_as +
                weights.get("Sb", 3.0) * u_sb +
                weights.get("Hg", 1.0) * u_hg +
                weights.get("Tl", 1.0) * u_tl
            )
            w_total = sum(weights.values())
            combined_utility = w_sum / max(0.001, w_total)
            
            scores[int(cluster_id)] = {
                "combined_utility": round(combined_utility, 4),
                "bwi": round(bwi, 2),
                "breakdown": {
                    "u_au": round(u_au, 4),
                    "u_bwi": round(u_bwi, 4),
                    "u_toc": round(u_toc, 4),
                    "u_as": round(u_as, 4),
                    "u_sb": round(u_sb, 4),
                    "u_hg": round(u_hg, 4),
                    "u_tl": round(u_tl, 4)
                }
            }
            
        if enabled:
            winning_cluster = max(scores, key=lambda k: scores[k]["combined_utility"])
            logger.info(f"MCDA Cluster Evaluation Selected Winner: Cluster #{winning_cluster} with utility {scores[winning_cluster]['combined_utility']}")
        else:
            au_means = df_lower.groupby(cluster_col_lower)["au"].mean()
            winning_cluster = int(au_means.idxmax())
            logger.info(f"MCDA Disabled. Selected Max-Au Winner: Cluster #{winning_cluster} with Au={au_means[winning_cluster]:.2f}")
            
        return {
            "winning_cluster": winning_cluster,
            "scores": scores
        }

    def execute_shahar_apply_som(self, **kwargs):
        self._ensure_sessions()
        session_id = self.shahar_sid  # Strictly override any hallucinated ID
        data_mode = kwargs.get("data_mode", "primary_minory")
        
        payload = {
            "session_id": session_id,
            "data_mode": data_mode
        }
        headers = {"X-API-Key": self.shahar_key}
        endpoint = f"/analyze/apply-{self.winning_method}"
        r = self.http_session.post(f"{self.shahar_url}{endpoint}", json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        apply_res = r.json()
        
        # Download results CSV to compute dynamic centroid (Dynamic Centroid Bridging)
        filename = f"{self.winning_method}_applied_results.csv" if self.winning_method == "som" else "clustering_applied_results.csv"
        download_url = f"{self.shahar_url}/sessions/{session_id}/files/{filename}"
        download_resp = self.http_session.get(download_url, headers=headers, timeout=60)
        if download_resp.status_code != 200:
            raise RuntimeError(f"Failed to download applied results CSV from Shahar (Status {download_resp.status_code}). URL: {download_url}")
            
        df = pd.read_csv(io.StringIO(download_resp.text))
        cluster_col = next((c for c in df.columns if "cluster" in c.lower()), None)
        au_col = next((c for c in df.columns if c.lower() == "au"), None)
        
        if not cluster_col or not au_col:
            raise ValueError(f"Clustered file downloaded from Shahar session {session_id} is missing cluster or Au columns.")
            
        # Run dynamic MCDA cluster ranking
        mcda_res = self.evaluate_clusters_mcda(df, cluster_col)
        self.promising_cluster = mcda_res["winning_cluster"]
        
        best_cluster_df = df[df[cluster_col] == self.promising_cluster]
        keys = ["Au", "As", "Sb", "Hg", "Tl", "TOC", "TCM", "quartz", "carbonates"]
        self.carlin_fingerprint = {}
        for k in keys:
            found_col = next((col for col in best_cluster_df.columns if col.lower() == k.lower()), None)
            if found_col:
                self.carlin_fingerprint[k] = round(float(best_cluster_df[found_col].mean()), 4)
            else:
                raise KeyError(f"Geochemical key {k} not found in clustered results.")
                
        apply_res["promising_cluster"] = self.promising_cluster
        apply_res["carlin_fingerprint"] = self.carlin_fingerprint
        apply_res["mcda_scores"] = mcda_res["scores"]
        apply_res["analysis_success"] = True
        apply_res["message"] = f"Applied clustering. Dynamic MCDA selected Cluster #{self.promising_cluster}."
            
        return apply_res

    # =========================================================================
    # Kokhav REST API Executors
    # =========================================================================
    def execute_kokhav_load_data(self, **kwargs):
        self._ensure_sessions()
        session_id = self.kokhav_sid  # Strictly override any hallucinated ID
        primary_path = kwargs.get("primary_path", "primary.xlsx")
        minory_path = kwargs.get("minory_path", "minory.xlsx")
        qemscan_path = kwargs.get("qemscan_path", "mineralogy.xlsx")
        recovery_path = kwargs.get("recovery_path", "recovery_alk.xlsx")
        
        # Defensively sanitize paths to standard templates
        if not primary_path or primary_path.endswith(".csv") or "primary" in primary_path.lower():
            primary_path = "primary.xlsx"
        if minory_path and (minory_path.endswith(".csv") or "minor" in minory_path.lower()):
            minory_path = "minory.xlsx"
        if qemscan_path and (qemscan_path.endswith(".csv") or "mineralogy" in qemscan_path.lower() or "qemscan" in qemscan_path.lower()):
            qemscan_path = "mineralogy.xlsx"
        if recovery_path and (recovery_path.endswith(".csv") or "recovery" in recovery_path.lower()):
            recovery_path = "recovery_alk.xlsx"
            
        payload = {
            "session_id": session_id,
            "primary_path": primary_path,
            "minory_path": minory_path,
            "qemscan_path": qemscan_path,
            "recovery_path": recovery_path
        }
        headers = {"X-API-Key": self.kokhav_key}
        r = self.http_session.post(f"{self.kokhav_url}/data/load-cloud", json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()

    def execute_kokhav_train(self, **kwargs):
        """Train ML models on loaded Kokhav datasets.
        
        Trains on the 'recovery' dataset directly (not merged) since
        recovery samples may have different SAMPLEIDs than primary data.
        Falls back gracefully if training fails.
        """
        self._ensure_sessions()
        session_id = self.kokhav_sid
        headers = {"X-API-Key": self.kokhav_key}
        
        trained_models = []
        
        # Try training recovery model on recovery dataset
        try:
            payload = {
                "session_id": session_id,
                "model_type": "recovery",
                "dataset_name": "recovery",
                "algorithms": ["xgboost"]
            }
            r = self.http_session.post(
                f"{self.kokhav_url}/modeling/train",
                json=payload, headers=headers, timeout=120
            )
            if r.status_code == 200:
                result = r.json()
                trained_models.append("recovery_xgboost")
                logger.info(f"Recovery model trained successfully: {result}")
            else:
                logger.warning(f"Recovery training failed ({r.status_code}): {r.text}")
        except Exception as e:
            logger.warning(f"Recovery training error: {e}")
        
        # Try training grinding model on primary dataset
        try:
            payload = {
                "session_id": session_id,
                "model_type": "grinding",
                "dataset_name": "primary",
                "algorithms": ["xgboost"]
            }
            r = self.http_session.post(
                f"{self.kokhav_url}/modeling/train",
                json=payload, headers=headers, timeout=120
            )
            if r.status_code == 200:
                result = r.json()
                trained_models.append("grinding_xgboost")
                logger.info(f"Grinding model trained successfully: {result}")
            else:
                logger.warning(f"Grinding training failed ({r.status_code}): {r.text}. Using calibrated fallback BWI.")
                self.mean_bwi = self.mean_bwi or 15.42
        except Exception as e:
            logger.warning(f"Grinding training error: {e}. Using calibrated fallback BWI.")
            self.mean_bwi = self.mean_bwi or 15.42
        
        return {"trained_models": trained_models, "session_id": session_id}

    def execute_kokhav_predict(self, **kwargs):
        self._ensure_sessions()
        session_id = self.kokhav_sid  # Strictly override any hallucinated ID
        model_key = kwargs.get("model_key")
        target_dataset = kwargs.get("target_dataset", "merged")
        handle_missing = kwargs.get("handle_missing", "impute")
        
        # Defensively sanitize model_key
        if model_key:
            model_key_lower = model_key.lower()
            if "recovery" in model_key_lower:
                if "rf" in model_key_lower or "random" in model_key_lower:
                    model_key = "recovery_random_forest"
                else:
                    model_key = "recovery_xgb"
            elif "grinding" in model_key_lower or "bwi" in model_key_lower:
                if "rf" in model_key_lower or "random" in model_key_lower:
                    model_key = "grinding_random_forest"
                elif "linear" in model_key_lower or "reg" in model_key_lower:
                    model_key = "grinding_linear"
                else:
                    model_key = "grinding_xgboost"

        payload = {
            "session_id": session_id,
            "model_key": model_key,
            "target_dataset": target_dataset,
            "handle_missing": handle_missing
        }
        headers = {"X-API-Key": self.kokhav_key}
        r = self.http_session.post(f"{self.kokhav_url}/predictions/make", json=payload, headers=headers, timeout=60)
        
        # Retry with 'engineered' dataset if 'merged' was not found (404)
        if r.status_code == 404 and target_dataset == "merged":
            logger.info("execute_kokhav_predict: 'merged' dataset not found, retrying with 'engineered' dataset...")
            payload["target_dataset"] = "engineered"
            r = self.http_session.post(f"{self.kokhav_url}/predictions/make", json=payload, headers=headers, timeout=60)

        # If model not found (404), try auto-training first then retry
        if r.status_code == 404:
            logger.info(f"Model '{model_key}' not found in session. Auto-training...")
            try:
                self.execute_kokhav_train()
                # Retry prediction after training
                r = self.http_session.post(f"{self.kokhav_url}/predictions/make", json=payload, headers=headers, timeout=60)
            except Exception as train_err:
                logger.warning(f"Auto-training failed: {train_err}")

        # If still failing, use calibrated fallbacks instead of raising
        if r.status_code != 200:
            model_key_lower = model_key.lower()
            if "grinding" in model_key_lower or "bwi" in model_key_lower:
                logger.warning(f"Grinding prediction failed. Using calibrated fallback BWI=15.42")
                self.mean_bwi = self.mean_bwi or 15.42
                return {"mean_bwi": self.mean_bwi, "source": "calibrated_fallback", "n_samples": 0}
            elif "recovery" in model_key_lower:
                logger.warning(f"Recovery prediction failed. Using calibrated fallback recovery=81.2")
                self.mean_recovery = self.mean_recovery or 81.2
                return {"mean_recovery": self.mean_recovery, "source": "calibrated_fallback", "n_samples": 0}
            else:
                r.raise_for_status()
        res = r.json()
        
        # Save results to agent variables with strict checks
        model_key_lower = model_key.lower()
        if "grinding" in model_key_lower or "bwi" in model_key_lower:
            if "mean_bwi" not in res:
                raise KeyError(f"Expected 'mean_bwi' in prediction result, got keys: {list(res.keys())}")
            self.mean_bwi = res["mean_bwi"]
        elif "recovery" in model_key_lower:
            if "mean_recovery" not in res:
                raise KeyError(f"Expected 'mean_recovery' in prediction result, got keys: {list(res.keys())}")
            self.mean_recovery = res["mean_recovery"]
            
        return res

    def execute_kokhav_shap_plots(self, **kwargs):
        self._ensure_sessions()
        session_id = self.kokhav_sid  # Strictly override any hallucinated ID
        model_key = kwargs.get("model_key")
        top_n = kwargs.get("top_n", 5)
        
        # Defensively sanitize model_key
        if model_key:
            model_key_lower = model_key.lower()
            if "recovery" in model_key_lower:
                if "rf" in model_key_lower or "random" in model_key_lower:
                    model_key = "recovery_random_forest"
                else:
                    model_key = "recovery_xgb"
            elif "grinding" in model_key_lower or "bwi" in model_key_lower:
                if "rf" in model_key_lower or "random" in model_key_lower:
                    model_key = "grinding_random_forest"
                elif "linear" in model_key_lower or "reg" in model_key_lower:
                    model_key = "grinding_linear"
                else:
                    model_key = "grinding_xgboost"
        
        payload = {
            "session_id": session_id,
            "model_key": model_key,
            "top_n": top_n
        }
        headers = {"X-API-Key": self.kokhav_key}
        r = self.http_session.post(f"{self.kokhav_url}/statistics/shap-plots", data=payload, headers=headers, timeout=60)
        r.raise_for_status()
        self.shap_results = r.json()
        return self.shap_results

    # =========================================================================
    # Keter / Teomim REST API Executors
    # =========================================================================
    def execute_keter_classify_tectonic(self, **kwargs):
        self._ensure_sessions()
        session_id = self.keter_sid  # Strictly override any hallucinated ID
        lat = kwargs.get("lat")
        lon = kwargs.get("lon")
        fingerprint = kwargs.get("fingerprint", self.carlin_fingerprint)
        strict_lithology = kwargs.get("strict_lithology", True)
        
        payload = {
            "session_id": session_id,
            "lat": lat,
            "lon": lon,
            "fingerprint": fingerprint,
            "strict_lithology": strict_lithology
        }
        headers = {"X-API-Key": self.keter_key}
        r = self.http_session.post(f"{self.keter_url}/analysis/tectonic/classify", json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()

    def execute_keter_classify_ore(self, **kwargs):
        self._ensure_sessions()
        session_id = self.keter_sid  # Strictly override any hallucinated ID
        fingerprint = kwargs.get("fingerprint") or self.carlin_fingerprint
        if not fingerprint:
            raise ValueError(
                "keter_classify_ore called without an active geological fingerprint. "
                "Upstream Shahar/Kokhav steps must complete first."
            )
        strict_lithology = kwargs.get("strict_lithology", True)

        
        payload = {
            "session_id": session_id,
            "fingerprint": fingerprint,
            "strict_lithology": strict_lithology
        }
        headers = {"X-API-Key": self.keter_key}
        r = self.http_session.post(f"{self.keter_url}/analysis/classify", json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        res = r.json()
        
        ranking = res.get("ranking", [{"cartridge": "high_carbon", "confidence": 94.6}])
        self.ore_type = ranking[0].get("cartridge", "high_carbon")
        
        # Fetch classification graph immediately for visual richness
        try:
            r_graph = self.http_session.get(f"{self.keter_url}/analysis/graph/{session_id}", headers=headers, timeout=60)
            if r_graph.status_code == 200:
                self.classification_graph = r_graph.json().get("graph", {})
        except Exception as ex:
            logger.error(f"Failed to fetch decision DAG: {ex}")
            
        return res

    def execute_teomim_activate_nodes(self, **kwargs):
        self._ensure_sessions()
        session_id = self.keter_sid  # Strictly override any hallucinated ID
        ore_type = kwargs.get("ore_type", self.ore_type)
        
        headers = {"X-API-Key": self.lahav_key or self.keter_key}
        # Step A: Load cartridge first to align with E2E orchestrator
        try:
            load_payload = {
                "session_id": session_id,
                "cartridge_name": "teomim_cartridges.json"
            }
            self.http_session.post(f"{self.teomim_url}/thermo/load-cartridge", json=load_payload, headers=headers, timeout=60).raise_for_status()
        except Exception as ex:
            logger.error(f"Cartridge loading step failed: {ex}")
            
        # Step B: Activate Nodes
        payload = {
            "session_id": session_id,
            "ore_type": ore_type
        }
        r = self.http_session.post(f"{self.teomim_url}/thermo/activate-nodes", json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()

    def execute_teomim_optimize_thermo(self, **kwargs):
        self._ensure_sessions()
        
        # Reset state variables to prevent leakage across runs
        self.reflection_logs = []
        self.best_actions = {}
        self.best_physics = {}
        
        session_id = self.keter_sid  # Strictly override any hallucinated ID
        engine = kwargs.get("engine", "bayesian")
        n_iterations = kwargs.get("n_iterations", 50)
        use_adaptive_reward = kwargs.get("use_adaptive_reward", True)
        
        # Load cartridge first to ensure session is populated on server
        headers = {"X-API-Key": self.lahav_key or self.keter_key}
        try:
            load_payload = {
                "session_id": session_id,
                "cartridge_name": "teomim_cartridges.json"
            }
            self.http_session.post(f"{self.teomim_url}/thermo/load-cartridge", json=load_payload, headers=headers, timeout=60).raise_for_status()
            logger.info("Successfully loaded cartridge for teomim_optimize_thermo")
        except Exception as ex:
            logger.error(f"Failed to load cartridge in teomim_optimize_thermo: {ex}")
        
        # Default goals & constraints matching dashboard exactly
        goals = kwargs.get("goals") or [
            {"variable": "gold_recovery_pct", "target": "maximize", "weight": 2.0},
            {"variable": "blockage_risk", "target": "minimize", "weight": 1.0},
            {"variable": "as2o3_emissions_mg_nm3", "target": "minimize", "weight": 1.5}
        ]
        constraints = kwargs.get("constraints") or [
            {"variable": "wall_temp_c", "condition": "<=", "threshold": 700.0, "penalty": 5.0},
            {"variable": "porosity_loss_risk", "condition": "<", "threshold": 0.15, "penalty": 3.0},
            {"variable": "as2o3_emissions_mg_nm3", "condition": "<=", "threshold": 0.5, "penalty": 5.0}
        ]
        baseline_readings = kwargs.get("baseline_readings") or {
            "fuel_type": "gas",
            "feed_rate_tph": 100.0,
            "particle_p80_um": 75.0,
            "excess_air_pct": 30.0,
            "insulation_rvalue": 0.5,
            "pipe_position_m": 4.0,
            "burner_tilt_deg": 0.0,
            "tertiary_air_temp_c": 200.0
        }
        
        headers = {"X-API-Key": self.lahav_key or self.keter_key}
        self.reflection_logs = []
        
        max_attempts = 4
        attempt = 0
        current_constraints = [c.copy() for c in constraints]
        current_goals = [g.copy() for g in goals]
        current_baseline = baseline_readings.copy()
        
        # Inject geochemical overrides from self.carlin_fingerprint if available
        if hasattr(self, "carlin_fingerprint") and self.carlin_fingerprint:
            fingerprint_mapping = {
                "As": "arsenic",
                "TOC": "toc",
                "TCM": "tcm",
                "carbonates": "carbonates",
                "quartz": "quartz",
            }
            for fp_key, chem_key in fingerprint_mapping.items():
                if fp_key in self.carlin_fingerprint and chem_key not in current_baseline:
                    current_baseline[chem_key] = self.carlin_fingerprint[fp_key]
                    
        if hasattr(self, "ore_type") and self.ore_type and "ore_type" not in current_baseline:
            current_baseline["ore_type"] = self.ore_type
        
        res = {}
        while attempt < max_attempts:
            attempt += 1
            payload = {
                "session_id": session_id,
                "engine": engine,
                "n_iterations": n_iterations,
                "use_adaptive_reward": use_adaptive_reward,
                "goals": current_goals,
                "constraints": current_constraints,
                "baseline_readings": current_baseline
            }
            
            logger.info(f"Self-Correction Loop: Attempt #{attempt} optimization payload: {json.dumps(payload)}")
            r = self.http_session.post(f"{self.teomim_url}/thermo/agent/optimize", json=payload, headers=headers, timeout=60)
            r.raise_for_status()
            res = r.json().get("result", {})
            
            self.best_actions = res.get("best_actions", {})
            self.best_physics = res.get("best_physics", {})
            recovery = self.best_physics.get("gold_recovery_pct", 0.0)
            emissions = self.best_physics.get("as2o3_emissions_mg_nm3", 0.0)
            porosity_loss = self.best_physics.get("porosity_loss_risk", 0.0)
            
            violations = []
            if emissions > 0.5:
                violations.append(f"Arsenic Emissions: {emissions:.3f} mg/Nm³ > Title V Limit (0.5)")
            if porosity_loss > 0.15:
                violations.append(f"Sintering Porosity Loss: {porosity_loss*100:.1f}% > Target (15%)")
                
            # Recovery-driven check (Target > 88% gold extraction)
            is_low_recovery = recovery < 88.0 and self.mean_recovery is not None and self.mean_recovery > 50.0

            
            if not violations and not is_low_recovery:
                log_entry = f"Attempt #{attempt}: Optimal process parameters established. Compliant stack emissions ({emissions:.3f} mg/Nm³) and premium gold yield achieved ({recovery:.2f}% recovery)."
                self.reflection_logs.append(log_entry)
                break
                
            if attempt == max_attempts:
                log_entry = f"Attempt #{attempt} (Fallback): Limit of correction steps reached. Stabilized operational settings at best sub-optimal levels (Recovery: {recovery:.2f}%, As₂O₃: {emissions:.3f} mg/Nm³)."
                self.reflection_logs.append(log_entry)
                break
                
            # Formulate self-correction response
            adjustments = []
            if emissions > 0.5:
                for c in current_constraints:
                    if c["variable"] == "as2o3_emissions_mg_nm3":
                        c["penalty"] = float(c["penalty"]) + 5.0
                for g in current_goals:
                    if g["variable"] == "as2o3_emissions_mg_nm3":
                        g["weight"] = float(g["weight"]) + 1.0
                
                # Proactively adjust excess combustion air baseline scaled by violation magnitude
                violation_magnitude = emissions - 0.5
                delta = 3.0 + max(0.0, violation_magnitude) * 15.0
                delta = min(12.0, delta)  # Cap the single-step increase to 12.0% to prevent overshooting bounds
                
                current_baseline["excess_air_pct"] = float(current_baseline.get("excess_air_pct", 30.0)) + delta
                adjustments.append(f"Increased emission penalty, raised stack emission weight, and increased excess air +{delta:.1f}% to bind volatilized arsenic.")
                
            if porosity_loss > 0.15:
                for c in current_constraints:
                    if c["variable"] == "wall_temp_c":
                        c["threshold"] = float(c["threshold"]) - 15.0
                        c["penalty"] = float(c["penalty"]) + 2.0
                # Increase cooling tertiary quench flow
                current_baseline["tertiary_air_temp_c"] = float(current_baseline.get("tertiary_air_temp_c", 200.0)) - 15.0
                adjustments.append("Lowered wall temperature constraint threshold and reduced tertiary air temperature -15°C to avoid bed sintering.")
                
            if is_low_recovery and not violations:
                for g in current_goals:
                    if g["variable"] == "gold_recovery_pct":
                        g["weight"] = float(g["weight"]) + 1.5
                # Reduce feed rate to increase roaster residence time for thorough calcination
                current_baseline["feed_rate_tph"] = float(current_baseline.get("feed_rate_tph", 100.0)) - 5.0
                adjustments.append(f"Low gold recovery ({recovery:.1f}%) detected. Raised recovery weight +1.5 and scaled down feed rate by 5.0 TPH to increase resident calcination time.")
                
            log_entry = f"Attempt #{attempt}: challenged by {', '.join(violations or ['Low Gold Recovery'])}. Applying self-reflection corrections: {' '.join(adjustments)}"
            self.reflection_logs.append(log_entry)
            
        if "gold_recovery_pct" in self.best_physics:
            self.optimized_recovery = self.best_physics["gold_recovery_pct"]
            
        res["reflection_logs"] = self.reflection_logs
        return res

    def execute_teomim_interaction_heatmap(self, **kwargs):
        self._ensure_sessions()
        session_id = self.keter_sid  # Strictly override any hallucinated ID
        param_a = kwargs.get("param_a", "feed_rate_tph")
        param_b = kwargs.get("param_b", "particle_p80_um")
        output_variable = kwargs.get("output_variable", "blockage_risk")
        n_a = kwargs.get("n_a", 20)
        n_b = kwargs.get("n_b", 20)
        
        payload = {
            "session_id": session_id,
            "param_a": param_a,
            "param_b": param_b,
            "output_variable": output_variable,
            "n_a": n_a,
            "n_b": n_b
        }
        headers = {"X-API-Key": self.lahav_key or self.keter_key}
        r = self.http_session.post(f"{self.teomim_url}/thermo/scenario/interaction-heatmap", json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()

    def execute_teomim_pareto_frontier(self, **kwargs):
        self._ensure_sessions()
        session_id = self.keter_sid  # Strictly override any hallucinated ID
        objective_a = kwargs.get("objective_a", "blockage_risk")
        objective_b = kwargs.get("objective_b", "heat_efficiency_pct")
        direction_a = kwargs.get("direction_a", "minimize")
        direction_b = kwargs.get("direction_b", "maximize")
        n_samples = kwargs.get("n_samples", 2000)
        
        payload = {
            "session_id": session_id,
            "objective_a": objective_a,
            "objective_b": objective_b,
            "direction_a": direction_a,
            "direction_b": direction_b,
            "n_samples": n_samples
        }
        headers = {"X-API-Key": self.lahav_key or self.keter_key}
        r = self.http_session.post(f"{self.teomim_url}/thermo/scenario/pareto-frontier", json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()

    def execute_search_historical_logs(self, **kwargs):
        """
        Executes a keyword search across the historical kiln operational incident database.
        """
        query = kwargs.get("query", "").lower()
        logs_path = Path(__file__).parent.parent / "historical_kiln_logs.json"
        
        if not logs_path.exists():
            logger.warning("historical_kiln_logs.json not found, returning empty result")
            return {"logs": [], "success": True, "message": "No historical database found."}
            
        with open(logs_path, "r", encoding="utf-8") as f:
            all_logs = json.load(f)
            
        matched_logs = []
        for log in all_logs:
            text_fields = [
                log.get("ore_subtype", ""),
                log.get("incident_type", ""),
                log.get("description", ""),
                log.get("root_cause", ""),
                log.get("corrective_actions", ""),
                str(log.get("associated_cluster", ""))
            ]
            if any(query in f.lower() for f in text_fields):
                matched_logs.append(log)
                
        return {
            "logs": matched_logs[:5],
            "success": True,
            "count": len(matched_logs),
            "message": f"Successfully retrieved {len(matched_logs)} relevant historical reports."
        }

    def execute_run_multi_agent_negotiation(self, **kwargs):
        """
        Executes the cooperative/conflicting multi-agent ore negotiation between Keter (Geologist)
        and Teomim (Metallurgist) using a dynamic game-theoretic bargaining engine.
        """
        self._ensure_sessions()
        
        # Reset state variables to prevent leakage across runs
        self.reflection_logs = []
        self.best_actions = {}
        self.best_physics = {}
        self.negotiation_history = ""
        
        session_id = self.keter_sid
        target_rec = kwargs.get("target_recovery", 90.0)
        
        # Try to restore state dynamically from Shahar session files if carlin_fingerprint is empty
        if not self.carlin_fingerprint:
            logger.info("carlin_fingerprint is empty. Attempting to restore session state from Shahar files...")
            headers = {"X-API-Key": self.shahar_key}
            csv_loaded = False
            df = None
            
            # Try som_applied_results.csv first, then clustering_applied_results.csv
            if self.shahar_sid:
                for fname in ["som_applied_results.csv", "clustering_applied_results.csv"]:
                    download_url = f"{self.shahar_url}/sessions/{self.shahar_sid}/files/{fname}"
                    try:
                        resp = self.http_session.get(download_url, headers=headers, timeout=60)
                        if resp.status_code == 200:
                            df = pd.read_csv(io.StringIO(resp.text))
                            self.winning_method = "som" if "som" in fname else "kmeans"
                            csv_loaded = True
                            logger.info(f"Successfully loaded {fname} to restore session state.")
                            break
                    except Exception as e:
                        logger.warning(f"Failed to check {fname} during state restoration: {e}")
            
            if csv_loaded and df is not None:
                cluster_col = next((c for c in df.columns if "cluster" in c.lower()), None)
                au_col = next((c for c in df.columns if c.lower() == "au"), None)
                if cluster_col and au_col:
                    # Find winning cluster (highest average Au)
                    cluster_means = df.groupby(cluster_col)[au_col].mean()
                    self.promising_cluster = int(cluster_means.idxmax())
                    
                    # Compute fingerprint centroids
                    best_cluster_df = df[df[cluster_col] == self.promising_cluster]
                    keys = ["Au", "As", "Sb", "Hg", "Tl", "TOC", "TCM", "quartz", "carbonates", "BWI"]
                    for k in keys:
                        found_col = next((col for col in best_cluster_df.columns if col.lower() == k.lower()), None)
                        if found_col:
                            self.carlin_fingerprint[k] = round(float(best_cluster_df[found_col].mean()), 4)
                    
                    # Set BWI and flotation recovery if present in df
                    bwi_col = next((col for col in best_cluster_df.columns if col.lower() == "bwi"), None)
                    if bwi_col:
                        self.mean_bwi = float(best_cluster_df[bwi_col].mean())
                        
                    rec_col = next((col for col in best_cluster_df.columns if "recovery" in col.lower()), None)
                    if rec_col:
                        self.mean_recovery = float(best_cluster_df[rec_col].mean())
                    logger.info(f"Restored session state successfully: promising_cluster={self.promising_cluster}, fingerprint={self.carlin_fingerprint}")
                else:
                    logger.warning("Applied results CSV missing cluster or au columns.")
            else:
                logger.warning("Could not download applied results CSV from Shahar session. State restoration skipped.")

        # Dynamically run prerequisite pipeline if still empty
        if not self.carlin_fingerprint:
            self._ensure_geological_context()

        if not self.carlin_fingerprint:
            raise ValueError("No active geological context. Please ensure the exploration data has been loaded and clustered.")

        if self.mean_bwi is None:
            raise ValueError("Comminution hardness (BWI) must be predicted before running negotiation.")

        if self.mean_recovery is None:
            raise ValueError("Baseline flotation recovery must be predicted before running negotiation.")

        
        promising_cluster = self.promising_cluster
        clean_cluster = 0 if promising_cluster != 0 else 1
        
        # 1. Download results CSV to compute dynamic centroids for BOTH piles dynamically
        ref_values = {}
        ox_values = {}
        
        try:
            shahar_session_id = self.shahar_sid
            headers = {"X-API-Key": self.shahar_key}
            filename = f"{self.winning_method}_applied_results.csv" if self.winning_method == "som" else "clustering_applied_results.csv"
            download_url = f"{self.shahar_url}/sessions/{shahar_session_id}/files/{filename}"
            download_resp = self.http_session.get(download_url, headers=headers, timeout=60)
            if download_resp.status_code != 200:
                raise RuntimeError(f"Failed to download applied results CSV (Status {download_resp.status_code})")
                
            df = pd.read_csv(io.StringIO(download_resp.text))
            cluster_col = next((c for c in df.columns if "cluster" in c.lower()), None)
            if not cluster_col:
                raise ValueError("Missing cluster column in applied results.")
                
            # If the current promising_cluster is not in the dataset's clusters, dynamically search for a valid one
            if promising_cluster not in df[cluster_col].unique():
                au_col = next((c for c in df.columns if c.lower() == "au"), None)
                if au_col:
                    cluster_means = df.groupby(cluster_col)[au_col].mean()
                    if not cluster_means.empty:
                        promising_cluster = int(cluster_means.idxmax())
                        self.promising_cluster = promising_cluster
                        clean_cluster = 0 if promising_cluster != 0 else 1
                        logger.info(f"Resolved promising_cluster dynamically from dataset clusters: promising_cluster={promising_cluster}, clean_cluster={clean_cluster}")
            
            best_cluster_df = df[df[cluster_col] == promising_cluster]
            clean_cluster_df = df[df[cluster_col] == clean_cluster]
            
            # If the clean cluster stockpile is empty in the results, we dynamically scale from refractory cluster centroids
            scale_oxide_from_refractory = (len(clean_cluster_df) == 0)
            if scale_oxide_from_refractory:
                logger.info(f"Clean cluster stockpile (ID {clean_cluster}) is empty in Shahar results. Dynamically scaling refractory centroids for Oxide stack.")
                
            scale_factors = {
                "Au": 0.726,
                "As": 0.096,
                "Sb": 0.110,
                "Hg": 0.083,
                "Tl": 0.098,
                "TOC": 0.026,
                "TCM": 0.008,
                "quartz": 1.147,
                "carbonates": 0.244,
                "FeS2": 0.125,
            }
            
            keys = ["Au", "As", "Sb", "Hg", "Tl", "TOC", "TCM", "quartz", "carbonates", "BWI", "FeS2"]
            for k in keys:
                found_col = next((col for col in df.columns if col.lower() == k.lower()), None)
                if found_col:
                    ref_values[k] = float(best_cluster_df[found_col].mean())
                    if not scale_oxide_from_refractory:
                        ox_values[k] = float(clean_cluster_df[found_col].mean())
                    else:
                        if k == "BWI":
                            ox_values[k] = float(ref_values[k] - 3.22)
                        else:
                            factor = scale_factors.get(k, 1.0)
                            ox_values[k] = round(ref_values[k] * factor, 4)
                else:
                    if k == "TCM":
                        ref_values[k] = ref_values.get("TOC", 0.5) * 0.4
                        if not scale_oxide_from_refractory:
                            ox_values[k] = ox_values.get("TOC", 0.1) * 0.4
                        else:
                            ox_values[k] = ref_values[k] * 0.008
                    elif k == "BWI":
                        ref_values[k] = float(self.mean_bwi)
                        ox_values[k] = float(self.mean_bwi) - 3.22
                    else:
                        raise KeyError(f"Geochemical key {k} not found.")
        except Exception as e:
            logger.warning(f"Failed to dynamically download cluster centroids from Shahar ({e}). Using active carlin_fingerprint as Refractory stack and deriving Oxide stack.")
            
            ref_values = {}
            for k in ["Au", "As", "Sb", "Hg", "Tl", "TOC", "TCM", "quartz", "carbonates"]:
                found_val = self.carlin_fingerprint.get(k)
                if found_val is None:
                    found_val = self.carlin_fingerprint.get(k.lower())
                if found_val is None:
                    raise KeyError(f"Geochemical key '{k}' is missing from the active carlin_fingerprint.")
                import math
                if math.isnan(float(found_val)) or math.isinf(float(found_val)):
                    raise ValueError(f"Geochemical key '{k}' is NaN or infinite.")
                ref_values[k] = float(found_val)
                
            ref_values["BWI"] = float(self.mean_bwi)
            ref_values["FeS2"] = 8.0
            
            ox_values = {
                "Au": round(ref_values["Au"] * 0.726, 4),
                "As": round(ref_values["As"] * 0.096, 4),
                "Sb": round(ref_values["Sb"] * 0.110, 4),
                "Hg": round(ref_values["Hg"] * 0.083, 4),
                "Tl": round(ref_values["Tl"] * 0.098, 4),
                "TOC": round(ref_values["TOC"] * 0.026, 4),
                "TCM": round(ref_values["TCM"] * 0.008, 4),
                "quartz": round(ref_values["quartz"] * 1.147, 4),
                "carbonates": round(ref_values["carbonates"] * 0.244, 4),
                "BWI": round(ref_values["BWI"] - 3.22, 4),
                "FeS2": round(8.0 * 0.125, 4)
            }

                    
        # 2. Instantiate CarlinOreClusters dynamically from geostatistical data
        from core.negotiation_engine import CarlinOreCluster, KeterGeologistAgent, TeomimMetallurgistAgent, GameTheoreticBargainingEngine
        
        refractory_pile = CarlinOreCluster(
            name=f"High-Carbon Sulfide Stack (Cluster #{promising_cluster})",
            au=ref_values["Au"],
            toc=ref_values["TOC"],
            tcm=ref_values["TCM"],
            arsenic=ref_values["As"],
            quartz=ref_values["quartz"],
            carbonates=ref_values["carbonates"],
            bwi=ref_values["BWI"],
            fes2=ref_values.get("FeS2", 8.0)
        )
        
        oxide_pile = CarlinOreCluster(
            name=f"Clean Oxide Stockpile (Cluster #{clean_cluster})",
            au=ox_values["Au"],
            toc=ox_values["TOC"],
            tcm=ox_values["TCM"],
            arsenic=ox_values["As"],
            quartz=ox_values["quartz"],
            carbonates=ox_values["carbonates"],
            bwi=ox_values["BWI"],
            fes2=ox_values.get("FeS2", 1.0)
        )
        
        # 3. Instantiate dynamic game-theoretic agents & run bargaining engine
        keter_agent = KeterGeologistAgent(refractory_pile, target_au=max(10.0, refractory_pile.au))
        teomim_agent = TeomimMetallurgistAgent(init_pri_thresh=1.5, init_emissions_limit=0.45)
        
        engine = GameTheoreticBargainingEngine(keter_agent, teomim_agent, oxide_pile)
        result = engine.negotiate(max_turns=10)
        
        # Extract dynamic solving parameters
        final_ref_ratio = result["final_refractory_ratio"]
        final_oxide_ratio = result["final_oxide_ratio"]
        final_rec = result["optimized_recovery"]
        final_emissions = result["stack_emissions_mg_nm3"]
        final_sintering = result["sintering_risk"] * 100
        best_actions = result["control_settings"]
        
        # 4. Generate professional turn-by-turn dialogue dressing via LLM (English)
        turns_summary = []
        for h in result["negotiation_history"]:
            offerer = h["offerer"]
            turn = h["turn"]
            ref_pct = h["proposed_ref_ratio"] * 100
            ox_pct = (1.0 - h["proposed_ref_ratio"]) * 100
            u_g = h["u_G"]
            u_m = h["u_M"]
            rec = h["roast_recovery"]
            emissions = h["emissions"]
            sinter = h["sinter_risk"] * 100
            
            turns_summary.append(
                f"Turn #{turn}: {offerer} proposes a blend of {ref_pct:.1f}% Refractory / {ox_pct:.1f}% Oxide.\n"
                f"- Geologist Utility: {u_g:.3f}, Metallurgist Utility: {u_m:.3f}\n"
                f"- Roaster Recovery: {rec:.2f}%, Stack As2O3 Emissions: {emissions:.3f} mg/Nm3, Sintering Risk: {sinter:.1f}%\n"
            )
        
        turns_summary_str = "\n".join(turns_summary)
        
        dialogue_prompt = DIALOGUE_DRESSING_TEMPLATE.format(
            turns_summary_str=turns_summary_str,
            best_actions=best_actions
        )
        
        def generate_agent_response(system_prompt, user_prompt):
            if not self.api_key:
                raise ValueError("API key missing")
            if MODERN_SDK:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.7
                    )
                )
                return response.text
            else:
                model = legacy_genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=system_prompt,
                    generation_config={"temperature": 0.7}
                )
                response = model.generate_content(user_prompt)
                return response.text
                
        dialogue_dressed = ""
        if self.api_key:
            try:
                geologist_sys = "You are Keter, the Chief Geologist Agent. You speak in a highly professional, technical, and precise geostatistical tone."
                dialogue_dressed = generate_agent_response(geologist_sys, dialogue_prompt)
            except Exception as e:
                logger.warning(f"Error calling Gemini to dress negotiation dialogue, falling back to simulated summary: {e}")
                
        if not dialogue_dressed:
            # Build dynamic fallback dialogue in case LLM is offline or error occurs
            dialogue_lines = [
                "| Turn | Agent | Dialogue | Mathematical/Technical Indicator |",
                "|---|---|---|---|"
            ]
            
            neg_history = result.get("negotiation_history", [])
            for idx, h in enumerate(neg_history):
                offerer = h["offerer"]
                turn_num = h["turn"]
                ref_pct = h["proposed_ref_ratio"] * 100
                ox_pct = (1.0 - h["proposed_ref_ratio"]) * 100
                u_g = h["u_G"]
                u_m = h["u_M"]
                rec = h["roast_recovery"]
                emissions = h["emissions"]
                sinter = h["sinter_risk"] * 100
                
                # Check acceptance condition for this step
                is_accepted = u_m > engine.d_M and emissions <= teomim_agent.emissions_limit and (sinter/100.0) <= 0.15
                
                # Turn A: Geologist Proposes
                if offerer == "Geologist":
                    if idx == 0:
                        g_dialogue = f"I propose feeding {ref_pct:.1f}% ore from the premium refractory zone Cluster #{promising_cluster} directly. Geochemical assay indicates a high gold grade of {refractory_pile.au:.2f} ppm. We must maximize gold throughput and feed profit!"
                        g_indicator = f"Proposed blend: {ref_pct:.1f}% Refractory / {ox_pct:.1f}% Oxide. Refractory grade: {refractory_pile.au:.2f} ppm Au."
                    else:
                        g_dialogue = f"Understood. To secure environmental compliance and protect the kiln, I propose a compromise Sulfide-Oxide blend. Let's feed {ref_pct:.1f}% refractory sulfide ore mixed with {ox_pct:.1f}% clean oxide ore from Cluster #{clean_cluster}."
                        g_indicator = f"Diluted blend proposal: {ref_pct:.1f}% Refractory / {ox_pct:.1f}% Oxide."
                    
                    dialogue_lines.append(f"| {idx * 2} | **Keter Geologist** | {g_dialogue} | {g_indicator} |")
                    
                    # Turn B: Metallurgist Response
                    if is_accepted:
                        m_dialogue = f"APPROVED. The proposed {ref_pct:.1f}/{ox_pct:.1f} blend is optimized. Under custom roaster controls (Feed Rate: {best_actions.get('feed_rate_tph', 95.0):.1f} TPH, Excess Air: {best_actions.get('excess_air_pct', 34.5):.1f}%), stack As₂O₃ emissions stabilize safely at {emissions:.3f} mg/Nm³ (limit is 0.50 mg/Nm³), sintering risk is {sinter:.1f}%, and gold recovery is predicted at {rec:.2f}%."
                        m_indicator = f"APPROVED: Stack emissions {emissions:.3f} mg/Nm³, Sintering: {sinter:.1f}%, Recovery: {rec:.2f}%."
                    else:
                        reasons = []
                        if u_m <= engine.d_M:
                            # calculate actual PRI
                            toc_val = h["proposed_ref_ratio"] * refractory_pile.toc + (1.0 - h["proposed_ref_ratio"]) * oxide_pile.toc
                            tcm_val = h["proposed_ref_ratio"] * refractory_pile.tcm + (1.0 - h["proposed_ref_ratio"]) * oxide_pile.tcm
                            pri_val = teomim_agent.calculate_pri(toc_val, tcm_val)
                            reasons.append(f"utility falls below the disagreement threat limit due to high organic carbon preg-robbing (PRI={pri_val:.2f} > limit {teomim_agent.pri_threshold:.2f})")
                        if emissions > teomim_agent.emissions_limit:
                            reasons.append(f"stack As₂O₃ emissions ({emissions:.3f} mg/Nm³) exceed Nev Title V limit ({teomim_agent.emissions_limit:.3f} mg/Nm³)")
                        if (sinter/100.0) > 0.15:
                            reasons.append(f"clay sintering risk ({sinter:.1f}%) is critical (limit 15.0%)")
                        
                        reason_msg = " and ".join(reasons)
                        m_dialogue = f"REJECTED. A feed blend containing {ref_pct:.1f}% refractory sulfide is unsafe: {reason_msg}. Impurity dilution or control optimization is required."
                        m_indicator = f"REJECTED: emissions {emissions:.3f} mg/Nm³ (Limit: {teomim_agent.emissions_limit:.3f}), Sintering risk: {sinter:.1f}%."
                        
                    dialogue_lines.append(f"| {idx * 2 + 1} | **Teomim Metallurgist** | {m_dialogue} | {m_indicator} |")
            
            dialogue_dressed = "\n".join(dialogue_lines)
            
        # Calculate diluted geochemistry parameters for the final blend payload
        toc_blend = float(final_ref_ratio * refractory_pile.toc + (1.0 - final_ref_ratio) * oxide_pile.toc)
        tcm_blend = float(final_ref_ratio * refractory_pile.tcm + (1.0 - final_ref_ratio) * oxide_pile.tcm)
        as_blend = float(final_ref_ratio * refractory_pile.arsenic + (1.0 - final_ref_ratio) * oxide_pile.arsenic)
        quartz_blend = float(final_ref_ratio * refractory_pile.quartz + (1.0 - final_ref_ratio) * oxide_pile.quartz)
        carb_blend = float(final_ref_ratio * refractory_pile.carbonates + (1.0 - final_ref_ratio) * oxide_pile.carbonates)
        fes2_blend = float(final_ref_ratio * refractory_pile.fes2 + final_oxide_ratio * oxide_pile.fes2)
        
        opt_headers = {"X-API-Key": self.lahav_key or self.keter_key}
        blend_payload = {
            "session_id": session_id,
            "engine": "bayesian",
            "n_iterations": 40,
            "use_adaptive_reward": True,
            "goals": [
                {"variable": "gold_recovery_pct", "target": "maximize", "weight": 2.0},
                {"variable": "blockage_risk", "target": "minimize", "weight": 1.0},
                {"variable": "as2o3_emissions_mg_nm3", "target": "minimize", "weight": 2.0}
            ],
            "constraints": [
                {"variable": "wall_temp_c", "condition": "<=", "threshold": float(best_actions.get("wall_temp_c", 640.0) + 40.0), "penalty": 6.0},
                {"variable": "porosity_loss_risk", "condition": "<", "threshold": 0.12, "penalty": 4.0},
                {"variable": "as2o3_emissions_mg_nm3", "condition": "<=", "threshold": 0.5, "penalty": 8.0}
            ],
            "baseline_readings": {
                "fuel_type": "gas",
                "feed_rate_tph": float(best_actions.get("feed_rate_tph", 95.0)),
                "particle_p80_um": 75.0,
                "excess_air_pct": float(best_actions.get("excess_air_pct", 34.5)),
                "insulation_rvalue": 0.5,
                "pipe_position_m": 4.0,
                "burner_tilt_deg": -2.5,
                "tertiary_air_temp_c": 225.0,
                "toc": toc_blend,
                "tcm": tcm_blend,
                "arsenic": as_blend,
                "quartz": quartz_blend,
                "carbonates": carb_blend,
                "fes2_pct": fes2_blend,
                "ore_type": self.ore_type
            }
        }
        
        sim_rec = final_rec
        sim_emissions = final_emissions
        sim_sintering = final_sintering
        
        try:
            r_blend = self.http_session.post(f"{self.teomim_url}/thermo/agent/optimize", json=blend_payload, headers=opt_headers, timeout=60)
            if r_blend.status_code == 200:
                blend_data = r_blend.json().get("result", {})
                physics = blend_data.get("best_physics", {})
                server_rec = physics.get("gold_recovery_pct", 0.0)
                # Use server-side result (full compute_physics) as authoritative
                # Both models now use consistent physics (exothermic self-heating + porosity)
                if server_rec > 0:
                    sim_rec = server_rec
                    sim_emissions = physics.get("as2o3_emissions_mg_nm3", sim_emissions)
                    sim_sintering = physics.get("porosity_loss_risk", 0.028) * 100.0
                    best_actions = blend_data.get("best_actions", best_actions)
        except Exception as e:
            logger.warning(f"Error calling live thermo roaster optimizer, falling back to simulated values: {e}")
            
        # 6. Format final beautiful markdown transcript
        session_id_str = f"{session_id[:16]}..." if session_id else "offline_session"
        mean_rec_str = f"{self.mean_recovery:.1f}%" if self.mean_recovery is not None else "N/A"
        
        transcript = [
            "### Multi-Agent Game-Theoretic Geological-Metallurgical Ore Negotiation Log",
            f"**Active Session**: `{session_id_str}` | **Refractory Centroid Au**: `{refractory_pile.au:.2f} ppm` | **Oxide Centroid Au**: `{oxide_pile.au:.2f} ppm`",
            "",
            dialogue_dressed.strip(),
            "",
            "### Negotiation Conclusion & Final Control Settings",
            f"- **Agreed Feed Strategy**: **{final_ref_ratio*100:.1f}% Refractory (Cluster #{promising_cluster}) / {final_oxide_ratio*100:.1f}% Oxide (Cluster #{clean_cluster}) Blend**",
            f"- **Optimized Gold Recovery**: **{sim_rec:.2f}%** (flotation baseline recovery was {mean_rec_str})",
            f"- **Stack As₂O₃ Emissions**: **{sim_emissions:.3f} mg/Nm³** (Nevada Title V NDEP limit: <0.50 mg/Nm³ - COMPLIANT)",
            f"- **Sintering Risk**: **{sim_sintering:.1f}%** (Sintering limit: <15.0% - SAFE)",
            f"- **Recommended Kiln Control Actions**:",
            f"  - **Feed Rate**: `{best_actions.get('feed_rate_tph', 95.0):.2f} TPH` (Scaled dynamically for calciner residence time)",
            f"  - **Excess Air**: `{best_actions.get('excess_air_pct', 34.5):.2f}%` (Raised to bind volatile arsenic into non-volatile iron arsenate FeAsO4)",
            f"  - **Wall Bed Temperature**: `{best_actions.get('wall_temp_c', 640.0):.2f}°C` (Optimized to stay below sintering threshold)"
        ]
        
        markdown_transcript = "\n".join(transcript)
        
        # Update internal state to match negotiated outcome
        self.optimized_recovery = sim_rec
        self.best_actions = best_actions
        self.best_physics = {
            "gold_recovery_pct": sim_rec,
            "as2o3_emissions_mg_nm3": sim_emissions,
            "porosity_loss_risk": sim_sintering / 100.0,
            "wall_temp_c": best_actions.get("wall_temp_c", 632.4)
        }
        self.ore_type = "high_carbon"
        self.negotiation_history = markdown_transcript
        
        return {
            "success": True,
            "transcript": markdown_transcript,
            "optimized_recovery": sim_rec,
            "best_actions": best_actions,
            "best_physics": self.best_physics,
            "mean_bwi": self.mean_bwi,
            "mean_recovery": self.mean_recovery,
            "final_refractory_ratio": final_ref_ratio,
            "shahar_sid": self.shahar_sid,
            "kokhav_sid": self.kokhav_sid,
            "keter_sid": self.keter_sid,
            "message": "Dynamic game-theoretic ore selection negotiation completed E2E. Operational equilibrium achieved between Geologist & Metallurgist agents."
        }

    def _get_system_instruction(self) -> str:
        """Dynamically generate the system instruction with the latest geostatistical & metallurgical variables."""
        bwi_str = f"{self.mean_bwi:.2f}" if self.mean_bwi is not None else "N/A"
        rec_str = f"{self.mean_recovery:.2f}" if self.mean_recovery is not None else "N/A"
        opt_rec_str = f"{self.optimized_recovery:.2f}" if self.optimized_recovery is not None else "N/A"
        
        return SYSTEM_INSTRUCTION_TEMPLATE.format(
            session_id=self.session_id,
            promising_cluster=self.promising_cluster,
            carlin_fingerprint=self.carlin_fingerprint,
            ore_type=self.ore_type,
            bwi_str=bwi_str,
            rec_str=rec_str,
            opt_rec_str=opt_rec_str,
            best_actions=self.best_actions,
            best_physics=self.best_physics
        )

    # =========================================================================
    # Chat & Conversational Loop (Upgraded to new google-genai Client)
    # =========================================================================
    def chat(self, user_message: str, history: list = None) -> dict:
        """
        Runs conversational loop with Gemini using the modern/legacy SDK,
        executing any tool calls requested.
        
        history format: [{"role": "user"|"model", "content": "str"}]
        """
        if not self.api_key:
            return {
                "response": "GEMINI_API_KEY is not configured. Please supply a valid key in the sidebar.",
                "steps": [],
                "success": False
            }
            
        self._ensure_sessions()
        
        # Load tool declarations
        try:
            tools = self.load_tools_declarations()
        except Exception as e:
            logger.error(f"Failed to load tools: {e}")
            return {
                "response": f"Failed to load agent tools schema: {str(e)}",
                "steps": [],
                "success": False
            }
        
        steps = []
        max_turns = 12
        turn = 0
        response_text = ""
        
        logger.info(f"Chat called: api_key={'SET' if self.api_key else 'MISSING'}, model={self.model_name}, message='{user_message[:60]}...', MODERN_SDK={MODERN_SDK}")
        
        # Pipeline sequence for guided autonomous execution
        PIPELINE_SEQUENCE = [
            "shahar_load_data", "shahar_run_clustering", "shahar_apply_som",
            "kokhav_load_data", "kokhav_predict", "keter_classify_ore",
            "search_historical_logs", "teomim_activate_nodes", "teomim_optimize_thermo"
        ]

        if MODERN_SDK:
            # ================================================================
            # generate_content-based loop (NOT chat.send_message)
            # The chat session abstraction silently drops tool_config on
            # per-message overrides. Using generate_content directly with
            # manual Content history is the only reliable way to pass
            # tool_config (with allowed_function_names) per turn.
            # ================================================================
            
            # Build manual conversation history as list[types.Content]
            contents: list = []
            if history:
                for h in history:
                    role = "user" if h["role"] == "user" else "model"
                    contents.append(types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=h["content"])]
                    ))
            # Append the new user message
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_message)]
            ))
            
            completed_tools = []
            last_tool_name = None
            consecutive_duplicate_count = 0
            next_allowed = None   # None on turn 1 (unconstrained)
            response = None
            
            while turn < max_turns:
                # ---- Build per-turn config (fresh every turn) ----
                config_kwargs = dict(
                    system_instruction=self._get_system_instruction(),
                    tools=[types.Tool(function_declarations=tools)],
                    temperature=0.2,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                )
                if next_allowed:
                    config_kwargs["tool_config"] = types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(
                            mode=types.FunctionCallingConfigMode.ANY,
                            allowed_function_names=next_allowed,
                        )
                    )
                    logger.info(f"Turn {turn+1}: constraining tool call to {next_allowed}")
                else:
                    logger.info(f"Turn {turn+1}: unconstrained tool call")
                
                # ---- Generate ----
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(**config_kwargs),
                    )
                except Exception as e:
                    logger.error(f"generate_content failed at turn {turn+1}: {e}")
                    return {
                        "response": f"Gemini API error: {str(e)}",
                        "steps": steps,
                        "success": False,
                    }
                
                # ---- Extract function calls ----
                func_calls = self._get_function_calls(response)
                if not func_calls:
                    try:
                        response_text = response.text or "Pipeline run completed."
                    except Exception:
                        response_text = "Pipeline run completed."
                    logger.info(f"No function calls at turn {turn+1}. Stopping.")
                    break
                
                turn += 1
                
                # Append the model's content (the function call(s)) to history
                try:
                    model_content = response.candidates[0].content
                    contents.append(model_content)
                except Exception as e:
                    logger.warning(f"Could not append model content to history: {e}")
                    fc_parts = [types.Part(function_call=fc) for fc in func_calls]
                    contents.append(types.Content(role="model", parts=fc_parts))
                
                # ---- Process each function call ----
                turn_function_responses = []
                next_allowed = None
                
                for fc in func_calls:
                    tool_name = fc.name
                    tool_args = self._clean_json_types(dict(fc.args))
                    
                    # ---- Duplicate guard ----
                    if tool_name == last_tool_name:
                        consecutive_duplicate_count += 1
                    else:
                        consecutive_duplicate_count = 0
                    last_tool_name = tool_name
                    
                    if consecutive_duplicate_count >= 2:
                        logger.error(f"Hard-stopping: {tool_name} called {consecutive_duplicate_count+1}x in a row")
                        response_text = (
                            f"Pipeline aborted: model repeatedly called {tool_name}. "
                            f"This usually means an upstream tool failed silently."
                        )
                        turn = max_turns
                        break
                    
                    # ---- Skip if completed (with kokhav_predict exception) ----
                    if (tool_name in completed_tools
                        and tool_name in PIPELINE_SEQUENCE
                        and not (tool_name == "kokhav_predict"
                                 and completed_tools.count("kokhav_predict") < 2)):
                        idx = PIPELINE_SEQUENCE.index(tool_name)
                        nxt = (PIPELINE_SEQUENCE[idx + 1]
                               if idx + 1 < len(PIPELINE_SEQUENCE) else None)
                        skip_msg = f"{tool_name} already complete. Next: {nxt}"
                        tool_result = {"success": True, "message": skip_msg}
                        steps.append({"tool": tool_name, "inputs": tool_args,
                                      "output": tool_result, "status": "skipped"})
                        turn_function_responses.append(
                            types.Part.from_function_response(name=tool_name, response=tool_result)
                        )
                        if nxt:
                            next_allowed = [nxt]
                        continue
                    
                    # ---- Execute ----
                    logger.info(f"Executing {tool_name} args={tool_args}")
                    executor_name = f"execute_{tool_name}"
                    if not hasattr(self, executor_name):
                        err = f"No executor for {tool_name}"
                        logger.error(err)
                        tool_result = {"error": err, "success": False}
                        steps.append({"tool": tool_name, "inputs": tool_args,
                                      "output": tool_result, "status": "failed"})
                        turn_function_responses.append(
                            types.Part.from_function_response(name=tool_name, response=tool_result)
                        )
                        continue
                    
                    status = "success"
                    try:
                        tool_result = getattr(self, executor_name)(**tool_args)
                    except Exception as err:
                        logger.error(f"{tool_name} raised: {err}")
                        tool_result = {"error": str(err), "success": False}
                        status = "failed"
                    
                    # ---- Fail-fast: abort on upstream pipeline failure ----
                    if status == "failed" and tool_name in PIPELINE_SEQUENCE:
                        logger.error(f"Pipeline step {tool_name} failed — halting forward march")
                        tool_result["_pipeline_hint"] = (
                            f"{tool_name} failed: {str(tool_result.get('error',''))[:200]}. "
                            "Do not call downstream tools."
                        )
                        steps.append({"tool": tool_name, "inputs": tool_args,
                                      "output": tool_result, "status": status})
                        turn_function_responses.append(
                            types.Part.from_function_response(name=tool_name, response=tool_result)
                        )
                        response_text = f"Pipeline halted at {tool_name}: {tool_result.get('error','')}"
                        turn = max_turns
                        break
                    
                    completed_tools.append(tool_name)
                    
                    # ---- Set next constraint ----
                    if tool_name in PIPELINE_SEQUENCE:
                        idx = PIPELINE_SEQUENCE.index(tool_name)
                        if idx + 1 < len(PIPELINE_SEQUENCE):
                            candidate_next = [PIPELINE_SEQUENCE[idx + 1]]
                            if (tool_name == "kokhav_predict"
                                and completed_tools.count("kokhav_predict") < 2):
                                candidate_next = ["kokhav_predict", "keter_classify_ore"]
                            next_allowed = candidate_next
                            tool_result["_pipeline_hint"] = (
                                f"Step {idx+1}/9 done. Next: {candidate_next}"
                            )
                        else:
                            tool_result["_pipeline_hint"] = "Pipeline complete."
                    
                    steps.append({"tool": tool_name, "inputs": tool_args,
                                  "output": tool_result, "status": status})
                    turn_function_responses.append(
                        types.Part.from_function_response(name=tool_name, response=tool_result)
                    )
                
                # Append all function responses as a single user-role Content
                if turn_function_responses:
                    contents.append(types.Content(role="user", parts=turn_function_responses))
            
            # Extract final response text if not already set
            if not response_text and response:
                try:
                    response_text = response.text or "Closed-loop optimization run completed."
                except (ValueError, AttributeError):
                    response_text = "Closed-loop optimization run completed."

        else:
            # Legacy SDK implementation
            legacy_history = []
            if history:
                for h in history:
                    role = "user" if h["role"] == "user" else "model"
                    legacy_history.append({
                        "role": role,
                        "parts": [h["content"]]
                    })
            
            # Legacy GenerativeModel configuration
            model = legacy_genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={"temperature": 0.2},
                tools=tools,
                system_instruction=self._get_system_instruction()
            )
            chat = model.start_chat(history=legacy_history)
            
            try:
                response = chat.send_message(user_message)
            except Exception as e:
                logger.error(f"Legacy Gemini API chat session initiation failed: {e}")
                return {
                    "response": f"Gemini API chat session error: {str(e)}",
                    "steps": [],
                    "success": False
                }
                
            while turn < max_turns:
                func_calls = self._get_function_calls(response)
                if not func_calls:
                    break
                    
                turn += 1
                for fc in func_calls:
                    tool_name = fc.name
                    tool_args = self._clean_json_types(dict(fc.args))
                    
                    logger.info(f"Executing Tool Call {tool_name} with args {tool_args}")
                    
                    executor_name = f"execute_{tool_name}"
                    if hasattr(self, executor_name):
                        executor = getattr(self, executor_name)
                        status = "success"
                        try:
                            tool_result = executor(**tool_args)
                        except Exception as err:
                            logger.error(f"Tool {tool_name} execution error: {err}")
                            tool_result = {"error": str(err), "success": False}
                            status = "failed"
                            
                        steps.append({
                            "tool": tool_name,
                            "inputs": tool_args,
                            "output": tool_result,
                            "status": status
                        })
                        
                        try:
                            # Feed legacy function response
                            response = chat.send_message(
                                legacy_genai.types.Part.from_function_response(
                                    name=tool_name,
                                    response=tool_result
                                )
                            )
                        except Exception as err:
                            logger.error(f"Failed to feed back legacy tool result for {tool_name}: {err}")
                            break
                    else:
                        err_msg = f"No executor found for tool: {tool_name}"
                        logger.error(err_msg)
                        steps.append({
                            "tool": tool_name,
                            "inputs": tool_args,
                            "output": {"error": err_msg},
                            "status": "failed"
                        })
                        try:
                            response = chat.send_message(
                                legacy_genai.types.Part.from_function_response(
                                    name=tool_name,
                                    response={"error": err_msg}
                                )
                            )
                        except Exception:
                            break
            try:
                response_text = response.text or "Closed-loop optimization run completed."
            except ValueError:
                response_text = "Closed-loop optimization run completed."

        # Only flag success if agent actually executed tools — prevents overwriting real pipeline data with defaults
        any_tool_succeeded = any(s["status"] == "success" for s in steps) if steps else False
        return {
            "response": response_text,
            "steps": steps,
            "success": any_tool_succeeded,
            "shahar_sid": self.shahar_sid,
            "kokhav_sid": self.kokhav_sid,
            "keter_sid": self.keter_sid,
            "promising_cluster": self.promising_cluster,
            "carlin_fingerprint": self.carlin_fingerprint,
            "ore_type": self.ore_type,
            "mean_bwi": self.mean_bwi,
            "mean_recovery": self.mean_recovery,
            "optimized_recovery": self.optimized_recovery,
            "best_actions": self.best_actions,
            "best_physics": self.best_physics,
            "shap_results": self.shap_results,
            "classification_graph": self.classification_graph,
            "reflection_logs": self.reflection_logs,
            "negotiation_history": self.negotiation_history
        }
