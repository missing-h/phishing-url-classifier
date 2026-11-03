import streamlit as st
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
import joblib
import os
import warnings
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

warnings.filterwarnings("ignore")

# ----------------------
# Cargar modelos (desde el mismo directorio)
# ----------------------
MODEL_URL_PATH = "phishing_model.pkl"
MODEL_EMAIL_PATH = "phishing_email_model.pkl"

def load_models():
    url_model = None
    email_model = None
    if os.path.exists(MODEL_URL_PATH):
        try:
            url_model = joblib.load(MODEL_URL_PATH)
        except Exception as e:
            st.warning(f"No se pudo cargar {MODEL_URL_PATH}: {e}")
    else:
        st.info(f"No se encontró {MODEL_URL_PATH} — predicción de URLs deshabilitada.")
    if os.path.exists(MODEL_EMAIL_PATH):
        try:
            email_model = joblib.load(MODEL_EMAIL_PATH)
        except Exception as e:
            st.warning(f"No se pudo cargar {MODEL_EMAIL_PATH}: {e}")
    else:
        st.info(f"No se encontró {MODEL_EMAIL_PATH} — predicción de Emails deshabilitada.")
    return url_model, email_model

url_model, email_model = load_models()


URL_FEATURES_EXPECTED = [
    'url_length','domain_length','num_dots','num_hyphens_domain','num_at',
    'num_qm','num_eq','num_slash','has_https','uses_http','has_www','has_ip',
    'num_subdomains','suspicious_words_domain','suspicious_path_words',
    'suspicious_path','has_port','is_shortened','fake_domain_score',
    'known_benign_domain','benign_indicators'
]

EMAIL_FEATURES_EXPECTED = [
    'email_length','num_urls','num_exclamation','num_question',
    'has_uppercase_excess','suspicious_words','benign_mentions',
    'suspicious_links_score','phishing_email_score','benign_email_indicators',
    'benign_verification_code'
]

# Use model.feature_names_in_ if available (keeps exact ordering)
if url_model is not None and hasattr(url_model, "feature_names_in_"):
    URL_FEATURES = list(url_model.feature_names_in_)
else:
    URL_FEATURES = URL_FEATURES_EXPECTED.copy()

if email_model is not None and hasattr(email_model, "feature_names_in_"):
    EMAIL_FEATURES = list(email_model.feature_names_in_)
else:
    EMAIL_FEATURES = EMAIL_FEATURES_EXPECTED.copy()

# ----------------------
# Extractors: producen SOLO las columnas que el modelo espera
# ----------------------
def extract_features_for_model(url: str):
    """
    Extrae únicamente las features esperadas por el modelo de URLs.
    Devuelve un dict con todas las keys de URL_FEATURES (si falla, devuelve 0s).
    """
    try:
        parsed = urlparse(str(url).strip())
        domain = parsed.netloc or (str(url).split('/')[0] if '/' in str(url) else str(url))
        path = parsed.path or ""
        base_domain = domain.replace('www.', '').lower()

        feats = {}
        feats['url_length'] = len(str(url))
        feats['domain_length'] = len(domain)
        feats['num_dots'] = domain.count('.')
        feats['num_hyphens_domain'] = domain.count('-')
        feats['num_at'] = str(url).count('@')
        feats['num_qm'] = str(url).count('?')
        feats['num_eq'] = str(url).count('=')
        feats['num_slash'] = str(url).count('/')
        feats['has_https'] = 1 if str(url).lower().startswith("https") else 0
        feats['uses_http'] = 1 if str(url).lower().startswith("http://") else 0
        feats['has_www'] = 1 if "www." in domain.lower() else 0
        feats['has_ip'] = 1 if re.search(r"(\d{1,3}\.){3}\d{1,3}", domain) else 0

        # num_subdomains: cantidad de partes menos el dominio base (simple)
        feats['num_subdomains'] = max(0, domain.count('.') - 1)

        feats['suspicious_words_domain'] = len(re.findall(r"(verify|secure|update|account|bank|paypal)", domain.lower()))
        feats['suspicious_path_words'] = len(re.findall(r"(login|reset|confirm|secure|signin)", path.lower()))
        feats['suspicious_path'] = 1 if any(w in path.lower() for w in ["login","reset","confirm","signin"]) else 0
        feats['has_port'] = 1 if ":" in domain and not domain.endswith(":") else 0
        feats['is_shortened'] = 1 if re.search(r"(bit\.ly|tinyurl|t\.co|goo\.gl|ow\.ly)", str(url).lower()) else 0
        feats['fake_domain_score'] = domain.count('-')
        trusted_list = ['google.com','microsoft.com','amazon.com','paypal.com','outlook.com','gmail.com','apple.com']
        feats['known_benign_domain'] = 1 if base_domain in trusted_list else 0

        # benign_indicators: small heuristic but lo dejamos como 0/1+ values
        benign_ind = 0
        if feats['known_benign_domain'] == 1:
            benign_ind += 2
        if feats['has_https'] == 1 and feats['has_ip'] == 0:
            benign_ind += 1
        if feats['has_www'] == 1:
            benign_ind += 1
        feats['benign_indicators'] = benign_ind

        # ensure all keys present (fill missing with 0)
        for k in URL_FEATURES:
            if k not in feats:
                feats[k] = 0

        # reorder according to URL_FEATURES when converting to DataFrame
        return {k: feats.get(k, 0) for k in URL_FEATURES}

    except Exception:
        return {k: 0 for k in URL_FEATURES}


def extract_email_features_for_model(text: str, url_model_local=None):
    """
    Extrae únicamente las features esperadas por el modelo de emails.
    suspicious_links_score: si hay URL model, calcula promedio de probabilidades de URLs (reindexando correctamente).
    """
    try:
        txt = str(text)
        txt_lower = txt.lower()

        feats = {}
        feats['email_length'] = len(txt)
        urls = re.findall(r'https?://\S+', txt_lower)
        feats['num_urls'] = len(urls)
        feats['num_exclamation'] = txt.count('!')
        feats['num_question'] = txt.count('?')
        feats['has_uppercase_excess'] = 1 if sum(1 for c in txt if c.isupper()) > 15 else 0
        feats['suspicious_words'] = len(re.findall(r"(urgente|alerta|verificar|suspendida|cuenta|password|confirmar|problema)", txt_lower))
        feats['benign_mentions'] = len(re.findall(r"(gracias|soporte|equipo|customer service|atentamente)", txt_lower))

        # suspicious_links_score: promedio de probabilidades de URLs (si existe url_model_local)
        if url_model_local is not None and len(urls) > 0:
            url_feats_list = [extract_features_for_model(u) for u in urls]
            url_df = pd.DataFrame(url_feats_list)
            # reindex to url_model_local.feature_names_in_ if available, else to URL_FEATURES
            if hasattr(url_model_local, "feature_names_in_"):
                url_df = url_df.reindex(columns=list(url_model_local.feature_names_in_), fill_value=0)
            else:
                url_df = url_df.reindex(columns=URL_FEATURES, fill_value=0)
            try:
                probs = url_model_local.predict_proba(url_df)[:,1]
                feats['suspicious_links_score'] = float(np.mean(probs))
            except Exception:
                feats['suspicious_links_score'] = 0.0
        else:
            feats['suspicious_links_score'] = 0.0

        feats['phishing_email_score'] = feats['suspicious_words'] * 2 + feats['num_exclamation']
        feats['benign_email_indicators'] = feats['benign_mentions']
        feats['benign_verification_code'] = len(re.findall(r"(código|code|verification code|tu código)", txt_lower))

        # ensure all keys present
        for k in EMAIL_FEATURES:
            if k not in feats:
                feats[k] = 0

        return {k: feats.get(k, 0) for k in EMAIL_FEATURES}

    except Exception:
        return {k: 0 for k in EMAIL_FEATURES}


# ----------------------
# Streamlit UI setup
# ----------------------
st.set_page_config(page_title="PhishGuard AI", page_icon="🔒", layout="wide")
st.title("🔒 PhishGuard AI — Detector de Phishing con IA (URLs & Emails)")
st.markdown("Analiza URLs o Emails para detectar phishing. Basado en Random Forest con features personalizadas.")

st.sidebar.header("Configuración")
threshold = st.sidebar.slider("Threshold de phishing (URL)", 0.0, 1.0, 0.20, 0.01)
threshold_email = st.sidebar.slider("Threshold de phishing (Email)", 0.0, 1.0, 0.30, 0.01)
use_rule = st.sidebar.checkbox("Aplicar regla heurística (benign indicators)", True)

if 'history' not in st.session_state:
    st.session_state['history'] = pd.DataFrame(columns=[
        'id','type','raw','timestamp','prob_phishing','pred','final_pred','notes','label'
    ])

tab1, tab2, tab3 = st.tabs(["🔍 Analizar Individual", "📁 Carga Masiva", "📊 Estadísticas"])
# ----------------------
# TAB 1: Analizar Individual
# ----------------------
with tab1:
    st.header("Analizar Individualmente")
    mode = st.radio("Tipo de entrada", ["URL", "Email"])

    if mode == "URL":
        url_input = st.text_input("Ingresa la URL:")
        if st.button("Detectar URL"):
            if not url_input or url_input.strip() == "":
                st.error("Ingresa una URL válida.")
            elif url_model is None:
                st.error("Modelo de URLs no cargado. Coloca phishing_model.pkl en el directorio.")
            else:
                feats = extract_features_for_model(url_input)
                df_feats = pd.DataFrame([feats])
                # Reindex en el orden exacto que espera el modelo:
                if hasattr(url_model, "feature_names_in_"):
                    df_feats = df_feats.reindex(columns=list(url_model.feature_names_in_), fill_value=0)
                else:
                    df_feats = df_feats.reindex(columns=URL_FEATURES, fill_value=0)

                try:
                    proba = float(url_model.predict_proba(df_feats)[:,1][0])
                except Exception:
                    # fallback a predict (si el modelo no tiene predict_proba)
                    proba = float(url_model.predict(df_feats)[0])

                pred = int(proba > threshold)
                final_pred = pred
                notes = ""
                if use_rule:
                    # Aplicar regla heurística segura (ejemplo)
                    if int(feats.get('benign_indicators',0)) >= 3 and int(feats.get('fake_domain_score',0)) == 0:
                        final_pred = 0
                        notes = "Regla aplicada: indicadores benignos"
                st.markdown(f"**Probabilidad phishing:** {proba:.2%}")
                st.markdown(f"**Predicción:** {'🔴 Phishing' if final_pred==1 else '🟢 Benigno'}")
                if notes:
                    st.info(notes)
                # --- Guardar en historial ---
                new_row = {
                    'id': len(st.session_state['history']) + 1,
                    'type': 'url',
                    'raw': url_input,
                    'timestamp': pd.Timestamp.now(),
                    'prob_phishing': proba,
                    'pred': pred,
                    'final_pred': final_pred,
                    'notes': notes,
                    'label': np.nan
                }
                st.session_state['history'] = pd.concat(
                    [st.session_state['history'], pd.DataFrame([new_row])],
                    ignore_index=True
                )

                    

    else:  # Email
        email_input = st.text_area("Pega asunto + cuerpo del email:")
        if st.button("Detectar Email"):
            if not email_input or email_input.strip() == "":
                st.error("Ingresa el texto del email.")
            elif email_model is None:
                st.error("Modelo de Emails no cargado. Coloca phishing_email_model.pkl en el directorio.")
            else:
                feats = extract_email_features_for_model(email_input, url_model_local=url_model)
                df_feats = pd.DataFrame([feats])
                # Reindex:
                if hasattr(email_model, "feature_names_in_"):
                    df_feats = df_feats.reindex(columns=list(email_model.feature_names_in_), fill_value=0)
                else:
                    df_feats = df_feats.reindex(columns=EMAIL_FEATURES, fill_value=0)

                try:
                    proba = float(email_model.predict_proba(df_feats)[:,1][0])
                except Exception:
                    proba = float(email_model.predict(df_feats)[0])

                pred = int(proba > threshold_email)
                final_pred = pred
                notes = ""
                if use_rule:
                    if int(feats.get('benign_email_indicators',0)) >= 2 and float(feats.get('phishing_email_score',0)) < 3:
                        final_pred = 0
                        notes = "Regla aplicada: indicadores benignos en email"
                st.markdown(f"**Probabilidad phishing:** {proba:.2%}")
                st.markdown(f"**Predicción:** {'🔴 Phishing' if final_pred==1 else '🟢 Benigno'}")
                if notes:
                    st.info(notes)
                # --- Guardar en historial ---
                new_row = {
                    'id': len(st.session_state['history']) + 1,
                    'type': 'email',
                    'raw': email_input,
                    'timestamp': pd.Timestamp.now(),
                    'prob_phishing': proba,
                    'pred': pred,
                    'final_pred': final_pred,
                    'notes': notes,
                    'label': np.nan
                }
                st.session_state['history'] = pd.concat(
                    [st.session_state['history'], pd.DataFrame([new_row])],
                    ignore_index=True
                )


# ----------------------
# TAB 2: Carga Masiva
# ----------------------
with tab2:
    st.header("Carga Masiva (CSV / XLSX)")
    st.markdown("Formato recomendado: columna `text`/`content`/`url`, opcional `type` ('url'/'email'), `timestamp` y `label` (0/1).")
    uploaded = st.file_uploader("Sube archivo CSV o XLSX", type=['csv','xlsx'])
    if uploaded is not None:
        try:
            if uploaded.name.endswith('.csv'):
                df_in = pd.read_csv(uploaded)
            else:
                df_in = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Error leyendo archivo: {e}")
            df_in = None

        if df_in is not None:
            # preview
            st.write("Preview:")
            st.dataframe(df_in.head())

            # normalize cols
            df_in.columns = [c.lower() for c in df_in.columns]

            text_col = None
            for c in ['text','content','email','message','url','body']:
                if c in df_in.columns:
                    text_col = c
                    break
            if text_col is None:
                st.error("No se detectó columna de texto/url. Renombra tu columna a 'text' o 'url'.")
            else:
                # inferir type si no existe
                if 'type' not in df_in.columns:
                    def infer_type(x):
                        s = str(x).lower()
                        return 'url' if s.startswith('http') or 'www.' in s else 'email'
                    df_in['type'] = df_in[text_col].apply(infer_type)
                # timestamp
                if 'timestamp' not in df_in.columns:
                    df_in['timestamp'] = pd.Timestamp.now()
                if 'id' not in df_in.columns:
                    df_in['id'] = range(len(df_in))

                results = []
                for _, row in df_in.iterrows():
                    raw = row[text_col]
                    typ = row['type']
                    ts = row['timestamp']
                    try:
                        ts = pd.to_datetime(ts)
                    except:
                        ts = pd.Timestamp.now()

                    prob = np.nan
                    pred = np.nan
                    final_pred = np.nan
                    notes = ""

                    if typ == 'url':
                        feats = extract_features_for_model(raw)
                        df_feats = pd.DataFrame([feats])
                        # reindex to model features
                        if url_model is not None and hasattr(url_model, "feature_names_in_"):
                            df_feats = df_feats.reindex(columns=list(url_model.feature_names_in_), fill_value=0)
                        else:
                            df_feats = df_feats.reindex(columns=URL_FEATURES, fill_value=0)
                        if url_model is not None:
                            try:
                                prob = float(url_model.predict_proba(df_feats)[:,1][0])
                            except Exception:
                                prob = float(url_model.predict(df_feats)[0])
                            pred = int(prob > threshold)
                            final_pred = pred
                            if use_rule and int(feats.get('benign_indicators',0)) >= 3 and int(feats.get('fake_domain_score',0)) == 0:
                                final_pred = 0
                                notes = "Regla aplicada: benign indicators"
                        else:
                            notes = "Modelo URL no cargado"
                    else:
                        feats = extract_email_features_for_model(raw, url_model_local=url_model)
                        df_feats = pd.DataFrame([feats])
                        if email_model is not None and hasattr(email_model, "feature_names_in_"):
                            df_feats = df_feats.reindex(columns=list(email_model.feature_names_in_), fill_value=0)
                        else:
                            df_feats = df_feats.reindex(columns=EMAIL_FEATURES, fill_value=0)
                        if email_model is not None:
                            try:
                                prob = float(email_model.predict_proba(df_feats)[:,1][0])
                            except Exception:
                                prob = float(email_model.predict(df_feats)[0])
                            pred = int(prob > threshold_email)
                            final_pred = pred
                            if use_rule and int(feats.get('benign_email_indicators',0)) >= 2 and float(feats.get('phishing_email_score',0)) < 3:
                                final_pred = 0
                                notes = "Regla aplicada: benign email indicators"
                        else:
                            notes = "Modelo Email no cargado"

                    results.append({
                        'id': row['id'],
                        'type': typ,
                        'raw': raw,
                        'timestamp': ts,
                        'prob_phishing': prob,
                        'pred': pred,
                        'final_pred': final_pred,
                        'notes': notes,
                        'label': row['label'] if 'label' in row and not pd.isna(row['label']) else np.nan
                    })

                df_results = pd.DataFrame(results)
                # append to history
                st.session_state['history'] = pd.concat([st.session_state['history'], df_results], ignore_index=True)
                st.success(f"Procesadas {len(df_results)} filas. Añadidas al historial.")
                st.dataframe(df_results.head())

                st.download_button(
                    "Descargar resultados procesados (CSV)",
                    df_results.to_csv(index=False).encode('utf-8'),
                    file_name="results_processed.csv",
                    mime="text/csv"
                )

# ----------------------
# TAB 3: Estadísticas y Reportes
# ----------------------
with tab3:
    st.header("Estadísticas y Reportes")
    hist = st.session_state['history']
    if hist.empty:
        st.info("No hay historial. Procesa alguna predicción individual o sube un archivo.")
    else:
        # ensure datetime
        hist['timestamp'] = pd.to_datetime(hist['timestamp'])
        min_date = hist['timestamp'].min().date()
        max_date = hist['timestamp'].max().date()
        st.markdown(f"Historial: {len(hist)} registros  —  rango: {min_date} a {max_date}")

        date_range = st.date_input("Selecciona rango de fechas", [min_date, max_date])
        start_dt = pd.to_datetime(date_range[0])
        end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        mask = (hist['timestamp'] >= start_dt) & (hist['timestamp'] <= end_dt)
        filtered = hist.loc[mask].copy()
        st.write(f"Registros en rango: {len(filtered)}")
        st.dataframe(filtered.head(200))

        # supervised metrics if labels present
        if 'label' in filtered.columns and filtered['label'].notna().sum() > 0:
            valid = filtered.dropna(subset=['label','final_pred'])
            if len(valid) > 0:
                y_true = valid['label'].astype(int)
                y_pred = valid['final_pred'].astype(int)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                acc = accuracy_score(y_true, y_pred)
                st.subheader("Métricas (labels disponibles)")
                st.metric("Accuracy", f"{acc:.3f}")
                st.metric("Precision", f"{prec:.3f}")
                st.metric("Recall", f"{rec:.3f}")
                st.metric("F1", f"{f1:.3f}")
            else:
                st.info("No hay filas completas con 'label' y 'final_pred' para métricas.")
        else:
            st.info("No se encontraron labels reales en el filtrado. Mostrando estadísticas agregadas.")

        if not filtered.empty:
            filtered['prob_phishing'] = pd.to_numeric(filtered['prob_phishing'], errors='coerce')
            daily = (filtered.set_index('timestamp')
                     .resample('D')
                     .agg({
                         'final_pred': lambda x: np.nan if len(x.dropna())==0 else (x.dropna().astype(float).mean()),
                         'prob_phishing': 'mean',
                         'id': 'count'
                     })
                     .rename(columns={'final_pred':'positives_rate','prob_phishing':'avg_prob','id':'count'})
                    ).fillna(0)
            daily.index = daily.index.date
            st.subheader("Estadísticas diarias")
            st.dataframe(daily)

            st.subheader("Gráficas")
            st.line_chart(daily[['positives_rate','avg_prob']])
            st.bar_chart(daily['count'])

            agg_type = filtered.groupby('type').agg(
                count=('id','count'),
                positives=('final_pred', lambda x: int(x.dropna().astype(float).sum()) if x.notna().sum()>0 else 0),
                avg_prob=('prob_phishing','mean')
            ).reset_index()
            agg_type['positive_rate'] = agg_type['positives'] / agg_type['count']
            st.subheader("Resumen por tipo")
            st.dataframe(agg_type)

            st.download_button(
                "Descargar historial filtrado (CSV)",
                filtered.to_csv(index=False).encode('utf-8'),
                file_name="historial_filtrado.csv",
                mime="text/csv"
            )

st.markdown("---")
st.markdown("Desarrollado para demo/tesis.")
