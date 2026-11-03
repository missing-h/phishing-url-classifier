import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder  # NUEVO: Para codificar labels de emails
from sklearn.utils.class_weight import compute_class_weight  # NUEVO: Para pesos dinámicos
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings('ignore')
import joblib
# NUEVO: Para extraer URLs de emails
from bs4 import BeautifulSoup  # pip install beautifulsoup4

# ==========================================
# LISTA DE DOMINIOS BENIGNOS CONOCIDOS 
# ==========================================
KNOWN_BENIGN_DOMAINS = [
    'microsoft.com', 'google.com', 'amazon.com', 'paypal.com', 'apple.com',
    'wikipedia.org', 'youtube.com', 'facebook.com', 'twitter.com', 'linkedin.com',
    'github.com', 'stackoverflow.com', 'nytimes.com', 'bbc.com', 'cnn.com',
    'netflix.com', 'spotify.com', 'ebay.com', 'yahoo.com', 'bing.com'
]

# ==========================================
# FUNCIÓN DE FEATURES MEJORADA (PARA URLs - INTACTA)
# ==========================================
def extract_features(url):
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        if not domain:
            if '//' in url:
                domain = url.split('//')[1].split('/')[0]
            else:
                domain = url.split('/')[0] if '/' in url else url
        path = parsed.path if parsed.path else ''
        
        # Dominio base (sin www o subdominios para chequeo)
        base_domain = domain.replace('www.', '').split('/')[0].rstrip('/')
        
        features = {}
        features['url_length'] = len(url)
        features['domain_length'] = len(domain)
        features['num_dots'] = url.count('.')
        features['num_hyphens_domain'] = domain.count('-')
        features['num_at'] = url.count('@')
        features['num_qm'] = url.count('?')
        features['num_eq'] = url.count('=')
        features['num_slash'] = url.count('/') - (2 if '://' in url else 0)
        features['has_https'] = 1 if url.startswith("https") else 0
        features['uses_http'] = 1 if url.startswith("http://") else 0
        features['has_www'] = 1 if 'www.' in domain else 0
        features['has_ip'] = 1 if re.search(r'(\d{1,3}\.){3}\d{1,3}', url) else 0
        features['num_subdomains'] = max(0, domain.count('.') - 1)
        suspicious_domain = len(re.findall(r"(login|secure|update|verify|bank|paypal)", domain))
        features['suspicious_words_domain'] = suspicious_domain
        suspicious_path = len(re.findall(r"(control|panel|admin|dashboard|cpanel|signin|logon)", path.lower()))
        features['suspicious_path_words'] = suspicious_path
        features['suspicious_path'] = 1 if re.search(r"(php|asp|jsp)\?id=", path) or suspicious_path > 0 else 0
        features['has_port'] = 1 if re.search(r':\d{2,5}(?![/])', url) and not re.search(r':443|:80', url) else 0
        features['is_shortened'] = 1 if any(short in url for short in ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co']) else 0
        features['fake_domain_score'] = features['num_hyphens_domain'] + features['suspicious_words_domain']
        
        # NUEVA: Chequea dominio conocido benigno
        features['known_benign_domain'] = 1 if base_domain in KNOWN_BENIGN_DOMAINS else 0
        
        # benign_indicators mejorado: + known_benign_domain
        features['benign_indicators'] = (
            features['has_www'] + 
            (1 if features['domain_length'] < 15 else 0) + 
            (1 if features['num_hyphens_domain'] == 0 else 0) - 
            features['uses_http'] + 
            features['known_benign_domain']
        )
        return features
    except:
        return {k: 0 for k in ['url_length', 'domain_length', 'num_dots', 'num_hyphens_domain', 'num_at', 'num_qm', 'num_eq', 'num_slash', 
                               'has_https', 'uses_http', 'has_www', 'has_ip', 'num_subdomains', 'suspicious_words_domain', 
                               'suspicious_path_words', 'suspicious_path', 'has_port', 'is_shortened', 'fake_domain_score', 
                               'known_benign_domain', 'benign_indicators']}

# ==========================================
# NUEVO: FUNCIÓN PARA EXTRAER FEATURES DE EMAILS
# ==========================================
def extract_email_features(email_text, url_model=None):
    try:
        # Limpieza básica
        email_text = str(email_text).lower().strip()
        
        # Extraer URLs del email (patrón simple para http/https)
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', email_text)
        
        features = {}
        features['email_length'] = len(email_text)
        features['num_urls'] = len(urls)
        features['num_exclamation'] = email_text.count('!')
        features['num_question'] = email_text.count('?')
        features['has_uppercase_excess'] = 1 if sum(1 for c in email_text if c.isupper()) / len(email_text) > 0.2 else 0
        
        # Palabras sospechosas comunes en phishing emails
        suspicious_words = len(re.findall(r"(urgente|verificar|cuenta suspendida|clic aquí|actualizar|login|secure|bank|paypal|password|confidencial|urgent|verify|account suspended|click here|update|login|secure|bank|paypal|password|confidential)", email_text))
        features['suspicious_words'] = suspicious_words
        
        # Indicadores benignos (ej: menciona remitentes legítimos)
        benign_mentions = len(re.findall(r"(microsoft|google|amazon|paypal oficial|thanks for using|gracias por usar|Si tú no iniciaste esta sesión|If you DID NOT start this session)", email_text))
        features['benign_mentions'] = benign_mentions
        
        # Score de enlaces sospechosos: Usa el modelo de URLs existente si se proporciona
        suspicious_links_score = 0
        if url_model and urls:
            url_features_list = [extract_features(url) for url in urls[:5]]
            url_features_df = pd.DataFrame(url_features_list)
            link_probs = url_model.predict_proba(url_features_df)[:, 1]
            suspicious_links_score = np.mean(link_probs) if len(link_probs) > 0 else 0
        features['suspicious_links_score'] = suspicious_links_score * features['num_urls']
        
        # Score general de phishing en email
        features['phishing_email_score'] = features['suspicious_words'] + features['num_exclamation'] + (features['suspicious_links_score'] * 10)
        features['benign_email_indicators'] = features['benign_mentions'] + (1 if 'gracias por usar' in email_text else 0)
        
        # NUEVO: Feature para detectar emails benignos con códigos de verificación
        benign_verification_phrases = len(re.findall(r"(tu código de verificación|código de verificación|verification code|your verification code|no compartas este código|do not share this code|no lo des a nadie|keep this code secure)", email_text))
        features['benign_verification_code'] = benign_verification_phrases
        
        # Actualiza benign_email_indicators para incluir este feature
        features['benign_email_indicators'] += benign_verification_phrases
        return features
    except:
        # CORRECCIÓN: Agrega 'benign_verification_code' al fallback
        return {k: 0 for k in ['email_length', 'num_urls', 'num_exclamation', 'num_question', 'has_uppercase_excess', 
                               'suspicious_words', 'benign_mentions', 'suspicious_links_score', 'phishing_email_score', 
                               'benign_email_indicators', 'benign_verification_code']}

# ==========================================
# ENTRENAMIENTO PARA URLs (INTACTO)
# ==========================================
df_urls = pd.read_csv("dataset_unido.csv")
print("Ejemplos de datos URLs (primeras 5):\n", df_urls.head())

df_urls['URLs'] = df_urls['URLs'].astype(str).str.strip().str.lower()
df_urls['Label'] = df_urls['Label'].apply(lambda x: 1 if str(x).lower() in ["phishing", "malware", "spam"] else 0)

print("\nDistribución original de labels (URLs):")
print(df_urls['Label'].value_counts())

features_temp_urls = df_urls['URLs'].apply(extract_features)
X_temp_urls = pd.DataFrame(list(features_temp_urls))
y_temp_urls = df_urls['Label']

phish_count_urls = sum(y_temp_urls == 1)
benign_count_urls = sum(y_temp_urls == 0)
ratio_urls = phish_count_urls / benign_count_urls if benign_count_urls > 0 else float('inf')

if ratio_urls > 1.5:
    print(f"Desbalance URLs ({ratio_urls:.1f}:1). Aplicando SMOTEENN...")
    try:
        smote_enn = SMOTEENN(random_state=42)
        X_temp_bal_urls, y_temp_bal_urls = smote_enn.fit_resample(X_temp_urls, y_temp_urls)
        print(f"Después (URLs): {pd.Series(y_temp_bal_urls).value_counts()}")
    except Exception as e:
        print(f"Error URLs: {e}. Usando original.")
        X_temp_bal_urls, y_temp_bal_urls = X_temp_urls, y_temp_urls
else:
    X_temp_bal_urls, y_temp_bal_urls = X_temp_urls, y_temp_urls
    print("Ya balanceado (URLs).")

X_urls = X_temp_bal_urls
y_urls = y_temp_bal_urls

print(f"\nForma de X (URLs): {X_urls.shape}")

X_train_urls, X_test_urls, y_train_urls, y_test_urls = train_test_split(X_urls, y_urls, test_size=0.3, random_state=42, stratify=y_urls)

class_weights_urls = {0: 1.5, 1: 1.0}
url_model = RandomForestClassifier(
    n_estimators=200, 
    random_state=42, 
    class_weight=class_weights_urls,
    min_samples_leaf=5,
    max_depth=10,
    max_features='sqrt'
)
url_model.fit(X_train_urls, y_train_urls)

joblib.dump(url_model, 'phishing_model.pkl')

cv_scores_urls = cross_val_score(url_model, X_train_urls, y_train_urls, cv=5, scoring='f1')
print(f"\nCV F1-score (URLs): {cv_scores_urls.mean():.3f} (+/- {cv_scores_urls.std()*2:.3f})")

y_pred_urls = url_model.predict(X_test_urls)
print("\nClassification Report (URLs):\n")
print(classification_report(y_test_urls, y_pred_urls, target_names=["Benign", "Phishing"]))

importances_urls = pd.Series(url_model.feature_importances_, index=X_urls.columns).sort_values(ascending=False)
print("\nTop 10 Importancias (URLs):\n")
print(importances_urls.head(10))

# ==========================================
# ENTRENAMIENTO PARA EMAILS
# ==========================================
email_model = None

try:
    df_emails = pd.read_csv("Phishing_Email.csv")
    print("\nEjemplos de datos Emails (primeras 5):\n", df_emails.head())
    
    required_cols = ['Email Text', 'Email Type']
    if not all(col in df_emails.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_emails.columns]
        raise KeyError(f"Columnas faltantes en CSV: {missing}")
    
    df_emails['Email Text'] = df_emails['Email Text'].astype(str).str.strip().str.lower()
    df_emails['Email Type'] = df_emails['Email Type'].astype(str).str.strip().str.lower()
    
    df_emails['Email Type'] = df_emails['Email Type'].replace({
        "phishing email": "phishing",
        "safe email": "safe",
        "phishing": "phishing",
        "safe": "safe"
    })
    
    print("\nDistribución original de labels (Emails):")
    print(df_emails['Email Type'].value_counts())
    
    le = LabelEncoder()
    y_temp_emails = le.fit_transform(df_emails['Email Type'])
    print(f"Labels codificados: {le.classes_} → {np.unique(y_temp_emails)}")
    
    print("\nExtrayendo features de emails...")
    features_temp_emails = df_emails['Email Text'].apply(lambda x: extract_email_features(x, url_model))
    # CORRECCIÓN: No especificar columnas manualmente, deja que Pandas las infiera
    X_temp_emails = pd.DataFrame(list(features_temp_emails))
    X_temp_emails = X_temp_emails.fillna(0)
    
    print(f"Forma inicial de X (Emails): {X_temp_emails.shape}")
    
    phish_count_emails = np.sum(y_temp_emails == 1)
    benign_count_emails = np.sum(y_temp_emails == 0)
    ratio_emails = phish_count_emails / benign_count_emails if benign_count_emails > 0 else float('inf')
    
    print(f"Conteo: Safe={benign_count_emails}, Phishing={phish_count_emails}, Ratio={ratio_emails:.2f}:1")
    
    if ratio_emails > 1.5 or ratio_emails < 0.5:
        print(f"Desbalance detectado ({ratio_emails:.1f}:1). Aplicando SMOTEENN...")
        try:
            smote_enn = SMOTEENN(random_state=42)
            X_temp_bal_emails, y_temp_bal_emails = smote_enn.fit_resample(X_temp_emails, y_temp_emails)
            print(f"Después del balanceo (Emails): {pd.Series(y_temp_bal_emails).value_counts().sort_index()}")
        except Exception as e:
            print(f"Error en SMOTEENN (Emails): {e}. Usando datos originales.")
            X_temp_bal_emails, y_temp_bal_emails = X_temp_emails, y_temp_emails
    else:
        X_temp_bal_emails, y_temp_bal_emails = X_temp_emails, y_temp_emails
        print("Dataset ya balanceado (Emails).")
    
    X_emails = X_temp_bal_emails
    y_emails = y_temp_bal_emails
    
    print(f"\nForma final de X (Emails): {X_emails.shape}")
    print(f"Forma de y (Emails): {y_emails.shape}")
    
    X_train_emails, X_test_emails, y_train_emails, y_test_emails = train_test_split(
        X_emails, y_emails, test_size=0.3, random_state=42, stratify=y_emails
    )
    
    classes = np.unique(y_train_emails)
    class_weights_emails = dict(enumerate(compute_class_weight(
        'balanced', classes=classes, y=y_train_emails
    )))
    print(f"Pesos de clases calculados: {class_weights_emails}")
    
    email_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight=class_weights_emails,
        min_samples_leaf=5,
        max_depth=10,
        max_features='sqrt'
    )
    email_model.fit(X_train_emails, y_train_emails)
    
    joblib.dump(email_model, 'phishing_email_model.pkl')
    print("\nModelo guardado como 'phishing_email_model.pkl'")
    
    cv_scores_emails = cross_val_score(email_model, X_train_emails, y_train_emails, cv=5, scoring='f1')
    print(f"\nCV F1-score (Emails): {cv_scores_emails.mean():.3f} (+/- {cv_scores_emails.std()*2:.3f})")
    
    y_pred_emails = email_model.predict(X_test_emails)
    print("\nClassification Report (Emails):\n")
    print(classification_report(y_test_emails, y_pred_emails, target_names=le.classes_, digits=3))
    
    importances_emails = pd.Series(email_model.feature_importances_, index=X_emails.columns).sort_values(ascending=False)
    print("\nTop 10 Importancias de Features (Emails):\n")
    print(importances_emails.head(10))
    
    print("\nEntrenamiento completado exitosamente para modelo de emails.")

except FileNotFoundError:
    print("\nNo se encontró 'Phishing_Email.csv'. Se omite el entrenamiento del modelo para emails.")
    email_model = None
except KeyError as e:
    print(f"\nError: Columna faltante en el CSV: {e}")
    email_model = None
except ValueError as e:
    print(f"\nError de validación: {e}")
    email_model = None
except Exception as e:
    print(f"\nError general durante el entrenamiento: {e}")
    email_model = None

# ==========================================
# PREDICCIONES DE PRUEBA PARA URLs (INTACTO)
# ==========================================
test_urls = [
    "https://paypal-login-secure.com",
    "https://www.microsoft.com",
    "http://www.cwa4501.org/ControlPanel",
    "http://www.portal.newagemusics.com/nahp/artic-es/wiki_K"
]

test_features_list = [extract_features(url) for url in test_urls]
test_features = pd.DataFrame(test_features_list)
y_pred_proba = url_model.predict_proba(test_features)

threshold = 0.2
predictions_adjusted = (y_pred_proba[:, 1] > threshold).astype(int)

print("\n=== PREDICCIONES MEJORADAS PARA URLs (Threshold 0.2 + Regla Benign) ===")
for i, (url, features, proba, pred) in enumerate(zip(test_urls, test_features_list, y_pred_proba[:, 1], predictions_adjusted)):
    if features['benign_indicators'] >= 3 and features['fake_domain_score'] == 0:
        final_pred = 0
        final_label = 'Benign (Forzado por regla)'
    else:
        final_pred = pred
        final_label = 'Phishing' if pred == 1 else 'Benign'
    
    key_feats = {
        'uses_http': features['uses_http'],
        'suspicious_path_words': features['suspicious_path_words'],
        'benign_indicators': features['benign_indicators'],
        'fake_domain_score': features['fake_domain_score'],
        'suspicious_words_domain': features['suspicious_words_domain'],
        'known_benign_domain': features['known_benign_domain']
    }
    print(f"{url} -> {final_label} (Prob phishing: {proba:.2f})")
    print(f"  Features clave: {key_feats}")
    print()

# ==========================================
# PREDICCIONES DE PRUEBA PARA EMAILS
# ==========================================
if email_model is not None:
    test_emails = [
        "Estimado usuario, su cuenta ha sido suspendida. Por favor, verifique su información haciendo clic aquí: http://secure-paypal-login.com",
        "Hola, gracias por usar nuestro servicio. Su factura está disponible en su cuenta.",
        "URGENTE: Actualice su contraseña ahora para evitar bloqueo de cuenta. Visite https://bank-secure-login.com",
        "Reunión programada para mañana a las 10am. Saludos cordiales.",
        "Tu código de verificación es 123456. No lo des a nadie."  # Ejemplo benigno con código
    ]

    test_email_features_list = [extract_email_features(email, url_model) for email in test_emails]
    test_email_features = pd.DataFrame(test_email_features_list)
    y_pred_proba_emails = email_model.predict_proba(test_email_features)

    threshold_email = 0.3
    predictions_adjusted_emails = (y_pred_proba_emails[:, 1] > threshold_email).astype(int)

    print("\n=== PREDICCIONES MEJORADAS PARA EMAILS (Threshold 0.3) ===")
    for i, (email, features, proba, pred) in enumerate(zip(test_emails, test_email_features_list, y_pred_proba_emails[:, 1], predictions_adjusted_emails)):
        if features['benign_email_indicators'] >= 2 and features['phishing_email_score'] < 3:
            final_pred = 0
            final_label = 'Safe (Forzado por regla)'
        else:
            final_pred = pred
            final_label = 'Phishing' if pred == 1 else 'Safe'
        
        key_feats = {
            'suspicious_words': features['suspicious_words'],
            'num_exclamation': features['num_exclamation'],
            'suspicious_links_score': features['suspicious_links_score'],
            'benign_email_indicators': features['benign_email_indicators'],
            'phishing_email_score': features['phishing_email_score'],
            'benign_verification_code': features['benign_verification_code']  # Nuevo feature
        }
        print(f"Email {i+1}: {email[:100]}... -> {final_label} (Prob phishing: {proba:.2f})")
        print(f"  Features clave: {key_feats}")
        print()
else:
    print("\nNo hay modelo de emails entrenado para hacer predicciones.")

print("\n=== SCRIPT COMPLETADO ===")
