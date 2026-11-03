import joblib

model_url = joblib.load("phishing_model.pkl")
model_email = joblib.load("phishing_email_model.pkl")

print("\n=== URL MODEL FEATURE NAMES ===")
print(model_url.feature_names_in_)

print("\n=== EMAIL MODEL FEATURE NAMES ===")
print(model_email.feature_names_in_)
