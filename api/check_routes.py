from flask import Flask
from api.app import app  # Importez votre application Flask

print("Routes enregistrées dans l'application Flask:")
for rule in app.url_map.iter_rules():
    print(f"{rule.endpoint}: {rule.rule}")
