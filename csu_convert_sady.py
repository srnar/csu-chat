import csv
import json

def short_uroven(data):
    if not(data):
        return ''
    return [item.get('nazevUrovne') for item in data if item.get('nazevUrovne')]

# Read JSON file
with open("./csu-sady.json", mode="r", encoding="utf-8") as file:
    data = json.load(file)

# Export selected fields to CSV
with open("./output.csv", mode="w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow(["kod", "nazev", "urovneTypObdobi", "urovneTypUzemi"])  # CSV Header
    for item in data:
        writer.writerow([item.get("kod"), item.get("nazev"), short_uroven(item.get("urovneTypObdobi")), short_uroven(item.get("urovneTypUzemi"))])