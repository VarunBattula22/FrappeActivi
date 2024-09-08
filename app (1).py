import pickle
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import joblib
import os

app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

encoders_path = os.path.dirname(os.path.abspath(__file__))

dayencoder = joblib.load(os.path.join(encoders_path, 'DayTimeEncoder'))
wkencoder = joblib.load(os.path.join(encoders_path, 'WeekdayEncoder'))
wkndencoder = joblib.load(os.path.join(encoders_path, 'WkndEncoder'))
hwencoder = joblib.load(os.path.join(encoders_path, 'hwencoder'))
wencoder = joblib.load(os.path.join(encoders_path, 'WeatherEncoder'))
cencoder = joblib.load(os.path.join(encoders_path, 'CostEncoder'))
nncoder = joblib.load(os.path.join(encoders_path, 'NameEncoder'))

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

# Route to predict page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    return render_template("predict.html")

@app.route('/predictionpage', methods=['POST'])
def predictionpage():
    try:
        # Read CSV file for additional data if needed
        df = pd.read_csv(r'C:\Users\Lohith\Desktop\project1\dataset\frappe.csv')
        item = int(request.form['item'])
        daytime = request.form.get('daytime')
        weekday = request.form.get("weekday")
        cost = request.form.get("cost")
        weather = request.form.get("weather")
        sname = int(request.form.get('sname'))

        if not all([daytime, weekday, cost, weather]):
            return "Error: Missing form data.", 400

        if daytime not in dayencoder.classes_:
            return f"Error: 'daytime' contains an unknown label: {daytime}", 400
        if weekday not in wkencoder.classes_:
            return f"Error: 'weekday' contains an unknown label: {weekday}", 400
        if cost not in cencoder.classes_:
            return f"Error: 'cost' contains an unknown label: {cost}", 400
        if weather not in wencoder.classes_:
            return f"Error: 'weather' contains an unknown label: {weather}", 400
        daytime_encoded = dayencoder.transform([daytime])[0]
        weekday_encoded = wkencoder.transform([weekday])[0]
        cost_encoded = cencoder.transform([cost])[0]
        weather_encoded = wencoder.transform([weather])[0]

        iswknd = "weekend" if weekday.lower() in ['sunday', 'saturday'] else "workday"
        iswknd_encoded = wkndencoder.transform([iswknd])[0]
        x_test = [[item, daytime_encoded, weekday_encoded, iswknd_encoded, cost_encoded, weather_encoded,sname]]
        x_test_scaled = scaler.transform(x_test)

        pred = model.predict(x_test_scaled)
        activity_mapping = {0: "Homework", 1: "Unknown", 2: "Work"}
        result = f"The phone activity was most likely for {activity_mapping.get(pred[0], 'Unknown')}"
        return render_template("predictionpage.html", bc_final=result)
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)