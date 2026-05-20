
from flask import Flask, render_template, request, jsonify
import pandas as pd
from google import genai
import os
import base64
import numpy as np
import joblib
import numpy as np
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Configure Gemini API
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))



# =========================
# LOAD DATA + TRAIN MODEL
# =========================

data = pd.read_csv('data.csv')

# Clean data
data = data.dropna(subset=['mag', 'depth', 'latitude', 'longitude'])

# Create risk labels
def risk_level(mag):
    if mag < 4:
        return 0
    elif mag < 6:
        return 1
    else:
        return 2

data['risk'] = data['mag'].apply(risk_level)

# Features + Target
X = data[['mag', 'depth', 'latitude', 'longitude']]
y = data['risk']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)
# Save trained model
joblib.dump(model, 'earthquake_model.pkl')
# Load model
model = joblib.load('earthquake_model.pkl')


# =========================
# HOME PAGE
# =========================
@app.route('/', methods=['GET', 'POST'])
def home():

    search = request.form.get('search', '').strip().lower()
    min_mag = request.form.get('min_mag', 0)

    try:
        min_mag = float(min_mag)
    except:
        min_mag = 0

    filtered_data = data.copy()

    # Convert time
    filtered_data['time'] = pd.to_datetime(
        filtered_data['time'],
        errors='coerce'
    ).dt.strftime('%Y-%m-%d %H:%M')

    # ----------------------------
    # FIX 1: safer search logic
    # ----------------------------
    if search:
        filtered_data = filtered_data[
            filtered_data['place'].str.lower().str.contains(search, na=False)
        ]

    # ----------------------------
    # FIX 2: magnitude filter
    # ----------------------------
    filtered_data = filtered_data[filtered_data['mag'] >= min_mag]

    # ----------------------------
    # FIX 3: avoid repeated same top results
    # ----------------------------
    if len(filtered_data) > 0:
        filtered_data = filtered_data.sample(
            n=min(30, len(filtered_data)),
            random_state=None
        )

    # Select columns safely
    if filtered_data.empty:
        table = "<p>No results found</p>"
    else:
        table = filtered_data[['time', 'place', 'mag']].to_html(
            classes='table',
            index=False
        )

    return render_template(
        'index.html',
        table=table,
        search=search,
        min_mag=min_mag
    )


# =========================
# ADVANCED EARTHQUAKE PREDICTION
# =========================

@app.route('/predict', methods=['POST'])
def predict():

    try:

        place = request.form.get('place', '').strip().lower()

        if not place:
            return jsonify({
                "error": "Location is required"
            }), 400

        # Clean place names
        temp_data = data.copy()

        temp_data['place'] = (
            temp_data['place']
            .astype(str)
            .str.lower()
            .str.strip()
        )

        # Filter matching region
        filtered = temp_data[
            temp_data['place'].str.contains(place, na=False)
        ]

        # No matching region
        if filtered.empty:
            return jsonify({
                "risk": "NO DATA",
                "confidence": 0
            })

        # =========================
        # FEATURE ENGINEERING
        # =========================

        avg_mag = filtered['mag'].mean()
        avg_depth = filtered['depth'].mean()

        avg_lat = filtered['latitude'].mean()
        avg_long = filtered['longitude'].mean()

        quake_count = len(filtered)

        max_mag = filtered['mag'].max()

        recent_quakes = len(
            filtered[
                filtered['mag'] >= 5
            ]
        )

        # Seismic activity score
        seismic_score = min(
            round((avg_mag * 15) + (recent_quakes * 2)),
            100
        )

        # Regional stability score
        stability_score = max(
            100 - seismic_score,
            5
        )

        # Magnitude range
        mag_min = round(avg_mag - 0.5, 1)
        mag_max = round(avg_mag + 0.5, 1)

        # Depth range
        depth_min = round(avg_depth - 10, 1)
        depth_max = round(avg_depth + 10, 1)

        # =========================
        # ML MODEL PREDICTION
        # =========================

        features = np.array([[
            avg_mag,
            avg_depth,
            avg_lat,
            avg_long
        ]])

        prediction = model.predict(features)[0]

        probabilities = model.predict_proba(features)[0]


        # =========================
        # REALISTIC CONFIDENCE SCORE
        # =========================

        confidence = (
            (avg_mag * 8) +
            (recent_quakes * 0.4) +
            (quake_count * 0.05)
        )

        # Depth effect
        if avg_depth < 70:
            confidence += 8

        # Strong earthquake effect
        if max_mag >= 7:
            confidence += 10

        # Stable regions reduce confidence
        if stability_score > 70:
            confidence -= 12

        # Clamp range
        confidence = max(35, min(confidence, 95))

        confidence = round(confidence, 2)




        

        # =========================
        # RISK LABELS
        # =========================

        risk_labels = {
            0: "LOW RISK",
            1: "MODERATE RISK",
            2: "HIGH RISK"
        }

        risk = risk_labels.get(
            prediction,
            "UNKNOWN"
        )

        # Very High logic
        if avg_mag >= 7:
            risk = "VERY HIGH RISK"

        # =========================
        # TREND ANALYSIS
        # =========================

        if recent_quakes > 50:
            trend = "Increasing Activity"

        elif recent_quakes > 20:
            trend = "Moderate Activity"

        else:
            trend = "Stable Activity"

        # =========================
        # AI INSIGHTS
        # =========================

        insights = []

        if avg_mag >= 6:
            insights.append(
                "Region shows elevated seismic intensity."
            )

        if avg_depth < 70:
            insights.append(
                "Most earthquakes are shallow depth events."
            )

        if recent_quakes > 30:
            insights.append(
                "Recent seismic activity has increased significantly."
            )

        if stability_score < 40:
            insights.append(
                "Regional tectonic stability appears weak."
            )

        if not insights:
            insights.append(
                "Seismic activity appears relatively stable."
            )

        ai_summary = " ".join(insights)

        # =========================
        # PROBABILITY ANALYSIS
        # =========================

        prob_4 = round(
            (len(filtered[filtered['mag'] >= 4]) / quake_count) * 100,
            2
        )

        prob_5 = round(
            (len(filtered[filtered['mag'] >= 5]) / quake_count) * 100,
            2
        )

        prob_6 = round(
            (len(filtered[filtered['mag'] >= 6]) / quake_count) * 100,
            2
        )

        prob_7 = round(
            (len(filtered[filtered['mag'] >= 7]) / quake_count) * 100,
            2
        )

        # =========================
        # ALERT LEVEL
        # =========================

        if seismic_score > 80:
            alert = "CRITICAL ALERT"

        elif seismic_score > 60:
            alert = "HIGH ALERT"

        elif seismic_score > 40:
            alert = "MODERATE ALERT"

        else:
            alert = "LOW ALERT"

        # =========================
        # FINAL RESPONSE
        # =========================

        return jsonify({

            "location": place.title(),

            "timestamp": datetime.now().strftime(
                "%d %B %Y | %I:%M %p"
            ),

            "risk": risk,

            "confidence": confidence,

            "magnitude_range":
                f"{mag_min} - {mag_max}",

            "depth_range":
                f"{depth_min} km - {depth_max} km",

            "seismic_score": seismic_score,

            "stability_score": stability_score,

            "quake_count": quake_count,

            "max_magnitude": round(max_mag, 2),

            "average_magnitude": round(avg_mag, 2),

            "average_depth": round(avg_depth, 2),

            "recent_earthquakes": recent_quakes,

            "trend": trend,

            "ai_insights": ai_summary,

            "probability_4": prob_4,

            "probability_5": prob_5,

            "probability_6": prob_6,

            "probability_7": prob_7,

            "alert_level": alert

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500



# =========================
# IMAGE ANALYSIS
# =========================

@app.route('/analyze-image', methods=['POST'])
def analyze_image():

    if 'image' not in request.files:
        return jsonify({
            'error': 'No image file provided'
        }), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({
            'error': 'No image selected'
        }), 400

    # Read image bytes
    image_data = image_file.read()

    # Determine media type
    filename = image_file.filename.lower()

    if filename.endswith('.jpg') or filename.endswith('.jpeg'):
        media_type = 'image/jpeg'

    elif filename.endswith('.png'):
        media_type = 'image/png'

    elif filename.endswith('.gif'):
        media_type = 'image/gif'

    elif filename.endswith('.webp'):
        media_type = 'image/webp'

    else:
        media_type = 'image/jpeg'

    prompt = """
You are an expert seismologist and geological analyst.

Analyze this image carefully.

If this image contains earthquake-related information,
extract and provide:

1. Magnitude
2. Location
3. Time/Date
4. Depth
5. Additional seismic info
6. Image type
7. Confidence level

If not earthquake-related,
describe the image normally.
"""

    try:

        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

        image_part = {
            "mime_type": media_type,
            "data": base64.b64encode(
                image_data
            ).decode('utf-8')
        }

        response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[
        prompt,
        {"inline_data": image_part}
    ]
)

        analysis_text = response.text

        result = {
            'success': True,
            'raw_analysis': analysis_text,
            'image_name': image_file.filename
        }

        return jsonify(result)

    except Exception as e:

        return jsonify({
            'error': f'Analysis failed: {str(e)}'
        }), 500


# =========================
# RUN APP
# =========================

if __name__ == '__main__':
    app.run(debug=True)

