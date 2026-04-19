from flask import Flask, render_template, request, jsonify
import pandas as pd
import google.generativeai as genai
import os
import base64

app = Flask(__name__)

# Configure Gemini API
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

@app.route('/', methods=['GET', 'POST'])
def home():
    data = pd.read_csv('data.csv')

    # format time
    data['time'] = pd.to_datetime(data['time'], errors='coerce')
    data['time'] = data['time'].dt.strftime('%Y-%m-%d %H:%M')

    search = ""
    min_mag = 0

    if request.method == 'POST':
        search = request.form.get('search', '')
        min_mag = request.form.get('min_mag', 0)

        if min_mag:
            min_mag = float(min_mag)
        else:
            min_mag = 0

        # Search filter
        if search:
            data = data[data['place'].str.contains(search, case=False, na=False)]

        # Magnitude filter
        data = data[data['mag'] >= min_mag]

    # Sort by highest magnitude
    data = data.sort_values(by='mag', ascending=False)

    # Select useful columns
    data = data[['time', 'place', 'mag']].head(30)

    table = data.to_html(classes='table', index=False)

    return render_template('index.html', table=table, search=search, min_mag=min_mag)


@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    """Analyze an earthquake-related image using Google Gemini Vision API."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

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

    prompt = """You are an expert seismologist and geological analyst. Analyze this image carefully.

If this image contains any earthquake-related information (seismograph readings, earthquake reports, news articles, damage photos, scientific charts, USGS data, etc.), extract and provide:

1. **Magnitude**: The earthquake magnitude if visible or inferable (e.g., 7.2, 6.5)
2. **Location/Place**: Where the earthquake occurred
3. **Time/Date**: When the earthquake happened (UTC or local time)
4. **Depth**: Depth in km if available
5. **Additional Info**: Any other relevant seismic data (aftershocks, tsunami warnings, casualty info, damage assessment, etc.)
6. **Image Type**: What type of image this is (seismograph, news report, damage photo, map, chart, etc.)
7. **Confidence**: Your confidence level in the extracted data (High/Medium/Low)

If this is NOT an earthquake-related image, still analyze it and mention:
- What the image actually shows
- Whether it could be related to geological/seismic activity in any way

Format your response in a clear, structured way using the labels above. Be precise and scientific."""

    try:
        # Use Gemini Vision model
        model = genai.GenerativeModel('gemini-2.0-flash')

        # Prepare image for Gemini
        image_part = {
            "mime_type": media_type,
            "data": base64.b64encode(image_data).decode('utf-8')
        }

        response = model.generate_content([
            prompt,
            {"inline_data": image_part}
        ])

        analysis_text = response.text

        result = {
            'success': True,
            'raw_analysis': analysis_text,
            'image_name': image_file.filename
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)