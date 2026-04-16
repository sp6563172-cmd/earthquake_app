from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

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

        # 🔍 Search filter
        if search:
            data = data[data['place'].str.contains(search, case=False, na=False)]

        # 📊 Magnitude filter
        data = data[data['mag'] >= min_mag]

    # 🔽 Sort by highest magnitude
    data = data.sort_values(by='mag', ascending=False)

    # 🎯 Select useful columns
    data = data[['time', 'place', 'mag']].head(30)

    table = data.to_html(classes='table', index=False)

    return render_template('index.html', table=table, search=search, min_mag=min_mag)

if __name__ == '__main__':
    app.run(debug=True)