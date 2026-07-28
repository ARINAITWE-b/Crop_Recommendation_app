from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import joblib
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from flask import send_file
from datetime import datetime
import io
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER


app = Flask(__name__)


# Load trained model
model = tf.keras.models.load_model(
    "models/model.keras"
)


# Load preprocessing tools
scaler = joblib.load(
    "models/scaler.pkl"
)

label_encoder = joblib.load(
    "models/label_encoder.pkl"
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])

        temperature = float(request.form["temperature"])

        humidity = float(request.form["humidity"])

        ph = float(request.form["ph"])

        rainfall = float(request.form["rainfall"])


        # Arrange inputs exactly like training data

        input_data = np.array([[
            N,
            P,
            K,
            temperature,
            humidity,
            ph,
            rainfall
        ]])


        # Scaling
        scaled_data = scaler.transform(input_data)


        # Prediction
        prediction = model.predict(
            scaled_data
        )


        predicted_id = np.argmax(prediction)


        crop = label_encoder.inverse_transform(
            [predicted_id]
        )[0]


        confidence = np.max(prediction)*100


        return render_template(
            "result.html",
            crop=crop,
            confidence=round(confidence,2), 

                N=N,
    P=P,
    K=K,
    temperature=temperature,
    humidity=humidity,
    ph=ph,
    rainfall=rainfall
        )


    except Exception as e:

        return str(e)




@app.route("/download_pdf")
def download_pdf():

    crop = request.args.get("crop")
    confidence = request.args.get("confidence")

    N = request.args.get("N")
    P = request.args.get("P")
    K = request.args.get("K")

    temperature = request.args.get("temperature")
    humidity = request.args.get("humidity")
    ph = request.args.get("ph")
    rainfall = request.args.get("rainfall")


    # Create PDF in memory

    pdf = io.BytesIO()


    document = SimpleDocTemplate(
        pdf,
        pagesize=letter
    )


    styles = getSampleStyleSheet()

    content = []


    # Title

    title = Paragraph(
        "<b>AI Crop Recommendation Report</b>",
        styles["Title"]
    )

    content.append(title)

    content.append(Spacer(1,20))


    date = Paragraph(
        f"Date Generated : {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        styles["Normal"]
    )

    content.append(date)

    content.append(Spacer(1,20))


    # Farm Conditions Heading

    heading = Paragraph(
        "<b>Farm Input Conditions</b>",
        styles["Heading2"]
    )

    content.append(heading)


    content.append(Spacer(1,10))


    # Create table data

    farm_data = [

        ["Parameter", "Value"],

        ["Nitrogen (N)", N],

        ["Phosphorus (P)", P],

        ["Potassium (K)", K],

        ["Temperature", f"{temperature} °C"],

        ["Humidity", f"{humidity} %"],

        ["Soil pH", ph],

        ["Rainfall", f"{rainfall} mm"]

    ]


    table = Table(
        farm_data,
        colWidths=[200,150]
    )


    table.setStyle(

        TableStyle([

            ('GRID',(0,0),(-1,-1),1,None),

            ('BACKGROUND',(0,0),(-1,0),None),

            ('ALIGN',(0,0),(-1,-1),'CENTER'),

            ('FONT',(0,0),(-1,0),'Helvetica-Bold'),

            ('BOTTOMPADDING',(0,0),(-1,0),12)

        ])

    )


    content.append(table)


    content.append(Spacer(1,30))


    # Recommendation section

    recommendation = Paragraph(

        f"""
        <b>AI Recommendation</b><br/><br/>

        Recommended Crop:
        <b>{crop}</b><br/><br/>

        Model Confidence:
        <b>{confidence}%</b>

        """,

        styles["Normal"]

    )


    content.append(recommendation)



    # Build PDF

    document.build(content)


    pdf.seek(0)


    return send_file(

        pdf,

        as_attachment=True,

        download_name="Crop_Recommendation_Report.pdf",

        mimetype="application/pdf"

    )


if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
        
    