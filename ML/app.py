import streamlit as st
import sqlite3
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression

# -----------------------------
# DATABASE SETUP
# -----------------------------
conn = sqlite3.connect('waste.db')
c = conn.cursor()

c.execute('''
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    result TEXT,
    confidence REAL
)
''')
conn.commit()

# -----------------------------
# SIMPLE MODEL (DUMMY TRAINING)
# -----------------------------
# Fake training data (just for demo)
X = np.random.rand(100, 10)
y = np.random.randint(0, 3, 100)

model = LogisticRegression()
model.fit(X, y)

classes = ['plastic','paper','glass','metal','carboard']

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("♻️ Simple Waste Classifier")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to simple features
    img = image.resize((10, 10))
    img = np.array(img).flatten()
    img = img[:10] / 255.0  # reduce size for model

    if st.button("Predict"):
        pred = model.predict([img])
        prob = model.predict_proba([img])

        result = classes[pred[0]]
        confidence = np.max(prob) * 100

        st.success(f"Prediction: {result}")
        st.info(f"Confidence: {confidence:.2f}%")

        # Save to DB
        c.execute("INSERT INTO predictions (result, confidence) VALUES (?, ?)",
                  (result, confidence))
        conn.commit()

# -----------------------------
# SHOW DATABASE HISTORY
# -----------------------------
st.subheader("📊 Prediction History")

c.execute("SELECT * FROM predictions")
rows = c.fetchall()

for row in rows:
    st.write(f"ID: {row[0]}, Result: {row[1]}, Confidence: {row[2]:.2f}%")
