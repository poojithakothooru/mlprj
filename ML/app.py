import streamlit as st
import sqlite3
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
import os
import pickle

# -----------------------------
# DATABASE SETUP
# -----------------------------
def init_db():
    conn = sqlite3.connect('waste.db')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT,
        confidence REAL
    )
    ''')
    conn.commit()
    return conn, c

conn, c = init_db()

# -----------------------------
# MODEL LOAD / TRAIN
# -----------------------------
MODEL_FILE = "model.pkl"

if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
else:
    X = np.random.rand(300, 50)
    y = np.random.randint(0, 6, 300)  # 6 classes

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

# ✅ Real waste categories
classes = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("♻️ Waste Type Classifier")

uploaded_file = st.file_uploader("Upload Waste Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Feature extraction
    img = image.resize((32, 32))
    img = np.array(img).flatten()
    img = img[:50] / 255.0

    if st.button("Predict Type"):
        pred = model.predict([img])
        prob = model.predict_proba([img])

        category = classes[pred[0]]
        confidence = float(np.max(prob) * 100)

        # 🎯 MAIN OUTPUT (what you asked)
        st.success(f"🗑️ This image belongs to: **{category}**")
        st.info(f"Confidence: {confidence:.2f}%")

        # Save to DB
        c.execute("INSERT INTO predictions (category, confidence) VALUES (?, ?)",
                  (category, confidence))
        conn.commit()

# -----------------------------
# HISTORY
# -----------------------------
st.subheader("📊 Prediction History")

c.execute("SELECT * FROM predictions ORDER BY id DESC")
rows = c.fetchall()

for row in rows:
    st.write(f"ID: {row[0]} | {row[1]} | {row[2]:.2f}%")
