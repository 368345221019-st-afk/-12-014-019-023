import re
import joblib
import streamlit as st

st.set_page_config(page_title="วิเคราะห์ความรู้สึกรีวิวสินค้าแฟชั่น", page_icon="👗")

st.title("👗 ระบบ AI วิเคราะห์ความรู้สึกจากรีวิวสินค้าแฟชั่นออนไลน์")
st.write("กรอกข้อความรีวิวสินค้าด้านล่าง แล้วกดปุ่มวิเคราะห์ผล")


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_model():
    model = joblib.load("sentiment_model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    return model, vectorizer


try:
    model, vectorizer = load_model()
except FileNotFoundError:
    st.error(
        "❌ ไม่พบไฟล์ sentiment_model.joblib หรือ vectorizer.joblib "
        "กรุณารัน train_model.py เพื่อเทรนโมเดลก่อน แล้ววางไฟล์ทั้งสองไว้ในโฟลเดอร์เดียวกับ app.py"
    )
    st.stop()

review_text = st.text_area(
    "ข้อความรีวิวสินค้า",
    placeholder="เช่น This dress is absolutely beautiful and fits perfectly!",
    height=150,
)

if st.button("🔍 วิเคราะห์ผล", type="primary"):
    if not review_text.strip():
        st.warning("กรุณากรอกข้อความรีวิวก่อน")
    else:
        cleaned = clean_text(review_text)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        classes = model.classes_

        emoji_map = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}
        color_map = {"Positive": "green", "Neutral": "orange", "Negative": "red"}

        st.markdown(
            f"### ผลการวิเคราะห์: :{color_map[prediction]}[{prediction} {emoji_map[prediction]}]"
        )

        st.write("ความน่าจะเป็นของแต่ละคลาส:")
        proba_dict = {cls: float(p) for cls, p in zip(classes, proba)}
        st.bar_chart(proba_dict)

st.divider()
st.caption(
    "โมเดลเทรนจาก dataset: Women's E-Commerce Clothing Reviews (Kaggle) "
    "โดยจัดกลุ่มความรู้สึกจากคะแนน Rating: 1-2 ดาว = Negative, 3 ดาว = Neutral, 4-5 ดาว = Positive"
)
