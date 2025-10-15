import cv2
import numpy as np
from skimage import color, filters
import requests
import json
import time


# ---------- STEP 1: Capture image from webcam ----------
def capture_image():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise Exception("❌ Cannot access camera. Make sure it's connected and not used by another app.")

    print("📸 Press SPACE to capture, ESC to exit.")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break
        cv2.imshow("Skin Analyzer - Press SPACE to Capture", frame)
        key = cv2.waitKey(1)
        if key % 256 == 27:  # ESC
            print("❌ Capture cancelled.")
            cam.release()
            cv2.destroyAllWindows()
            return None
        elif key % 256 == 32:  # SPACE
            img_name = "captured_face.jpg"
            cv2.imwrite(img_name, frame)
            print(f"✅ Image saved as {img_name}")
            cam.release()
            cv2.destroyAllWindows()
            return img_name


# ---------- STEP 2: Analyze skin metrics ----------
def analyze_skin(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise Exception("❌ Image not found.")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    crop = rgb[h//4:3*h//4, w//4:3*w//4]

    lab = color.rgb2lab(crop)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    brightness = np.mean(L) / 100.0
    redness = (np.mean(a) + 128) / 255.0
    yellowness = (np.mean(b) + 128) / 255.0
    texture = np.std(filters.sobel(color.rgb2gray(crop)))

    redness = np.clip(redness, 0, 1)
    brightness = np.clip(brightness, 0, 1)
    texture = np.clip(texture, 0, 1)
    yellowness = np.clip(yellowness, 0, 1)

    return {
        "brightness": round(brightness, 3),
        "redness": round(redness, 3),
        "texture": round(texture, 3),
        "yellowness": round(yellowness, 3)
    }


# ---------- STEP 3: Send metrics to Ollama ----------
def get_recommendations_from_mistral(skin_data):
    prompt = f"""
    You are a friendly AI skincare assistant.
    Based on these skin analysis metrics, provide short, personalized skincare advice.

    - Brightness: {skin_data['brightness']}
    - Redness: {skin_data['redness']}
    - Texture (smoothness): {skin_data['texture']}
    - Yellowness: {skin_data['yellowness']}

    Explain what these results mean in human terms and suggest 3 simple skincare tips.
    Avoid medical terminology — keep it clear and encouraging.
    """

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llava:7b-v1.6-mistral-q2_K", "prompt": prompt, "stream": False},  # ✅ Use mistral instead of LLaVA
            timeout=120
        )
        text = response.text.strip()

        # Ollama sometimes returns multiple JSON objects — handle that
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract the last valid JSON block
            lines = [l for l in text.splitlines() if l.strip().startswith("{")]
            result = json.loads(lines[-1]) if lines else {"response": text}

        return result.get("response", "❌ No response received from Ollama.")

    except Exception as e:
        return f"❌ Error connecting to Ollama: {e}"


# ---------- MAIN ----------
if __name__ == "__main__":
    print("🚀 Starting Skin Analyzer...")
    image_path = capture_image()
    if image_path:
        print("\n🔍 Analyzing skin characteristics...")
        skin_metrics = analyze_skin(image_path)
        print("📊 Metrics:", skin_metrics)

        print("\n💬 Generating recommendations ...")
        advice = get_recommendations_from_mistral(skin_metrics)
        print("\n🌿 Skincare Recommendations:\n")
        print(advice)
