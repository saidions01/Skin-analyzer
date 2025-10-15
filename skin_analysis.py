import cv2
import base64
import subprocess
import json

def capture_image():
    cap = cv2.VideoCapture(0)
    print("📸 Press SPACE to capture, ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot access camera.")
            break

        cv2.imshow("Skin Analyzer", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        elif key == 32:
            img_path = "captured_skin.jpg"
            cv2.imwrite(img_path, frame)
            print(f"✅ Image saved as {img_path}")
            cap.release()
            cv2.destroyAllWindows()
            return img_path

    cap.release()
    cv2.destroyAllWindows()
    return None


def analyze_skin_with_llm(image_path):
    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = (
        "You are a dermatologist assistant. Analyze the provided photo for possible skin issues "
        "(acne, dryness, redness, etc.) and give clear skincare recommendations focusing on hydration, "
        "sun protection, and gentle care tips."
    )

    input_json = json.dumps({
        "model": "llava:7b-v1.6-mistral-q2_K",
        "prompt": prompt,
        "images": [img_base64],
        "stream": True
    })

    print("🧴 Sending image to the AI... please wait ⏳\n")

    process = subprocess.Popen(
        ["ollama", "run", "llava:7b-v1.6-mistral-q2_K"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    # Send JSON input
    process.stdin.write(input_json + "\n")
    process.stdin.flush()
    process.stdin.close()

    print("🧴 Skincare Recommendations:\n")
    # Stream output live
    for line in process.stdout:
        if line.strip():
            print(line.strip())

    process.wait()


if __name__ == "__main__":
    img_path = capture_image()
    if img_path:
        analyze_skin_with_llm(img_path)
