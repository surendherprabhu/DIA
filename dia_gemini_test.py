
from google import genai
from PIL import Image
from pathlib import Path

# Initialise Gemini
client = genai.Client()

MODEL = "gemini-3.8-flash"

# Ask for an image
image_path = input("Enter transistor image path: ").strip()
image_path = Path(image_path.strip('"').strip("'"))

if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

image = Image.open(image_path).convert("RGB")

prompt = """
You are DIA, an electronic component visual inspection assistant.

Examine this transistor image.

1. Describe the visible physical condition.
2. Look for bent leads, cut leads, misplaced parts,
   or damage to the transistor casing.
3. Decide whether it appears GOOD or DEFECTIVE.
4. Explain which visible features support your decision.
5. If you cannot confidently identify a defect, say so.

Do not invent defects. Only report what is visibly supported
by the image.
"""

print("\nAnalysing image with Gemini...\n")

response = client.models.generate_content(
    model=MODEL,
    contents=[prompt, image]
)

print("========== DIA / GEMINI ==========")
print(response.text)
print("=================================")
