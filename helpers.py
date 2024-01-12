import io
import numpy as np
import base64
from PIL import Image

def base64_encode_image(a):
	return base64.b64encode(a.getvalue()).decode("utf-8")

def base64_decode_image(a):
	image_bytes = base64.b64decode(a)
	img = Image.open(io.BytesIO(image_bytes))
	return img