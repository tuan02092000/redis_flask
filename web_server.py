import io
import flask
import redis
import uuid
import time
import json
from PIL import Image

import config
import helpers

# Flask app
app = flask.Flask(__name__)

# Redis DB
db = redis.StrictRedis(host=config.REDIS_HOST, port=config.REDIS_PORT, db=config.REDIS_DB)

@app.route("/")
def homepage():
	return "Demo Redis with Flask!"

@app.route("/predict", methods=["POST"])
def predict():
	data = {"success": False}

	if flask.request.method == "POST":
		if flask.request.files.get("image"):

			image = flask.request.files["image"].read()

			# image = Image.open(io.BytesIO(image))
			# buffered = io.BytesIO()
			# image.save(buffered, format="JPEG")

			buffered = io.BytesIO(image)

			k = str(uuid.uuid4())
			image = helpers.base64_encode_image(buffered)
			d = {"id": k, "image": image}
			db.rpush(config.IMAGE_QUEUE, json.dumps(d))

			while True:
				output = db.get(k)

				if output is not None:
					output = output.decode("utf-8")
					data["predictions"] = json.loads(output)

					db.delete(k)
					break

				time.sleep(config.CLIENT_SLEEP)

			data["success"] = True

	return flask.jsonify(data)

if __name__ == "__main__":
	print("* Starting web service...")
	app.run()
