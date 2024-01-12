import redis
import time
import json
import torch

import config
import helpers
from test_prob import GenderClass

# Connect to Redis server
db = redis.StrictRedis(host=config.REDIS_HOST, port=config.REDIS_PORT, db=config.REDIS_DB)

def classify_process():
	print("[INFO] Loading model...")
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = GenderClass(device=device)
	print("[INFO] Model loaded")

	while True:
		queue = db.lrange(config.IMAGE_QUEUE, 0, config.BATCH_SIZE - 1)
		imageIDs = []

		for q in queue:
			q = json.loads(q.decode("utf-8"))
			image = helpers.base64_decode_image(q["image"])
			imageIDs.append(q["id"])

		if len(imageIDs) > 0:
			rs = model.predict_image_redis_pil(image)

			output = {"label": rs[0], "probability": float(rs[1])}
			db.set(imageIDs[0], json.dumps(output))

			db.ltrim(config.IMAGE_QUEUE, len(imageIDs), -1)

		time.sleep(config.SERVER_SLEEP)

if __name__ == "__main__":
	classify_process()
