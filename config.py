##### MODEL #####
MODEL_SAVE_PATH = 'weights/gender_10_2_efficientv2_s1_body.pt'

GENDER = {'male': 0, 'female': 1}

GENDER_REVERT = {0: 'male', 1: 'female'}

N_GENDER = 2

RESIZE = [224, 224]

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

##### Redis ######
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

IMAGE_QUEUE = "image_queue"
BATCH_SIZE = 32
SERVER_SLEEP = 0.25
CLIENT_SLEEP = 0.25