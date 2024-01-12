from PIL import Image
import cv2
import torch
import torchvision
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class GenderClass():
    def __init__(self, device='cuda'):
        self.device = device
        self.model = self.get_model()
        self.thresh_hold = 0.72
        self.min_w = 30
        self.min_h = 100

    def get_model(self):
        model = torch.load('/home/nguyen-tuan/Documents/AI/project/API/practice/redis/weights/gender_10_2_efficientv2_s1_body.pt')
        model.eval()
        model.to(self.device)
        return model
    
    def get_transform_album(self, img):
        img_trans = A.Compose([
                A.Resize(224, 224),
                ToTensorV2(),
                A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        img = np.array(img)
        augmentations = img_trans(image=img)
        img = augmentations["image"]

        img = img.unsqueeze(0).to(self.device)
        return img
    
    def get_transform(self, img):
        img_trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),          
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        img = img_trans(img)
        img = img.unsqueeze(0).to(self.device)
        return img
        
    def get_transform_PIL(self, img):
        img_trans = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),          
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        img = img_trans(img)
        img = img.unsqueeze(0).to(self.device)
        return img

    def predict(self, img_path):
        GENDER_REVERT = {0: 'male', 1: 'female'}
        
        img = Image.open(img_path).convert('RGB')

        w, h = img.size
        if (w < self.min_w) or (h < self.min_h):
            return None

        img = self.get_transform_PIL(img)
        gender_output = self.model(img)

        # Gender
        gender_output = torch.softmax(gender_output, dim=1)
        gender_class_prob, gender_topclass = torch.max(gender_output, dim=1)
        gender_prob_predict = gender_class_prob.cpu().item()
        gender_predict = GENDER_REVERT[gender_topclass.cpu().item()]

        # print('[PREDICT] Image: {}, Gender: {} ({:.4f})'.format(img_path, gender_predict, gender_prob_predict))

        print(gender_prob_predict)
        if gender_prob_predict > self.thresh_hold:
            print('[PREDICT] Image: {}, Gender: {} ({:.4f})'.format(img_path, gender_predict, gender_prob_predict))
            return gender_predict
        else:
            return None
    
    def predict_image(self, img):
        # inference cv2 image
        GENDER_REVERT = {0: 'male', 1: 'female'}
        w, h, c = img.shape
        if (w < self.min_w) or (h < self.min_h):
            return None

        img = self.get_transform(img)
        gender_output = self.model(img)

        # Gender
        gender_output = torch.softmax(gender_output, dim=1)
        gender_class_prob, gender_topclass = torch.max(gender_output, dim=1)
        gender_prob_predict = gender_class_prob.cpu().item()
        gender_predict = GENDER_REVERT[gender_topclass.cpu().item()]

        if gender_prob_predict > self.thresh_hold:
            return gender_predict
        else:
            return None
        
    def predict_image_redis_pil(self, img):
        # inference cv2 image
        GENDER_REVERT = {0: 'male', 1: 'female'}

        img = self.get_transform_PIL(img)
        gender_output = self.model(img)

        # Gender
        gender_output = torch.softmax(gender_output, dim=1)
        gender_class_prob, gender_topclass = torch.max(gender_output, dim=1)
        gender_prob_predict = gender_class_prob.cpu().item()
        gender_predict = GENDER_REVERT[gender_topclass.cpu().item()]

        if gender_prob_predict > self.thresh_hold:
            return gender_predict, gender_prob_predict
        else:
            return None, None
    
    def predict_org(self, img_path):
        GENDER_REVERT = {0: 'male', 1: 'female'}
        
        img = Image.open(img_path).convert('RGB')

        img = self.get_transform(img)
        gender_output = self.model(img)

        # Gender
        gender_output = torch.softmax(gender_output, dim=1)
        gender_class_prob, gender_topclass = torch.max(gender_output, dim=1)
        gender_prob_predict = gender_class_prob.cpu().item()
        gender_predict = GENDER_REVERT[gender_topclass.cpu().item()]

        return gender_prob_predict, gender_predict

if __name__ == '__main__':
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    MODEL_PATH = '/home/nguyen-tuan/Documents/AI/project/API/practice/fastapi/age_gender/weights/gender_10_2_efficientv2_s1_body.pt'
    obj = GenderClass(device)

    # Predict image
    IMAGE_PATH = '/home/nguyen-tuan/Documents/AI/project/API/practice/redis/10in_81.jpg'
    obj.predict(IMAGE_PATH)

    