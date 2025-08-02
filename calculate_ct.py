import os
import csv
import numpy as np
from PIL import Image
import torch

class CTCalculator:
    def __init__(self):
        self.sobel_x = torch.tensor([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]], dtype=torch.float32)
        self.sobel_y = torch.tensor([[-1, -2, -1],
                                   [ 0,  0,  0],
                                   [ 1,  2,  1]], dtype=torch.float32)

    def compute_ct(self, image_path):
        try:
            with Image.open(image_path) as img:
                img_gray = img.convert('L')
                img_array = np.array(img_gray)
                H, W = img_array.shape
                
                img_tensor = torch.from_numpy(img_array).float().unsqueeze(0).unsqueeze(0)
                
                grad_x = torch.nn.functional.conv2d(img_tensor, self.sobel_x.view(1,1,3,3), padding=1)
                grad_y = torch.nn.functional.conv2d(img_tensor, self.sobel_y.view(1,1,3,3), padding=1)
                
                sum_abs = (torch.abs(grad_x) + torch.abs(grad_y)).sum().item()
                return round(sum_abs / (H * W), 4)
                
        except Exception as e:
            return None

def generate_sorted_ct_report(input_dir, output_csv):
    calculator = CTCalculator()
    data = []
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)  
                ct = calculator.compute_ct(file_path)
                if ct is not None:
                    data.append((ct, file_path))  
    
    sorted_data = sorted(data, key=lambda x: x[0])
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ct_value', 'filepath']) 
        writer.writerows(sorted_data)
    

if __name__ == "__main__":
    input_folder = ""
    output_csv = ""
    
    generate_sorted_ct_report(input_folder, output_csv)