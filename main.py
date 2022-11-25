import os
import clip
import torch
import pandas as pd
import os
from tqdm import tqdm
import urllib.request
import nltk
from PIL import Image

import numpy as np
import argparse


# Arguments
parser = argparse.ArgumentParser(description='PyTorch Clip-zeroshot')
parser.add_argument('--cnn', type=str, default='ViT-B/32',
                    help='Which image encoder do you want to use')
parser.add_argument('--csv', type=str, default='test.csv',
                    help='Name of the file where metrics are going to be saved')
parser.add_argument('--file_df', type=str, default='Med_Modality',
                    help='Tipe of labels over which you want to perform experimentation')
parser.add_argument('--tuning', type=str, default='ViT.pth',
                    help='Which model weights you want to upload')
parser.add_argument('--mode', type=str, default='demo',
                    help='model in which you want to run the main.py')
parser.add_argument('--img', type=str, default='synpic31463.jpg',
                    help='If you are in demo mode which image you want to perform demo on')

args = parser.parse_args()



# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"


model, preprocess = clip.load(args.cnn, device)

#If the model es fine tuning 
if args.tuning == "no":
    z = 0

else:
    our_model = os.path.join('models',args.tuning)#rut<

    state_dict = torch.load(our_model)#todo el pth

    model.load_state_dict(state_dict['state_dict'], strict=False)


'''
_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
    }
'''

# Download the dataset

df = pd.DataFrame(pd.read_csv(os.path.join('MedCLIP','medpix.csv')))

#New Med Modality
a = df["Core_Modality"].unique() #clases of Core Modality
b = df["Full_Modality"].unique() #clases of Full Modality

#List of Med modality
dic = ["Computed tomography","X-ray","Magnetic resonance","Intravenous urogram","UltraSound", "Upper Gastrointestinal series",
       "Mammograph","Nuclear medicine", "Computed tomography angiography", "Angiogram", "Not otherwise specified",
       "Barium swallow", "Gross photograph","Histology","Doppler UltraSound", "Montage of images","Not assigned",
       "Magnetic resonance angiography", "Positron Emission", "Positron Emission Computed tomography fusion",
       "Barium Enema", "Hepatic encephalopathy", "Electron microscopic", "Clinical photograph", "Magnetic resonance spectroscopy",
       "Small Bowel follow trough","Interventional procedure","Virtual Colonoscopy", "Hysterosalpingogram",
       "Operative photograph", "Electrocardiogram","Endoscopy", "Voiding Cystourethrogram","Retrograde Urogram",
       "Fundoscopy","Special"]

medMod = []
for x in range(0,len(df)):
    indic = df["Core_Modality"][x]
    positio = np.where(a == indic)[0][0]
    medMod.append(dic[positio])

df["Med_Modality"] = medMod


#construct 
df = df.replace(np.nan," ")
clases = df[args.file_df].unique()
base = df[args.file_df]

predictions = []
anotations = []

if args.mode == "test":
    # Prepare the inputs
    for url in tqdm(range(0,int(len(df)))):
        name = df["Image_URL"][url]
        filename = name.split("/")[-1]
        try:
            
            image_input = preprocess(Image.open(os.path.join('MedCLIP','medpix_dataset',filename))).unsqueeze(0).to(device)
            text_inputs = torch.cat([clip.tokenize(f"Medical image of {c}") for c in clases]).to(device)#, truncate=True) for c in clases]).to(device)

            # Calculate features
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_inputs)

            # Pick the top 5 most similar labels for the image
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(5)
            # Print the result
            print("\nTop predictions:\n")
            
            valu = False
            for value, index in zip(values, indices):
                print(f"{clases[index]:>16s}: {100 * value.item():.2f}%")
                
            if valu == False:
                predictions.append(clases[indices[0].cpu().numpy()])
                anotations.append(base[url])
                
        except:
            continue

    import sklearn
    from sklearn.metrics import classification_report
    report = classification_report(anotations,predictions)
    print(report)  

    import pandas as pd

    clsf_report = pd.DataFrame(classification_report(anotations,predictions, output_dict=True)).transpose()
    clsf_report.to_csv(args.csv, index= True)

if args.mode == "demo":

    image_input = preprocess(Image.open(os.path.join('MedCLIP','medpix_dataset',args.img))).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(f"Medical image of {c}") for c in clases]).to(device)#, truncate=True) for c in clases]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)
    # Print the result
    print("\nTop predictions:\n")
    
    valu = False
    for value, index in zip(values, indices):
        print(f"{clases[index]:>16s}: {100 * value.item():.2f}%")
    
    x1 = 0 
    for url in (range(0,int(len(df)))):
        name = df["Image_URL"][url]
        filename = name.split("/")[-1]
        if filename == args.img:
            x1 = url
            break


    predictions.append(clases[indices[0].cpu().numpy()])
    anotations.append(base[x1])
    print(f'The prediction is {predictions} and the real class is {anotations}')
