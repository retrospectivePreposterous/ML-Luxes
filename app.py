import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from PIL import Image, ImageOps 
import cv2
from sklearn.ensemble import AdaBoostRegressor

regr = pickle.load(open('LuxModel.sav', 'rb'))
THRESHOLD = 175
BLUR =   25
KSIZE = 21
DISTANCE = 0.3
VARS = ["orig_thr", "blur_thr", "blur_wa", "blur_ta", "blur_ba", "blur_ha", "op_thr", "dist_thr", "op_counts"]


def mask_ratios(mask):

    mask_ratio = mask.mean() / 255

    mask_vals = [i.sum()/255 for i in mask]
    mask_wr = np.sum(mask,axis=0)
    mask_wr[mask_wr>1] = 1
    mask_wr = np.sum(mask_wr)

    mask_wa = (mask_wr*360) / mask.shape[1]

    mask_tr = 0
    for i in mask: 
        if i.sum()/255 < max(mask_vals)/7:
            mask_tr += 1
        else:
            break

    mask_ta= (mask_tr*180) / mask.shape[0]

    mask_br = 0
    for i in np.flipud(mask):
        if i.sum()/255 < max(mask_vals)/7:
            mask_br+=1
        else:
            break
    mask_ba = (mask_br*180) / mask.shape[0]

    mask_hr = mask.shape[0] - (mask_tr+mask_br)
    mask_ha = (mask_hr*180) / mask.shape[0] 
    
    return [mask_ratio, mask_wa, mask_ta, mask_ba, mask_ha]

def data_from_image(image_path):
    datos =[]
    img = Image.open(image_path) 

    thres= np.copy(img)
    thres[thres > THRESHOLD] = 255
    thres[thres < 255] = 0
    orig_thr = round((thres.mean() / 255),4)
    datos.append(orig_thr) #3. Ratio de umbral original

    image_blur = cv2.medianBlur(thres,BLUR)
    image_blur_gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
    blur_thr, blur_wa, blur_ta, blur_ba, blur_ha = mask_ratios(image_blur_gray)
    datos.append(blur_thr) #4. Ratio de umbral de blur
    datos.append(blur_wa)  #5. Ratio de ancho de blur
    datos.append(blur_ta)  #6. Ratio de superior de blur
    datos.append(blur_ba)  #7. Ratio de inferior de blur
    datos.append(blur_ha)  #8. Ratio de alto de blur

    image_res ,image_thresh = cv2.threshold(image_blur_gray,240,255,cv2.THRESH_BINARY_INV)
    kernel = np.ones((KSIZE,KSIZE),np.uint8)
    opening = cv2.morphologyEx(image_thresh,cv2.MORPH_OPEN,kernel) 
    op_thr = 1 - (opening.mean() / 255)
    datos.append(op_thr) #9. Ratio de Operación Morfológica de Apertura

    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, dist_thr =  cv2.threshold(dist_transform, 0.3*dist_transform.max(),255,0)
    dist_thr = np.uint8(dist_thr)
    dist_thr = 1- (dist_thr.mean() / 255)
    datos.append(dist_thr) #10. Ratio de Filtro de Distancia

    edge = cv2.Canny(opening, 120, 210)
    op_counts, jerarquia = cv2.findContours(edge, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    op_counts = len(op_counts)
    datos.append(op_counts) #11. Ratio de número de clusters de las operaciones morfológicas

    return datos


st.header("AI LUX TOOL")
uploaded_file = st.file_uploader('')

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Loaded Image')
      
    st.write("Extracted data:")
    df = pd.DataFrame(columns=["Values"])
    predicted_vals = data_from_image(uploaded_file)
    for i in range(len(VARS)):
        df.loc[VARS[i]] = predicted_vals[i]
    st.write(df)
    
    prediction = float(regr.predict([predicted_vals]))
    st.write("The predicted Luxes are:", prediction)
    
 
    p1 = "If the picture belongs to a NON GREEN AREA (P1): "
    if prediction < 3:
        st.write(p1 + "**UNDERLIT**")
    elif prediction < 15:
        st.write(p1 + "**PROPERLY LIT**")
    else:
        st.write(p1 + "**OVERLIT**")
    
    p2 = "If the picture belongs to a GREEN AREA (P2): "
    if prediction < 2:
        st.write(p2 + "**UNDERLIT**")
    elif prediction < 10:
        st.write(p2 + "**PROPERLY LIT**")
    else:
        st.write(p2 + "**OVERLIT**")