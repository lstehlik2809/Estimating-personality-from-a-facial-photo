# -*- coding: utf-8 -*-

"""
Experimental Python script that estimates personality traits from facial photos using deep learning models.

Author: Ludek Stehlik (ludek.stehlik@gmail.com)

Caveat: This script is intended strictly for experimental and educational purposes. It does not provide a valid,
reliable, or scientifically endorsed method for inferring personality traits from facial images. Personality is a
complex, multi-dimensional construct that cannot be accurately assessed from appearance alone. Any apparent patterns
or outputs should be interpreted as illustrative artifacts of the dataset and method—not as diagnostic or predictive
insights. Use responsibly and with critical awareness of ethical and scientific limitations.
"""

# libraries to be used
from deepface import DeepFace # pip install deepface
from numpy.linalg import norm
from numpy import dot
from typing import Literal
import numpy as np

# function to estimate personality from a facial photo
def estimate_personality_from_facial_photo(photo_gender: Literal["male", "female"], path_to_photo: str, path_to_images_folder: str, model='Facenet512')-> dict:
  if photo_gender not in {"male", "female"}:
    raise ValueError(f"Invalid gender '{photo_gender}'. Must be 'male' or 'female'.")

  # getting embeddings for a photo
  embedding_photo = DeepFace.represent(img_path=path_to_photo, model_name=model)[0]["embedding"]
  embedding_photo = embedding_photo / norm(embedding_photo )

  # dictionary for saving results: more similar personality "prototype" image + abs. diff. in cosine sim to get better idea about the strength of evidence
  results = dict()

  # getting embeddings for personality "prototype" images

  # for males
  if photo_gender == "male":
    # agreeableness
    embedding_agr_m1 = DeepFace.represent(img_path=f"{path_to_images_folder}/Agr_MA.png", model_name=model)[0]["embedding"]
    embedding_agr_m1 = embedding_agr_m1 / norm(embedding_agr_m1)
    embedding_agr_m0 = DeepFace.represent(img_path=f"{path_to_images_folder}/Agr_MB.png", model_name=model)[0]["embedding"]
    embedding_agr_m0 = embedding_agr_m0 / norm(embedding_agr_m0)
    # cosine similarity [-1, 1]
    agr_m1_dis = dot(embedding_photo, embedding_agr_m1)
    agr_m0_dis = dot(embedding_photo, embedding_agr_m0)
    # comparing similarities
    if agr_m1_dis > agr_m0_dis:
      agr='More agreeable'
    elif agr_m1_dis == agr_m0_dis:
      agr='Moderately agreeable'
    else:
      agr='Less agreeable'
    results['Agreeableness'] = [agr, f"Abs. diff. in cosine sim.: {np.round(np.abs(agr_m1_dis-agr_m0_dis),3)}"]

    # conscientiousness
    embedding_con_m1 = DeepFace.represent(img_path=f"{path_to_images_folder}/Con_MA.png", model_name=model)[0]["embedding"]
    embedding_con_m1 = embedding_con_m1 / norm(embedding_con_m1)
    embedding_con_m0 = DeepFace.represent(img_path=f"{path_to_images_folder}/Con_MB.png", model_name=model)[0]["embedding"]
    embedding_con_m0 = embedding_con_m0 / norm(embedding_con_m0)
    # cosine similarity [-1, 1]
    con_m1_dis = dot(embedding_photo, embedding_con_m1)
    con_m0_dis = dot(embedding_photo, embedding_con_m0)
    # comparing similarities
    if con_m1_dis > con_m0_dis:
      con='More conscientious'
    elif con_m1_dis == con_m0_dis:
      con='Moderately conscientious'
    else:
      con='Less conscientious'
    results['Conscientiousness'] = [con, f"Abs. diff. in cosine sim.: {np.round(np.abs(con_m1_dis-con_m0_dis),3)}"]

    # extraversion
    embedding_ext_m1 = DeepFace.represent(img_path=f"{path_to_images_folder}/Ext_MA.png", model_name=model)[0]["embedding"]
    embedding_ext_m1 = embedding_ext_m1 / norm(embedding_ext_m1)
    embedding_ext_m0 = DeepFace.represent(img_path=f"{path_to_images_folder}/Ext_MB.png", model_name=model)[0]["embedding"]
    embedding_ext_m0 = embedding_ext_m0 / norm(embedding_ext_m0)
    # cosine similarity [-1, 1]
    ext_m1_dis = dot(embedding_photo, embedding_ext_m1)
    ext_m0_dis = dot(embedding_photo, embedding_ext_m0)
    # comparing similarities
    if ext_m1_dis > ext_m0_dis:
      ext='More extraverted'
    elif ext_m1_dis == ext_m0_dis:
      ext='Moderately extraverted'
    else:
      ext='Less extraverted'
    results['Extraversion'] = [ext, f"Abs. diff. in cosine sim.: {np.round(np.abs(ext_m1_dis-ext_m0_dis),3)}"]

    # neuroticism
    embedding_neu_m1 = DeepFace.represent(img_path=f"{path_to_images_folder}/Neu_MA.png", model_name=model)[0]["embedding"]
    embedding_neu_m1 = embedding_neu_m1 / norm(embedding_neu_m1)
    embedding_neu_m0 = DeepFace.represent(img_path=f"{path_to_images_folder}/Neu_MB.png", model_name=model)[0]["embedding"]
    embedding_neu_m0 = embedding_neu_m0 / norm(embedding_neu_m0)
    # cosine similarity [-1, 1]
    neu_m1_dis = dot(embedding_photo, embedding_neu_m1)
    neu_m0_dis = dot(embedding_photo, embedding_neu_m0)
    # comparing similarities
    if neu_m1_dis > neu_m0_dis:
      neu='Less emotionally stable'
    elif neu_m1_dis == neu_m0_dis:
      neu='Moderately emotionally stable'
    else:
      neu='More emotionally stable'
    results['Emotional Stability'] = [neu, f"Abs. diff. in cosine sim.: {np.round(np.abs(neu_m1_dis-neu_m0_dis),3)}"]

    # openness
    embedding_ope_m1 = DeepFace.represent(img_path=f"{path_to_images_folder}/Ope_MA.png", model_name=model)[0]["embedding"]
    embedding_ope_m1 = embedding_ope_m1 / norm(embedding_ope_m1)
    embedding_ope_m0 = DeepFace.represent(img_path=f"{path_to_images_folder}/Ope_MB.png", model_name=model)[0]["embedding"]
    embedding_ope_m0 = embedding_ope_m0 / norm(embedding_ope_m0)
    # cosine similarity [-1, 1]
    ope_m1_dis = dot(embedding_photo, embedding_ope_m1)
    ope_m0_dis = dot(embedding_photo, embedding_ope_m0)
    # comparing similarities
    if ope_m1_dis > ope_m0_dis:
      ope='More open'
    elif ope_m1_dis == ope_m0_dis:
      ope='Moderately open'
    else:
      ope='Less open'
    results['Openness'] = [ope, f"Abs. diff. in cosine sim.: {np.round(np.abs(ope_m1_dis-ope_m0_dis),3)}"]

  # for females
  else:
    # agreeableness
    embedding_agr_m1 = DeepFace.represent(img_path=f"{path_to_images_folder}/Agr_FA.png", model_name=model)[0]["embedding"]
    embedding_agr_m1 = embedding_agr_m1 / norm(embedding_agr_m1)
    embedding_agr_m0 = DeepFace.represent(img_path=f"{path_to_images_folder}/Agr_FB.png", model_name=model)[0]["embedding"]
    embedding_agr_m0 = embedding_agr_m0 / norm(embedding_agr_m0)
    # cosine similarity [-1, 1]
    agr_m1_dis = dot(embedding_photo, embedding_agr_m1)
    agr_m0_dis = dot(embedding_photo, embedding_agr_m0)
    # comparing similarities
    if agr_m1_dis > agr_m0_dis:
      agr='More agreeable'
    elif agr_m1_dis == agr_m0_dis:
      agr='Moderately agreeable'
    else:
      agr='Less agreeable'
    results['Agreeableness'] = [agr, f"Abs. diff. in cosine sim.: {np.round(np.abs(agr_m1_dis-agr_m0_dis),3)}"]

    # conscientiousness
    embedding_con_m1 = DeepFace.represent(img_path=f"{path_to_images_folder}/Con_FA.png", model_name=model)[0]["embedding"]
    embedding_con_m1 = embedding_con_m1 / norm(embedding_con_m1)
    embedding_con_m0 = DeepFace.represent(img_path=f"{path_to_images_folder}/Con_FB.png", model_name=model)[0]["embedding"]
    embedding_con_m0 = embedding_con_m0 / norm(embedding_con_m0)
    # cosine similarity [-1, 1]
    con_m1_dis = dot(embedding_photo, embedding_con_m1)
    con_m0_dis = dot(embedding_photo, embedding_con_m0)
    # comparing similarities
    if con_m1_dis > con_m0_dis:
      con='More conscientious'
    elif con_m1_dis == con_m0_dis:
      con='Moderately conscientious'
    else:
      con='Less conscientious'
    results['Conscientiousness'] = [con, f"Abs. diff. in cosine sim.: {np.round(np.abs(con_m1_dis-con_m0_dis),3)}"]

    # extraversion
    embedding_ext_m1 = DeepFace.represent(img_path=f"{path_to_images_folder}/Ext_FA.png", model_name=model)[0]["embedding"]
    embedding_ext_m1 = embedding_ext_m1 / norm(embedding_ext_m1)
    embedding_ext_m0 = DeepFace.represent(img_path=f"{path_to_images_folder}/Ext_FB.png", model_name=model)[0]["embedding"]
    embedding_ext_m0 = embedding_ext_m0 / norm(embedding_ext_m0)
    # cosine similarity [-1, 1]
    ext_m1_dis = dot(embedding_photo, embedding_ext_m1)
    ext_m0_dis = dot(embedding_photo, embedding_ext_m0)
    # comparing similarities
    if ext_m1_dis > ext_m0_dis:
      ext='More extraverted'
    elif ext_m1_dis == ext_m0_dis:
      ext='Moderately extraverted'
    else:
      ext='Less extraverted'
    results['Extraversion'] = [ext, f"Abs. diff. in cosine sim.: {np.round(np.abs(ext_m1_dis-ext_m0_dis),3)}"]

    # neuroticism
    embedding_neu_m1 = DeepFace.represent(img_path=f"{path_to_images_folder}/Neu_FA.png", model_name=model)[0]["embedding"]
    embedding_neu_m1 = embedding_neu_m1 / norm(embedding_neu_m1)
    embedding_neu_m0 = DeepFace.represent(img_path=f"{path_to_images_folder}/Neu_FB.png", model_name=model)[0]["embedding"]
    embedding_neu_m0 = embedding_neu_m0 / norm(embedding_neu_m0)
    # cosine similarity [-1, 1]
    neu_m1_dis = dot(embedding_photo, embedding_neu_m1)
    neu_m0_dis = dot(embedding_photo, embedding_neu_m0)
    # comparing similarities
    if neu_m1_dis > neu_m0_dis:
      neu='Less emotionally stable'
    elif neu_m1_dis == neu_m0_dis:
      neu='Moderately emotionally stable'
    else:
      neu='More emotionally stable'
    results['Emotional Stability'] = [neu, f"Abs. diff. in cosine sim.: {np.round(np.abs(neu_m1_dis-neu_m0_dis),3)}"]

    # openness
    embedding_ope_m1 = DeepFace.represent(img_path=f"{path_to_images_folder}/Ope_FA.png", model_name=model)[0]["embedding"]
    embedding_ope_m1 = embedding_ope_m1 / norm(embedding_ope_m1)
    embedding_ope_m0 = DeepFace.represent(img_path=f"{path_to_images_folder}/Ope_FB.png", model_name=model)[0]["embedding"]
    embedding_ope_m0 = embedding_ope_m0 / norm(embedding_ope_m0)
    # cosine similarity [-1, 1]
    ope_m1_dis = dot(embedding_photo, embedding_ope_m1)
    ope_m0_dis = dot(embedding_photo, embedding_ope_m0)
    # comparing similarities
    if ope_m1_dis > ope_m0_dis:
      ope='More open'
    elif ope_m1_dis == ope_m0_dis:
      ope='Moderately open'
    else:
      ope='Less open'
    results['Openness'] = [ope, f"Abs. diff. in cosine sim.: {np.round(np.abs(ope_m1_dis-ope_m0_dis),3)}"]


  return results

# running the function
estimate_personality_from_facial_photo(
    photo_gender='male',
    path_to_photo="./photo/Ludek_Stehlik_Profile_Photo.jpg",
    path_to_images_folder="./images",
    model='Facenet512'
    )

