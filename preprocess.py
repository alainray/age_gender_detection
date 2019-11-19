import pandas as pd
from datetime import datetime, timedelta
import scipy.io
import torch
def datenum_to_datetime(datenum):
    """
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    days = datenum % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60
    return datetime.fromordinal(int(datenum)) \
           + timedelta(days=int(days)) \
           + timedelta(hours=int(hours)) \
           + timedelta(minutes=int(minutes)) \
           + timedelta(seconds=round(seconds)) \
           - timedelta(days=366)
def to_age_range(age):
  if age <=15:
    return 0 # Child
  elif age > 15 and age <=30:
    return 1 # Young adult
  elif age > 30 and age <=60:
    return 2 # Adult
  else:
    return 3 # Elderly > 60
## ------------------------------------------------------------------------------
# 0 - dob: date of birth (Matlab serial date number)
# 1 - photo_taken: year when the photo was taken
# 2 - full_path: path to file
# 3 - gender: 0 for female and 1 for male, NaN if unknown
# 4 - name: name of the celebrity
# 5 - face_location: location of the face. To crop the face in Matlab run img(face_location(2):face_location(4),face_location(1):face_location(3),:))
# 6 - face_score: detector score (the higher the better). Inf implies that no face was found in the image and the face_location then just returns the entire image
# 7 - second_face_score: detector score of the face with the second highest score. This is useful to ignore images with more than one face. second_face_score is NaN if no second face was detected.
# 8 - celeb_names (IMDB only): list of all celebrity names
# 9 celeb_id (IMDB only): index of celebrity name

mat = scipy.io.loadmat('imdb/imdb.mat')
gender_target = mat['imdb'][0][0][3][0]
age_dob = mat['imdb'][0][0][0][0]
age_photo_taken = mat['imdb'][0][0][1][0]
img_paths = [x[0] for x in mat['imdb'][0][0][2][0]]
face_scores = mat['imdb'][0][0][6][0]
#b = face_scores == -float('inf')
frame = { 'path': img_paths, 'dob': age_dob, 'photo_taken': age_photo_taken, 'face_scores': face_scores, 'gender': gender_target} 
df = pd.DataFrame(frame)
df = df.dropna()
# Filter no faces
filter_inf = df['face_scores']!=-float('inf')
ds = df[filter_inf]
# Filter people from before the 1800s
filter_old = ds['dob']> 365*1800
de = ds[filter_old]
# From year to date 
de['photo_date'] = de['photo_taken'].apply(lambda x: datetime(x, 6, 15))
# From Matlab date format to python format
de['birth_date'] = de['dob'].apply(lambda x: datetime(1, 1, 1) + timedelta(x) - timedelta(366))
# Calculate age
de['age'] = de['photo_date']- de['birth_date']
de['age'] = de['age'].apply(lambda x: int(x.days/365))
de['age'] = de['age'].apply(lambda x: x if x<=100 else 100) # We clip to 100
# Finally we create an age range
de['age_range'] = de['age'].apply(to_age_range)

X = de['path'].tolist()
y = [(x,y) for x,y in zip(de['gender'].tolist(), de['age_range'].tolist())]

data = {'img': X, 'target': y}

torch.save(data, "features.pth")
