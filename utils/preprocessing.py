import nibabel as nib
import json
import numpy as np

def load_nifti(image_nifty_file, label_nifty_file):
    # load the image and label file, get the image content and return a numpy array for each
    image = nib.load(image_nifty_file)
    label = nib.load(label_nifty_file)
    
    return image, label

def save_to_json(data, path):
  with open(path, 'w') as fp:
    json.dump(data, fp)


def remove_slices(img,start, end):
  imgvol = np.array( img.dataobj )
  imgvol = imgvol[ :, :, start:end ]
  newimg = nib.Nifti1Image ( imgvol, img.affine )
  return newimg

def create_same_slice_nifti(data, slice_size ,dir):
  paths = []
  total = len(data)
  count = 1
  for entry in data:
    img, lbl = load_nifti(entry["image"], entry["label"])

    total_slize_size =img.shape[2]
    if(total_slize_size < slice_size): print("ERROR: slice upper limit exceeds")
    extra_slices = total_slize_size - slice_size
    end  = total_slize_size - (extra_slices // 2 )
    start = end - slice_size 

    newimg = remove_slices(img,start, end)
    newlbl = remove_slices(lbl,start, end)
    image_path = entry["image"].replace('PROSTATEx_masks/Files', "data/sliced")
    label_path = entry["label"].replace('PROSTATEx_masks/Files', "data/sliced")
    paths.append({"image":image_path, "label":label_path})
    newimg.to_filename(image_path );
    newlbl.to_filename(label_path);
    
    print(f"{count}/{total}")
    count += 1

  save_to_json({"path": paths}, dir + 'config.json')



def create_same_slice_zone(data, slice_size ,dir):
  paths = []
  total = len(data)
  count = 1
  for entry in data:
    img, lbl = load_nifti(entry["image"], entry["label"])

    total_slize_size =img.shape[2]
    if(total_slize_size < slice_size): print("ERROR: slice upper limit exceeds")
    extra_slices = total_slize_size - slice_size
    end  = total_slize_size - (extra_slices // 2 )
    start = end - slice_size 

    newlbl = remove_slices(lbl,start, end)
    image_path = entry["image"].replace('PROSTATEx_masks/Files', "data/sliced")
    label_path = entry["label"].replace('PROSTATEx_masks/Files', "data/sliced")
    paths.append({"image":image_path, "label":label_path})
    newlbl.to_filename(label_path)
    
    print(f"{count}/{total}")
    count += 1

  save_to_json({"path": paths}, dir)