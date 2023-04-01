from monai.transforms import(
    Compose,
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    
)
from monai.data import DataLoader, Dataset


def transform(data,a_min, a_max, spatial_size, pixdim):
    train_transforms = Compose(
      [
          LoadImaged(keys=["image", "label"]),
          AddChanneld(keys=["image", "label"]),
          # Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")), # not clear
          Orientationd(keys=["image", "label"], axcodes="RAS"),
          ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True), 
          CropForegroundd(keys=["image", "label"], source_key="image"),
          Resized(keys=["image", "label"], spatial_size=spatial_size),
          ToTensord(keys=["image", "label"])

      ]
    )

    ds = Dataset(data=data, transform=train_transforms)
    loader = DataLoader(ds, batch_size=1)

    return loader