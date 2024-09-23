import torch,torchvision,cv2,torchinfo
from SRFormer.basicsr.archs.srformer_arch import SRFormer
import numpy as np


def load_model(device:torch.device,
               model:torch.nn.Module,
               ckpt_dir:str="/home/muahmmad/projects/Image_enhancement/waternet/weights/waternet_exported_state_dict-daa0ee.pt"):
    model = model
    ckpt=torch.load(f=ckpt_dir,map_location=device,weights_only=True)
    print(ckpt.keys())
    model.load_state_dict(state_dict=ckpt["params"])
    model=model.to(device=device)
    return model
def transform_array_to_image(arr):
    arr=np.clip(a=arr,a_min=0,a_max=1)
    arr=(arr*255.0).astype(np.uint8)
    return arr
def transform_image(img,device,single_image:bool=True):
    trans=torchvision.transforms.Compose(transforms=[
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(size=(480,480),interpolation=torchvision.transforms.InterpolationMode.NEAREST_EXACT),
        torchvision.transforms.Normalize(mean=[0,0,0],
                                         std=[1,1,1])
    ])
    raw_image_tensor = trans(img)
    if single_image:
        return {"X":torch.unsqueeze(input=raw_image_tensor,dim=0).to(device)}
    else:
        wb_tensor=trans(wb)
        gc_tensor=trans(gc)
        he_tensor=trans(he)
        return {"X":torch.unsqueeze(input=raw_image_tensor,dim=0).to(device),
                "wb":torch.unsqueeze(input=wb_tensor,dim=0).to(device),
                "gc":torch.unsqueeze(input=gc_tensor,dim=0).to(device),
                "he":torch.unsqueeze(input=he_tensor,dim=0).to(device)}
if __name__=='__main__':
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt="/home/cplus/projects/m.tarek_master/Image_enhancement/weights/SRFormer/SRFormer_SRx4_DF2K.pth"
    model=load_model(device=device,ckpt_dir=ckpt,model=SRFormer(upscale= 4,
      in_chans= 3,
      img_size= 64,
      window_size= 22,
      img_range= 1.0,
      depths= [6, 6, 6, 6, 6, 6],
      embed_dim= 180,
      num_heads= [6, 6, 6, 6, 6, 6],
      mlp_ratio= 2,
      upsampler= "pixelshuffle",
      resi_connection="1conv"))
    image=cv2.imread(filename="/home/cplus/projects/m.tarek_master/Image_enhancement/images/left_image/000224_224_left.jpg")
    raw_tensor=transform_image(img=image,single_image=True,device=device)["X"]
    with torch.no_grad():
        pred=model(raw_tensor)
        #pred=torch.nn.functional.interpolate(input=pred[0],size=(720,720),mode="bilinear")


    pred=pred.squeeze_()
    pred=torch.permute(pred,dims=(1,2,0))
    pred=pred.detach().cpu().numpy()
    pred=transform_array_to_image(pred)
    cv2.imshow("test",pred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

