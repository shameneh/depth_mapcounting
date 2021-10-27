import torch
import matplotlib.pyplot as plt
import matplotlib.cm as CM
from tqdm import tqdm

from model import CSRNet
from dataset import CrowdDataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from PIL import Image
from torchvision import transforms
from pathlib import Path
import numpy as np
from utils import denormalize
import csv
def _visualize_(img,dmap,output,itr):
    #print(dmap.shape,np.max(np.array(img)))
   # keep the same aspect ratio as an input image
    fig, ax = plt.subplots(figsize=figaspect(1.0 * img.shape[0] / img.shape[1]))
    fig.subplots_adjust(0, 0, 1, 1)


    # plot a density map without axis
    ax.imshow(dmap, cmap="jet")
    plt.axis('off')
    fig.canvas.draw()


    # create a PIL image from a matplotlib figure
    dmap = Image.frombytes('RGB',
                           fig.canvas.get_width_height(),
                           fig.canvas.tostring_rgb())


    # add a alpha channel proportional to a density map value
    dmap.putalpha(dmap.convert('L'))
    img = Image.fromarray((img* 255).astype(np.uint8))
    # display an image with density map put on top of it
    alphacom = Image.alpha_composite(img.convert('RGBA'), dmap.resize(img.size))
    filename ='overlap_'+str(itr)+'.png'
    alphacom.save(Path(output,filename),format="png")
    plt.close('all')
    


def cal_mae(img_root,gt_dmap_root,model_param_path,out_path):
    '''
    Calculate the MAE of the test data.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    '''

    device=torch.device("cuda")
    model=CSRNet()
    model.load_state_dict(torch.load(model_param_path))
    model.to(device)
    dataset=CrowdDataset(img_root,gt_dmap_root,img_transform=ToTensor(), dmap_transform=ToTensor())
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    model.eval()
    mae=0
    header = ['image','sigma','gt_count','dmap_csr']


    dir_csv = Path(out_path , 'count_res_CSR_15.csv')
    with open(dir_csv, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    
    with torch.no_grad():
        for i,data in enumerate(tqdm(dataloader)):
            img = data['image'].to(device)
            gt_dmap = data['densitymap'].to(device)
            image_name = data['image_name']
            #img=img.to(device)
            #gt_dmap=gt_dmap.to(device)
            # forward propagation
            et_dmap=model(img)
            data_csv = [image_name, '15',str(gt_dmap.data.sum().item()),str(et_dmap.data.sum().item())]
            with open(dir_csv, 'a', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(data_csv)
            print("Mae image : "+str(i)+' pred: '+str(et_dmap.data.sum().item())+' gt: '+str(gt_dmap.data.sum().item()))
            mae+=abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()
            del img,gt_dmap,et_dmap

    print("model_param_path:"+model_param_path+" mae:"+str(mae/len(dataloader)))

def estimate_density_map(img_root,gt_dmap_root,model_param_path,index,output):
    '''
    Show one estimated density-map.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    index: the order of the test image in test dataset.
    '''
    device=torch.device("cuda")
    model=CSRNet().to(device)
    model.load_state_dict(torch.load(model_param_path))
    dataset=CrowdDataset(img_root,gt_dmap_root,img_transform=ToTensor(), dmap_transform=ToTensor())
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    model.eval()
    for i,data in enumerate(dataloader):
        if i<index:
            img = data['image'].to(device)
            gt_dmap = data['densitymap'].to(device)
            image_name = data['image_name']
#            img=img.to(device)
#            gt_dmap=gt_dmap.to(device)
            # forward propagation
            with torch.no_grad():
                et_dmap=model(img).detach()
            et_dmap=et_dmap.squeeze(0).squeeze(0).cpu().numpy()
            
            img=img.cpu().squeeze(0).squeeze(0).permute(1,2,0).cpu().numpy()
            gt_dmap=gt_dmap.squeeze(0).squeeze(0).cpu().numpy()
            _visualize_(img,et_dmap,output,i)
            plt.imshow(et_dmap,cmap=CM.jet)
            plt.savefig(f'{output}/pred_{image_name}.png')
            plt.imshow(gt_dmap,cmap=CM.jet)
            plt.savefig(f'{output}/gt_{image_name}.png')
            plt.imshow(img)
            plt.savefig(f'{output}/img_{image_name}.png')


if __name__=="__main__":
    torch.backends.cudnn.enabled=False
    img_root='../datasets'
    gt_dmap_root='test_whole_videos'
    model_param_path='./checkpoints/old/5.pth'
    cal_mae(img_root,gt_dmap_root,model_param_path,'test_results_whole')
    #estimate_density_map(img_root,gt_dmap_root,model_param_path,100000,'test_results_whole') 
