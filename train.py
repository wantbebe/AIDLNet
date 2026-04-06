import warnings, os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    # 
os.environ["CUDA_VISIBLE_DEVICES"]="0"   
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'') # # select your model.pt path
    # model.load('yolo11n.pt') # loading pretrain weights
    model.train(data=r'',) # select your data.pt path
                cache=False,
                imgsz=640,
                epochs=100,
                batch=32,
                close_mosaic=0, # 
                workers=0, # 
                # device='0,1', # 
                optimizer='SGD', # using SGD
                # patience=0, # set 0 to close earlystop.
                # resume=True,
                # amp=False, # close amp 
                # fraction=0.2,
                project='runs_new/train/',
                name='TIFD',
                )