import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLOMM

if __name__ == '__main__':
    model = YOLOMM('ultralytics/cfg/models/MSDA-YOLO.yaml')
    # model.load('yolo11n.pt') # loading pretrain weights
    model.train(data='dataset/m3fd_data.yaml',
                cache=True,
                # modality='rgb', 
                imgsz=640,
                epochs=350,
                batch=8,
                close_mosaic=0, 
                workers=4, 
                # device='0,1', 
                optimizer='SGD', 
                # patience=0, 
                # resume='last.pt path', 
                # amp=False, 
                # fraction=0.2, 
                project='runs/M3FD/train',
                name='MSDA-YOLO',
                )




