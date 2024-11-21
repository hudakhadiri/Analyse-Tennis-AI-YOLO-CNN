from ultralytics import YOLO

model = YOLO('models/best.pt')

model.predict('input_videos/image.png',conf=0.2,save=True)
model.predict('input_videos/input_video.mp4',save=True)
