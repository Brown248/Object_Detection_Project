from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')
    results = model.train(
        data='C:/Object_Detection_Project/dataset/data.yaml',
        epochs=50,
        imgsz=640)

    print("เทรนเสร็จแล้ว ไฟล์โมเดลที่ดีที่สุดจะอยู่ที่ runs/detect/bakery_detection/weights/best.pt")

if __name__ == "__main__":
    main()