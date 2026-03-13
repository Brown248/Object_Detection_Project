from ultralytics import YOLO

def main():
    # 1. โหลดโมเดลเปล่า 
    model = YOLO('yolov8n.pt')

    # 2. สั่งเทรน
    # data: ชี้ไปที่ไฟล์ data.yaml ที่เราสร้างไว้
    # epochs: จำนวนรอบที่จะให้โมเดลเรียนรู้ (ถ้าข้อมูลน้อย เอาซัก 50-100 รอบ)
    # imgsz: ขนาดภาพที่ใช้เทรน (640x640 เป็นมาตรฐาน)
    results = model.train(
        data='C:/Object_Detection_Project/dataset/data.yaml',
        epochs=50,
        imgsz=640)

    print("เทรนเสร็จแล้ว! ไฟล์โมเดลที่ดีที่สุดจะอยู่ที่ runs/detect/bakery_detection/weights/best.pt")

if __name__ == "__main__":
    main()