from ultralytics import YOLO
import glob
import os

def main():
    print("กำลังค้นหาโมเดลที่ดีที่สุด")
    
    list_of_files = glob.glob('C:/Object_Detection_Project/runs/detect/*/weights/best.pt')
    
    if not list_of_files:
        print("หาไฟล์โมเดลไม่เจอ")
        return
        
    latest_model_path = max(list_of_files, key=os.path.getctime).replace('\\', '/')
    print(f"พบไฟล์โมเดลล่าสุดที่: {latest_model_path}")
    
    model = YOLO(latest_model_path)

    print("\nกำลังประเมินประสิทธิภาพกับชุดข้อมูล Validation...")
    metrics = model.val(data='C:/Object_Detection_Project/dataset/data.yaml')

    # ดึงค่าสถิติจาก dictionary ของ metrics
    precision = metrics.results_dict['metrics/precision(B)']
    recall = metrics.results_dict['metrics/recall(B)']
    map50 = metrics.results_dict['metrics/mAP50(B)']
    map50_95 = metrics.results_dict['metrics/mAP50-95(B)']

    # แสดงผลลัพธ์แบบสวยงาม
    print("\n" + "="*60)
    print("สรุปผลการประเมินประสิทธิภาพ")
    print("="*60)
    
    print(f" Precision (ความแม่นยำเมื่อทายว่าเจอ): {precision:.4f}  ({precision*100:.2f}%)")
    print(f" Recall (ความสามารถในการกวาดหาวัตถุเจอ): {recall:.4f}  ({recall*100:.2f}%)")
    print(f" mAP@50 (ความแม่นยำเฉลี่ยมาตรฐาน): {map50:.4f}  ({map50*100:.2f}%)")
    print(f" mAP@50-95 (ความแม่นยำเฉลี่ยแบบเข้มงวด): {map50_95:.4f}  ({map50_95*100:.2f}%)")

if __name__ == '__main__':
    main()