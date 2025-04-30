import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import os
from pathlib import Path
import pytesseract
from pytesseract import Output

# Configure matplotlib to use a Thai-compatible font
plt.rcParams['font.family'] = 'Tahoma'  # or 'Arial', 'Angsana New', 'Cordia New', etc.
plt.rcParams['axes.unicode_minus'] = False

# Configure Tesseract path
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
else:
    print("Warning: Tesseract OCR not found at default path. Please install Tesseract OCR and set the correct path.")
    print("Default path checked:", TESSERACT_PATH)
    print("You can set the correct path by modifying TESSERACT_PATH in the code.")

def load_model_and_processor():
    """
    โหลดโมเดลและ processor สำหรับการแปลงรูปภาพเป็นข้อความ
    เลือกใช้โมเดล microsoft/git-base ซึ่งเป็นโมเดลที่ไม่ต้องการ token
    
    ตรวจสอบว่ามีโมเดลใน local หรือไม่ ถ้ามีจะใช้โมเดลใน local แทน
    """
    # ใช้โมเดล Git-base จาก Microsoft ซึ่งเป็นโมเดลขนาดกลางที่มีประสิทธิภาพดี
    model_name = "microsoft/git-base"
    
    # สร้างโฟลเดอร์เก็บโมเดลใน local
    local_model_path = "./model_cache/text_to_image"
    # local_model_path.mkdir(parents=True, exist_ok=True)
    
    # ตรวจสอบว่ามีโมเดลใน local หรือไม่
    if os.path.exists(local_model_path):
        print(f"กำลังโหลดโมเดลจาก local: {local_model_path}")
        processor = AutoProcessor.from_pretrained(str(local_model_path))
        model = AutoModelForCausalLM.from_pretrained(str(local_model_path))
    else:
        print("กำลังดาวน์โหลดโมเดลจาก HuggingFace... (อาจใช้เวลาสักครู่)")
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # บันทึกโมเดลและ processor ลงใน local
        print(f"กำลังบันทึกโมเดลลงใน local: {local_model_path}")
        processor.save_pretrained(str(local_model_path))
        model.save_pretrained(str(local_model_path))
    
    print("โหลดโมเดลเสร็จเรียบร้อย")
    return model, processor

def extract_text_from_image(image_path):
    """
    ดึงข้อความจากรูปภาพโดยใช้ OCR
    
    Args:
        image_path: เส้นทางไปยังไฟล์รูปภาพ
    
    Returns:
        ข้อความที่ดึงได้จากรูปภาพ
    """
    try:
        # โหลดรูปภาพ
        image = Image.open(image_path)
        
        # ตั้งค่าภาษาให้รองรับทั้งไทยและอังกฤษ
        custom_config = r'--oem 3 --psm 6 -l tha+eng'
        
        # ดึงข้อความจากรูปภาพ
        text = pytesseract.image_to_string(image, config=custom_config)
        
        # กรองข้อความว่างและช่องว่างที่ไม่จำเป็น
        text = text.strip()
        
        return text if text else "ไม่พบข้อความในรูปภาพ"
    
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการดึงข้อความ: {str(e)}"

def image_to_text(image_path, model, processor):
    """
    แปลงรูปภาพเป็นข้อความด้วยโมเดลและดึงข้อความจากรูปภาพ
    
    Args:
        image_path: เส้นทางไปยังไฟล์รูปภาพ
        model: โมเดลที่ใช้ในการแปลง
        processor: processor สำหรับเตรียมข้อมูล
    
    Returns:
        tuple: (คำอธิบายรูปภาพ, ข้อความที่ดึงได้จากรูปภาพ)
    """
    try:
        # โหลดรูปภาพ
        print(f"กำลังโหลดรูปภาพจาก {image_path}...")
        image = Image.open(image_path)
        
        # ทางเลือก 1: แสดงรูปภาพแบบไม่บล็อกการทำงาน
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.title("รูปภาพที่ต้องการแปลงเป็นข้อความ")
        plt.draw()  # วาดรูปภาพ
        plt.pause(0.001)  # แสดงรูปโดยไม่บล็อกการทำงาน
        
        # ดึงข้อความจากรูปภาพ
        print("กำลังดึงข้อความจากรูปภาพ...")
        extracted_text = extract_text_from_image(image_path)
        
        # แปลงรูปภาพเป็นรูปแบบที่โมเดลต้องการ
        print("กำลังประมวลผลรูปภาพ...")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        
        # สร้างคำอธิบายรูปภาพ
        print("กำลังสร้างคำอธิบายรูปภาพ...")
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values=pixel_values,
                max_length=50,
                num_beams=5,
                early_stopping=True
            )
        
        # แปลงกลับเป็นข้อความ
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # หากยังแสดงรูปภาพอยู่ ให้คงไว้สักครู่แล้วปิด
        plt.close('all')  # ปิดหน้าต่างแสดงรูปภาพทั้งหมด
        
        return generated_caption, extracted_text
    
    except Exception as e:
        plt.close('all')  # ปิดหน้าต่างแสดงรูปภาพในกรณีเกิดข้อผิดพลาด
        return f"เกิดข้อผิดพลาด: {str(e)}", ""

def main():
    parser = argparse.ArgumentParser(description='แปลงรูปภาพเป็นข้อความด้วย AI และดึงข้อความจากรูปภาพ')
    parser.add_argument('--image', type=str, required=True, help='เส้นทางไปยังไฟล์รูปภาพ')
    parser.add_argument('--no-display', action='store_true', help='ไม่แสดงรูปภาพ')
    args = parser.parse_args()
    
    # ตรวจสอบว่าไฟล์ภาพมีอยู่จริงหรือไม่
    if not os.path.exists(args.image):
        print(f"ไม่พบไฟล์ภาพที่ระบุ: {args.image}")
        return
    
    try:
        # โหลดโมเดลและ processor
        model, processor = load_model_and_processor()
        
        # แปลงรูปภาพเป็นข้อความและดึงข้อความจากรูปภาพ
        caption, extracted_text = image_to_text(args.image, model, processor)
        
        print("\nคำอธิบายรูปภาพ:")
        print(caption)
        
        print("\nข้อความที่ดึงได้จากรูปภาพ:")
        print(extracted_text)
        
        # เพิ่มการรอให้ผู้ใช้กดปุ่มก่อนจบโปรแกรม
        if not args.no_display:
            input("\nกด Enter เพื่อจบโปรแกรม...")
    
    except KeyboardInterrupt:
        print("\nโปรแกรมถูกยกเลิกโดยผู้ใช้")
    except Exception as e:
        print(f"\nเกิดข้อผิดพลาด: {str(e)}")

if __name__ == "__main__":
    main()