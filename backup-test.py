import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
import moviepy.editor as mp
from moviepy.editor import *
from gtts import gTTS
from pydub import AudioSegment
import time
import librosa

# กำหนดโฟลเดอร์สำหรับเก็บโมเดลที่ดาวน์โหลด
os.environ["TRANSFORMERS_CACHE"] = "./model_cache"
os.environ["HF_HOME"] = "./model_cache"

# 1. สร้างไดเร็กทอรีสำหรับเก็บไฟล์
os.makedirs("images", exist_ok=True)
os.makedirs("audio", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs("model_cache", exist_ok=True)

# 2. กำหนดเนื้อหาวิดีโอ - แต่ละฉากพร้อมคำบรรยาย
scenes = [
  {
    "prompt": "overworked office worker staring at a screen full of unread emails, gray tone, dramatic lighting",
    "text": "คุณเสียเวลากับการตอบอีเมลซ้ำ ๆ ทุกวันใช่ไหม?",
    "duration": 5
  },
  {
    "prompt": "workflow automation dashboard with colorful n8n nodes connecting, modern UI",
    "text": "แค่ส่งอีเมล ระบบจะเข้าใจและจัดการให้แบบอัตโนมัติ",
    "duration": 5
  },
  {
    "prompt": "split screen of email request, Jira ticket creation, and Jenkins pipeline running",
    "text": "ขอเปิดสิทธิ์? → สร้าง Ticket + Run Jenkins ทันที",
    "duration": 5
  },
  {
    "prompt": "automated report system generating and replying to email with attachment",
    "text": "แค่ส่งอีเมลขอรายงาน ระบบก็จัดให้!",
    "duration": 5
  },
  {
    "prompt": "clean tech-style promo screen with logo and call to action, futuristic design",
    "text": "ให้ระบบช่วยจัดการคำขอในอีเมลของคุณ — อัตโนมัติ. แม่นยำ. ปรับแต่งได้",
    "duration": 5
  }
]

# 3. สร้างภาพด้วย Stable Diffusion
def generate_images():
    print("กำลังสร้างภาพ...")
    
    # ตรวจสอบว่ามีโมเดลที่บันทึกไว้แล้วหรือไม่
    local_model_path = "./model_cache/stable-diffusion"
    
    if os.path.exists(local_model_path) and len(os.listdir(local_model_path)) > 0:
        print(f"ใช้โมเดลที่บันทึกไว้จาก {local_model_path}")
        pipe = StableDiffusionPipeline.from_pretrained(local_model_path, local_files_only=True)
    else:
        print("กำลังดาวน์โหลดโมเดล Stable Diffusion...")
        # ใช้โมเดล open access แทน runwayml/stable-diffusion-v1-5
        model_id = "CompVis/stable-diffusion-v1-4"  # โมเดลที่เป็น open access
        pipe = StableDiffusionPipeline.from_pretrained(model_id)
        # บันทึกโมเดลไว้ใช้ในครั้งต่อไป
        pipe.save_pretrained(local_model_path)
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        # ใช้ CPU โดยปรับใช้ float32 แทน float16
        pipe = pipe.to("cpu")
        pipe.safety_checker = None  # ปิด safety checker เพื่อประหยัดหน่วยความจำ
    
    # สร้างภาพตามคำอธิบาย (prompt) แต่ละฉาก
    for i, scene in enumerate(scenes):
        # ตรวจสอบว่ามีภาพที่สร้างไว้แล้วหรือไม่
        image_path = f"images/scene_{i+1}.png"
        if os.path.exists(image_path):
            print(f"ใช้ภาพที่มีอยู่แล้วที่ {image_path}")
        else:
            print(f"กำลังสร้างภาพที่ {i+1}/{len(scenes)}")
            image = pipe(scene["prompt"], num_inference_steps=30).images[0]
            image.save(image_path)
        
        scene["image_path"] = image_path
    
    print("สร้างภาพเสร็จสิ้น!")

# 4. สร้างเสียงพูดด้วย Google Text-to-Speech
def generate_speech():
    print("กำลังสร้างเสียงพูด...")
    
    # สร้างเสียงพูดสำหรับแต่ละฉาก
    for i, scene in enumerate(scenes):
        audio_path = f"audio/speech_{i+1}.mp3"
        
        # ตรวจสอบว่ามีไฟล์เสียงที่สร้างไว้แล้วหรือไม่
        if os.path.exists(audio_path):
            print(f"ใช้ไฟล์เสียงที่มีอยู่แล้วที่ {audio_path}")
            
            # ถ้ามีไฟล์อยู่แล้ว อ่านความยาวของเสียง
            try:
                audio_duration = librosa.get_duration(path=audio_path)
                print(f"  ความยาวของเสียง: {audio_duration:.2f} วินาที")
                
                # ปรับความยาวของฉากให้เท่ากับความยาวของเสียง + 1 วินาทีเพื่อความเรียบร้อย
                scene["duration"] = audio_duration + 1
            except Exception as e:
                print(f"  ไม่สามารถอ่านความยาวของเสียงได้: {e}")
        else:
            print(f"กำลังสร้างเสียงพูดที่ {i+1}/{len(scenes)}")
            
            # ทำให้แน่ใจว่าไม่มีการทำงานซ้อนกัน
            temp_audio_path = f"audio/temp_speech_{i+1}_{int(time.time())}.mp3"
            
            # สร้างเสียงพูดด้วย Google TTS (รองรับภาษาไทย)
            tts = gTTS(text=scene["text"], lang='th', slow=False)
            tts.save(temp_audio_path)
            
            # ตรวจสอบความยาวของเสียง
            audio_duration = librosa.get_duration(path=temp_audio_path)
            print(f"  ความยาวของเสียง: {audio_duration:.2f} วินาที")
            
            # ปรับความยาวของฉากให้เท่ากับความยาวของเสียง + 1 วินาทีเพื่อความเรียบร้อย
            scene["duration"] = audio_duration + 1
            
            # เปลี่ยนชื่อไฟล์จากชั่วคราวเป็นชื่อจริง
            os.rename(temp_audio_path, audio_path)
        
        scene["audio_path"] = audio_path
    
    print("สร้างเสียงพูดเสร็จสิ้น!")

# 5. สร้างวิดีโอจากภาพและเสียง
def create_video():
    print("กำลังสร้างวิดีโอ...")
    output_path = "output/final_video.mp4"
    
    # ตรวจสอบว่ามีไฟล์วิดีโอที่สร้างไว้แล้วหรือไม่
    if os.path.exists(output_path):
        print(f"ไฟล์วิดีโอมีอยู่แล้วที่ {output_path}")
        choice = input("ต้องการสร้างวิดีโอใหม่หรือไม่? (y/n): ")
        if choice.lower() != 'y':
            return
        # ลบไฟล์เดิม
        os.remove(output_path)
    
    video_clips = []
    
    for i, scene in enumerate(scenes):
        print(f"กำลังสร้างคลิปที่ {i+1}/{len(scenes)}")
        
        # สร้าง clip จากภาพ กำหนดให้ความยาวเท่ากับความยาวของเสียง
        img_clip = ImageClip(scene["image_path"]).set_duration(scene["duration"])
        
        # เพิ่มเสียงพูด
        audio_clip = AudioFileClip(scene["audio_path"])
        video_with_audio = img_clip.set_audio(audio_clip)
        
        # เพิ่มข้อความ (ถ้าต้องการ)
        # txt_clip = TextClip(scene["text"], fontsize=30, color='white', bg_color='black', size=img_clip.size)
        # txt_clip = txt_clip.set_position(('center', 'bottom')).set_duration(scene["duration"])
        # video_with_text = CompositeVideoClip([img_clip, txt_clip])
        
        video_clips.append(video_with_audio)
    
    # รวม clip ทั้งหมดโดยใช้การเฟดระหว่างคลิป
    final_clip = concatenate_videoclips(video_clips, method="compose") 
    
    # บันทึกวิดีโอ
    print("กำลังบันทึกวิดีโอ...")
    final_clip.write_videofile(output_path, fps=24, audio_codec="aac")
    
    print(f"สร้างวิดีโอเสร็จสิ้น! ไฟล์อยู่ที่ {output_path}")

# 6. รันฟังก์ชันทั้งหมด
def main():
    try:
        generate_images()
        generate_speech()
        create_video()
        print("เสร็จสิ้นการสร้างวิดีโอ! ไฟล์อยู่ที่ output/final_video.mp4")
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")
        import traceback
        traceback.print_exc()  # แสดงข้อผิดพลาดโดยละเอียด

if __name__ == "__main__":
    main()