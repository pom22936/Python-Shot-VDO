import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
from moviepy.editor import VideoFileClip, ImageClip, AudioFileClip, concatenate_videoclips, CompositeAudioClip
from transformers import pipeline, AutoProcessor, AutoModelForTextToWaveform
import soundfile as sf
import time
import librosa
from gtts import gTTS

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
        "prompt": "beautiful sunrise over mountain landscape, cinematic lighting",
        "text": "ธรรมชาติอันงดงามยามเช้า ท่ามกลางขุนเขาสูงตระหง่าน",
        "duration": 5  # วินาที
    },
    {
        "prompt": "modern city skyline at night with colorful lights",
        "text": "หรือจะเป็นเมืองที่เต็มไปด้วยแสงสีในยามค่ำคืน",
        "duration": 5
    },
    {
        "prompt": "beautiful beach with clear water and palm trees",
        "text": "หรือหาดทรายขาวละเอียด น้ำทะเลใสราวกับคริสตัล",
        "duration": 5
    }
]

# 3. สร้างภาพด้วย Stable Diffusion
def generate_images():
    print("กำลังสร้างภาพ...")
    
    # ตรวจสอบว่ามีโมเดลที่บันทึกไว้แล้วหรือไม่
    local_model_path = "./model_cache/stable-diffusion"
    
    if os.path.exists(local_model_path):
        print(f"ใช้โมเดลที่บันทึกไว้จาก {local_model_path}")
        pipe = StableDiffusionPipeline.from_pretrained(local_model_path, local_files_only=True)
    else:
        print("กำลังดาวน์โหลดโมเดล Stable Diffusion...")
        # ใช้โมเดล open access แทน runwayml/stable-diffusion-v1-5
        model_id = "CompVis/stable-diffusion-v1-4"  # โมเดลที่เป็น open access
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
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

# 4. สร้างเสียงพูดด้วย Text-to-Speech
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
        return
    
    video_clips = []
    
    for i, scene in enumerate(scenes):
        # สร้าง clip จากภาพ
        img_clip = ImageClip(scene["image_path"]).set_duration(scene["duration"])
        
        # เพิ่มเสียงพูด
        audio_clip = AudioFileClip(scene["audio_path"])
        img_clip = img_clip.set_audio(audio_clip)
        
        # เพิ่ม transition (crossfade) ระหว่างฉาก
        if i > 0:  # เริ่มจากฉากที่ 2 เป็นต้นไป
            img_clip = img_clip.crossfadein(1.0)
        
        video_clips.append(img_clip)
    
    # รวม clip ทั้งหมด
    final_clip = concatenate_videoclips(video_clips, method="compose")
    
    # บันทึกวิดีโอ
    final_clip.write_videofile(output_path, fps=24)
    
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

if __name__ == "__main__":
    main()