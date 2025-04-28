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

# 2. กำหนดเนื้อหาวิดีโอ - แต่ละฉากพร้อมคำบรรยายในสไตล์อนิเมะ
scenes = [
    {
        "prompt": "a penguin, a cat and a dog packing luggage together in a messy room with excited expressions",
        "text": "เพนกวิน: เฮ้ย! ทุกคนเตรียมตัวให้พร้อม! วันนี้เราจะไปเที่ยวทะเลกัน! \nแมว: แต่ฉันเอาเสื้อเชฟไปทำไมฟะ? \nหมา: ผมเอาเสื้อซูเปอร์ฮีโร่ไป...เผื่อต้องช่วยใคร!",
        "duration": 6
    },
    {
        "prompt": "the same penguin from scene 1 now standing on a tropical beach sweating with sunglasses sliding down its face",
        "text": "เพนกวิน: นี่...เราต้องมาทะเลร้อนๆ ทำไมนะ!? ผมคิดว่ามันจะเหมือนในโปสเตอร์น่ะ... \n(เสียงแมวกับหมาในระยะไกล): ก็แกเป็นคนเลือกที่เที่ยวเองไง!",
        "duration": 5
    },
    {
        "prompt": "the cat chef trying to barbecue fish but the fish jumps out of the grill and runs toward the ocean",
        "text": "แมว: เดี๋ยวๆ! อย่ากลับไปนะ! นั่นเป็นอาหารเย็นของเรา! \nเพนกวิน: (ยืนอ้าปากค้าง) โอ้...นั่นคือเพื่อนผม...",
        "duration": 5
    },
    {
        "prompt": "the superhero dog stuck in a palm tree trying to reach a coconut while the penguin and cat look up worried",
        "text": "หมา: ไม่ต้องห่วง! ผมเป็นซูเปอร์ฮีโร่...แค่ติดต้นไม้เฉยๆ! \nแมว: นั่นมันกะลาส้มไม่ใช่กะโหลกศีรษะนะโว้ย! \nเพนกวิน: ใครไปเรียกเจ้าหน้าที่ชายหาดที...",
        "duration": 6
    }
]

# 3. สร้างภาพด้วย Stable Diffusion สำหรับสไตล์อนิเมะ
def generate_images():
    print("กำลังสร้างภาพสไตล์อนิเมะ...")
    
    # ตรวจสอบว่ามีโมเดลที่บันทึกไว้แล้วหรือไม่
    # เลือกใช้โมเดลที่เหมาะกับการสร้างภาพอนิเมะ
    local_model_path = "./model_cache/TheRafal/everything-v2"
    
    if os.path.exists(local_model_path) and len(os.listdir(local_model_path)) > 0:
        print(f"ใช้โมเดลที่บันทึกไว้จาก {local_model_path}")
        pipe = StableDiffusionPipeline.from_pretrained(local_model_path, local_files_only=True)
    else:
        print("กำลังดาวน์โหลดโมเดล Stable Diffusion...")
        # โมเดลที่เหมาะกับการสร้างรูปแบบอนิเมะ
        model_id = "Linaqruf/anything-v3.0"  # โมเดลที่เหมาะกับการสร้างภาพอนิเมะ (open access)
        
        # หากโมเดลข้างต้นไม่สามารถเข้าถึงได้ ให้ใช้โมเดลนี้แทน
        # model_id = "CompVis/stable-diffusion-v1-4"  
        
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
        # กำหนดชื่อไฟล์ด้วย anime style
        image_path = f"images/anime_scene_{i+1}.png"
        
        # ตรวจสอบว่าต้องการสร้างภาพใหม่หรือไม่
        create_new = True
        if os.path.exists(image_path):
            print(f"พบภาพเดิมที่ {image_path}")
            choice = input("ต้องการสร้างภาพใหม่หรือไม่? (y/n): ")
            create_new = choice.lower() == 'y'
            
        if create_new:
            print(f"กำลังสร้างภาพอนิเมะที่ {i+1}/{len(scenes)}")
            
            # เพิ่ม negative prompt เพื่อหลีกเลี่ยงภาพที่ไม่สวยงาม
            negative_prompt = "low quality, worst quality, blurry, distorted, deformed, disfigured, bad anatomy, poorly drawn, bad proportions"
            
            # ใช้การตั้งค่าที่เหมาะสมสำหรับการสร้างภาพสไตล์อนิเมะที่มีคุณภาพสูง
            image = pipe(
                prompt=scene["prompt"],
                negative_prompt=negative_prompt,
                num_inference_steps=50,  # เพิ่มจำนวน steps เพื่อคุณภาพที่ดีขึ้น
                guidance_scale=7.5,      # ปรับการคุมทิศทางให้เหมาะสม
                width=512,
                height=512
            ).images[0]
            
            # บันทึกภาพ
            image.save(image_path)
        
        scene["image_path"] = image_path
    
    print("สร้างภาพสไตล์อนิเมะเสร็จสิ้น!")

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
    output_path = "output/anime_video.mp4"
    
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
        
        # เพิ่มข้อความสไตล์อนิเมะ (ถ้าต้องการ)
        # txt_clip = TextClip(scene["text"], fontsize=30, color='white', font='Arial-Bold', 
        #                    bg_color='rgba(0,0,0,0.5)', size=img_clip.size)
        # txt_clip = txt_clip.set_position(('center', 'bottom')).set_duration(scene["duration"])
        # video_with_text = CompositeVideoClip([img_clip, txt_clip])
        # video_with_audio = video_with_text.set_audio(audio_clip)
        
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
        print("เสร็จสิ้นการสร้างวิดีโอสไตล์อนิเมะ! ไฟล์อยู่ที่ output/anime_video.mp4")
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")
        import traceback
        traceback.print_exc()  # แสดงข้อผิดพลาดโดยละเอียด

if __name__ == "__main__":
    main()