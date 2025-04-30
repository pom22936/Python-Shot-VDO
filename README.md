# Python-Shot-VDO


# how to install
pip install -r requirements.txt

# window 
Download Tesseract OCR installer for Windows:
Go to https://github.com/UB-Mannheim/tesseract/wiki
Download the latest version (64-bit) installer (e.g., tesseract-ocr-w64-setup-v5.3.3.20231005.exe)
Install Tesseract:
Run the installer
During installation, make sure to:
Check "Additional language data (download)"
Select "Thai" language support
Note the installation directory (default is usually C:\Program Files\Tesseract-OCR)
Add Tesseract to your system PATH:
Open System Properties (Win + Pause/Break)
Click "Advanced system settings"
Click "Environment Variables"
Under "System variables", find and select "Path"
Click "Edit"
Click "New"
Add the Tesseract installation directory (e.g., C:\Program Files\Tesseract-OCR)
Click "OK" on all windows

how to run

- image to text
   python image_to_text.py --image image.png

- create shot vdo
   python main.py