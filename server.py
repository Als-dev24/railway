import base64
import io
import os
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from gradio_client import Client, handle_file

app = Flask(__name__)
CORS(app)

print('⏳ الاتصال بـ IDM-VTON...')
client = Client("yisol/IDM-VTON")
print('✅ جاهز!')

def b64_to_file(b64: str, suffix='.jpg') -> str:
    if ',' in b64:
        b64 = b64.split(',')[1]
    data = base64.b64decode(b64)
    img  = Image.open(io.BytesIO(data)).convert('RGB')
    max_size = 768
    if max(img.size) > max_size:
        ratio    = max_size / max(img.size)
        img      = img.resize((int(img.width*ratio), int(img.height*ratio)), Image.LANCZOS)
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    img.save(tmp.name, 'JPEG', quality=95)
    return tmp.name

def file_to_b64(path: str) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

@app.route('/')
def index():
    return jsonify({'status': 'ok', 'service': 'Fashn AI'})

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': 'IDM-VTON'})

@app.route('/tryon', methods=['POST'])
def tryon():
    try:
        data       = request.json
        person_b64 = data.get('person')
        cloth_b64  = data.get('cloth')

        if not person_b64 or not cloth_b64:
            return jsonify({'error': 'صور ناقصة'}), 400

        person_file = b64_to_file(person_b64, '.jpg')
        cloth_file  = b64_to_file(cloth_b64,  '.jpg')

        result = client.predict(
            dict={
                'background': handle_file(person_file),
                'layers':     [],
                'composite':  None,
            },
            garm_img=handle_file(cloth_file),
            garment_des='upper body garment',
            is_checked=True,
            is_checked_crop=False,
            denoise_steps=30,
            seed=42,
            api_name='/tryon'
        )

        result_path = result[0] if isinstance(result, tuple) else result
        result_b64  = file_to_b64(result_path)

        for f in [person_file, cloth_file]:
            try: os.unlink(f)
            except: pass

        return jsonify({'success': True, 'image': result_b64})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
