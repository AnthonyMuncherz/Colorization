import os
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import torch
import matplotlib.pyplot as plt
from colorizers import *
import shutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload and output directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)
os.makedirs('static/uploads', exist_ok=True)

# Initialize colorizer models
colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()

if torch.cuda.is_available():
    colorizer_eccv16.cuda()
    colorizer_siggraph17.cuda()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def process_image(image_path, filename_base):
    # Load and process image
    img = load_img(image_path)
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
    
    if torch.cuda.is_available():
        tens_l_rs = tens_l_rs.cuda()

    # Generate colorized images
    with torch.no_grad():
        out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
        out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())
    
    # Save results
    plt.imsave(f'static/results/{filename_base}_eccv16.png', out_img_eccv16)
    plt.imsave(f'static/results/{filename_base}_siggraph17.png', out_img_siggraph17)
    
    return {
        'original': f'uploads/{filename_base}.jpg',  # Path relative to static folder
        'eccv16': f'results/{filename_base}_eccv16.png',
        'siggraph17': f'results/{filename_base}_siggraph17.png'
    }

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            static_path = os.path.join('static/uploads', filename)
            
            # Save to temporary location first
            file.save(temp_path)
            
            # Copy to static folder for serving
            shutil.copy2(temp_path, static_path)
            
            filename_base = os.path.splitext(filename)[0]
            results = process_image(temp_path, filename_base)
            
            # Clean up temp file
            os.remove(temp_path)
            
            return render_template('index.html', results=results)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000) 