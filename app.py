from flask import Flask, render_template, request, jsonify
import subprocess
import psutil
import os


app = Flask(__name__)

tryon_process = None

# Sample recommendations data
# Define recommendations for each item
# Define recommendations for each item
recommendations = {
    'sunglasses1.png': ['sunglasses2.png', 'sunglasses3.png', 'sunglasses4.png'],
    'sunglasses2.png': ['sunglasses1.png', 'sunglasses3.png', 'sunglasses4.png','sunglasses5.png'],
    'sunglasses3.png': ['sunglasses1.png', 'sunglasses2.png', 'sunglasses4.png','sunglasses5.png'],
    'Watch1.png': ['Watch2.png', 'Watch3.png', 'Watch4.png'],
    'Watch2.png': ['Watch1.png', 'Watch3.png', 'Watch4.png'],
    'Watch3.png': ['Watch1.png', 'Watch2.png', 'Watch4.png'],
    'Neck1.png': ['Neck2.png', 'Neck3.png', 'Neck4.png'],
    'Neck2.png': ['Neck1.png', 'Neck3.png', 'Neck4.png'],
    'Neck3.png': ['Neck1.png', 'Neck2.png', 'Neck4.png'],
}



@app.route('/')
def index():
    return render_template('index.html')
@app.route('/try-now', methods=['GET'])
def try_now():
    global tryon_process
    item_type = request.args.get('type', 'sunglasses')
    item_image = request.args.get('item', 'sunglasses1.png')

    # Close any existing try-on process
    if tryon_process and tryon_process.poll() is None:
        for child in psutil.Process(tryon_process.pid).children(recursive=True):
            child.terminate()
        tryon_process.terminate()

    # Execute try-on logic
    script_path = os.path.join('tryon_modules', f'{item_type}_tryon.py')
    image_path = os.path.join('static', item_image)

    if os.path.exists(script_path):
        tryon_process = subprocess.Popen(['python', script_path, image_path])
        return jsonify({
            "success": True,
            "message": f"Virtual try-on initiated for {item_image}",
            "recommendations": recommendations.get(item_image, [])
        })
    else:
        return jsonify({"success": False, "message": f"Try-on script for {item_type} not found."})

@app.route('/start-tryon', methods=['POST'])
def start_tryon():
    global tryon_process
    item_type = request.json.get('type')
    item_image = request.json.get('item')
    
    # Close any existing try-on process
    if tryon_process and tryon_process.poll() is None:
        for child in psutil.Process(tryon_process.pid).children(recursive=True):
            child.terminate()
        tryon_process.terminate()
    
    # Execute the try-on logic for the selected item
    script_path = os.path.join('tryon_modules', f'{item_type}_tryon.py')
    image_path = os.path.join('static', item_image)
    
    if os.path.exists(script_path):
        tryon_process = subprocess.Popen(['python', script_path, image_path])
        return jsonify({"success": True, "message": f"Virtual try-on initiated for {item_image}"})
    else:
        return jsonify({"success": False, "message": f"Try-on script for {item_type} not found."})


@app.route('/close-tryon')
def close_tryon():
    global tryon_process
    # Close the try-on session
    if tryon_process and tryon_process.poll() is None:
        for child in psutil.Process(tryon_process.pid).children(recursive=True):
            child.terminate()
        tryon_process.terminate()
    
    return render_template('index.html')

@app.route('/recommendations/<item_image>', methods=['GET'])
def get_recommendations(item_image):
    related_products = recommendations.get(item_image, [])
    return jsonify(related_products)

if __name__ == '__main__':
    app.run(debug=True)
