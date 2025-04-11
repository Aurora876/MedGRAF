from flask import Flask, request, jsonify, send_from_directory, render_template
import os
from werkzeug.utils import secure_filename
import shutil
import logging

# 导入函数
from render_xray_vis import generate_image

app = Flask(__name__)

# 配置日志记录器
logging.basicConfig(
    filename='app.log',  # 日志文件名
    level=logging.ERROR,  # 记录 ERROR 级别及以上的日志
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局变量存储进度
progress = {"current": 1, "total": 100}

@app.route("/")
def index():
    return render_template("index.html")

# 配置上传目录
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 配置生成图片的保存目录，并确保其存在
SAVE_DIR = './static/generated_images'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 允许上传的文件类型
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/generate', methods=['POST'])
def generate():
    # 清空生成目录
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
    os.makedirs(SAVE_DIR)

    # 获取上传的图片文件
    if 'file' not in request.files:
        logger.error("No file part in the request.")
        return jsonify({"error": "An error occurred. Please check the logs for details."}), 400
    file = request.files['file']
    if file.filename == '':
        logger.error("No selected file in the request.")
        return jsonify({"error": "An error occurred. Please check the logs for details."}), 400

    # 检查文件类型
    if not allowed_file(file.filename):
        logger.error(f"Invalid file type: {file.filename}")
        return jsonify({"error": "An error occurred. Please check the logs for details."}), 400

    # 保存图片到本地
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # 检查文件是否有效
    if not os.path.exists(file_path):
        logger.error(f"Failed to save uploaded file: {file_path}")
        return jsonify({"error": "An error occurred. Please check the logs for details."}), 500

    # 配置其他参数
    config_file_path = '/home/zd/jzd/medgraf_vis/configs/knee.yaml'
    model_path = '/home/zd/jzd/medgraf_vis/results/ckpt/model.pt'
    img_size = 256

    # 从前端表单获取参数
    total_iterations = request.form.get('total_iterations', type=int, default=1000)
    test_j = request.form.get('test_j', type=int, default=0)
    save_every = request.form.get('save_every', type=int, default=100)  # 动态传入 save_every

    # 调用生成函数
    try:
        def update_progress(current, total):
            """更新进度"""
            progress["current"] = current
            progress["total"] = total

        generate_image(
            config_file_path=config_file_path,
            xray_img_path=file_path,  # 使用上传文件的路径作为输入路径
            save_dir=SAVE_DIR,
            model_path=model_path,
            img_size=img_size,
            save_every=save_every,                     # 动态传入 save_every
            total_iterations=total_iterations,         # 动态传入 total_iterations
            test_j=test_j,                             # 动态传入 test_j
            progress_callback=update_progress          # 进度回调函数
        )
    except Exception as e:
        import traceback
        error_message = f"Error during image generation: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_message)
        return jsonify({"error": "An error occurred. Please check the logs for details."}), 500

    # 返回生成的图片路径
    generated_images = [
        f"/static/generated_images/{f}"  # 返回相对路径，Flask 会自动提供静态文件服务
        for f in os.listdir(SAVE_DIR) if f.endswith('.png')
    ]
    return jsonify({"images": generated_images})

@app.route('/progress', methods=['GET'])
def get_progress():
    """返回当前进度"""
    return jsonify(progress)


if __name__ == '__main__':
    # 禁用 Flask 的默认请求日志
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)  # 或者使用 logging.CRITICAL

    app.run(host='192.168.3.12', port=6006, debug=False)