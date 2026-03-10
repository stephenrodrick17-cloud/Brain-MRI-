import time
import threading
import requests
import os
from deploy_inference import create_app

# Create app using local model
app = create_app(model_path='best_model.pth', device='cpu', threshold=0.5)

# Use werkzeug make_server to run in background thread
from werkzeug.serving import make_server
server = make_server('127.0.0.1', 5000, app)
thread = threading.Thread(target=server.serve_forever)
thread.daemon = True
thread.start()
print('Server started in background, sleeping 1s')

# Wait a bit for server to be ready
time.sleep(1)

# Prepare test file
test_path = os.path.join(os.path.dirname(__file__), 'predictions_visualization.png')
if not os.path.exists(test_path):
    print('Test file not found:', test_path)
else:
    try:
        with open(test_path, 'rb') as f:
            files = {'file': f}
            r = requests.post('http://127.0.0.1:5000/predict', files=files, timeout=120)
            print('HTTP', r.status_code)
            print(r.text[:4000])
    except Exception as e:
        print('Request error:', e)

# Shutdown server
server.shutdown()
thread.join()
print('Server shutdown')
