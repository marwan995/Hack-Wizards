from flask import Flask ,request, jsonify ,abort
import numpy as np
import cv2
import pandas as pd
import sys
sys.path.append('D:/3rd-cmp/HackTrick24/')        


app = Flask(__name__)


# 2.1 Start Game 
@app.route("/fox/start", methods=['POST'])
def start_game():
    data = request.get_json()
    if not isinstance(data.get('teamId'),str):
        abort(400, description='Bad Request: teamId must be a string')

    msg = "Hello, World!"
    img=cv2.imread('D:/3rd-cmp/HackTrick24/Riddles/cv_easy_example/actual.jpg')
    
    carrier_image = np.array(img)
    print(carrier_image)
    return jsonify({'msg':msg,'carrier_image': carrier_image.tolist()})

# 2.2 Get Riddle
@app.route("/fox/get-riddle", methods=['POST'])
def get_riddle():
    data = request.get_json()
    if not isinstance(data.get('teamId'),str):
        abort(400, description='Bad Request: teamId must be a string')
    if not isinstance(data.get('riddleId'),str):
        abort(400, description='Bad Request: riddleId must be a string')
    # test case depend on the riddle 
    # assume it's string (ps-medium)
    # file_path = 'F:\\LockD\\CMP2025\\Hackathones\\HackTrick24\\Riddles\\ml_easy_sample_example\\series_data.csv'
    # df = pd.read_csv(file_path)
    img_cominend=cv2.imread(r'D:/3rd-cmp/HackTrick24/Riddles/cv_medium_example/combined_large_image.png')
    img_patch=cv2.imread(r'D:/3rd-cmp/HackTrick24/Riddles/cv_medium_example/patch_image.png')
    

    
    # test_case = ("266200199BBCDFF1","0123456789ABCDEF")

    return jsonify({'test_case':[img_cominend.tolist(),img_patch.tolist()]})

# 2.3 Solve Riddle
@app.route("/fox/solve-riddle", methods=['POST'])
def solve_riddle():
    data = request.get_json()
    if not isinstance(data.get('teamId'),str):
        abort(400, description='Bad Request: teamId must be a string')
    if not hasattr(data.get('solution'),'__iter__'):
        
        abort(400, description='Bad Request: solution must be array')
    print(data.get('solution'))
    budget_increase = 100
    total_budget = 1000
    status = "success"

    return jsonify({'budget_increase':budget_increase, 'total_budget':total_budget, 'status':status})

# 2.4 Send Message
@app.route("/fox/send-message", methods=['POST'])
def send_message():
    data = request.get_json()
    if not isinstance(data.get('teamId'),str):
        abort(400, description='Bad Request: teamId must be a string')
    if not hasattr(data.get('messages'),'__iter__'):
        abort(400, description='Bad Request: message must be array')
    assert (isinstance(element, np.ndarray) for element in data.get('messages')) 
    if not hasattr(data.get('message_entities'),'__iter__'):
        abort(400, description='Bad Request: message_entities must be array')
    assert (isinstance(element, str) for element in data.get('message_entities'))

    status = "success"

    return jsonify({'status':status})


# 2.5 End Game
@app.route("/fox/end-game", methods=['POST'])
def end_game():
    data = request.get_json()
    if not isinstance(data.get('teamId'),str):
        abort(400, description='Bad Request: teamId must be a string')
    message = "Thanks for playing!"
    return message


