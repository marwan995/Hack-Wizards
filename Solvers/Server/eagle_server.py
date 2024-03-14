from flask import Flask ,request, jsonify ,abort
import numpy as np
import json
import cv2
import sys        
sys.path.append('D:/3rd-cmp/HackTrick24/')        

from LSBSteg  import encode

app = Flask(__name__)

fake = np.load('D:/3rd-cmp/HackTrick24/Solvers/fake.npz')
real = np.load('D:/3rd-cmp/HackTrick24/Solvers/real.npz')
x_array_fake = fake['x']
x_array_real = real['x']
counter=0

# 3.1 Start Game 
@app.route("/eagle/start", methods=['POST'])
def start_game():
    data = request.get_json()
    if not isinstance(data.get('teamId'),str):
        abort(400, description='Bad Request: teamId must be a string')
    footprint = {'1':  x_array_fake[90].tolist(), '2':x_array_real[12].tolist(), '3':x_array_fake[100].tolist()}
    # footprint=list(footprint)
    print("I'm in start game")
    return jsonify({'footprint':footprint})

# 3.2 Request Message
@app.route("/eagle/request-message", methods=['POST'])
def request_message():
    data = request.get_json()
    if not isinstance(data.get('teamId'),str):
        print("teamId error")
        abort(400, description='Bad Request: teamId must be a string')
    if not isinstance(data.get('channelId'),int):
        print("channelId error")
        abort(400, description='Bad Request: channelId must be a integer')
    img=cv2.imread(r'D:\3rd-cmp\HackTrick24\Riddles\cv_easy_example\actual.jpg')
    encodedMsg=encode(img,"harf harf harf harf1")
    encodedMsg=encodedMsg.tolist()
    print("I'm in request msg")
    return jsonify({'encodedMsg':encodedMsg})

# 3.3  Skip Message
@app.route("/eagle/skip-message", methods=['POST'])
def skip_message():
    data = request.get_json()
    if not isinstance(data.get('teamId'),str):
        abort(400, description='Bad Request: teamId must be a string')

    # Note : skip_message can also return a string
    # Note : spectogram1, spectrogram2, spectrogram3 are numpy arrays
    # nextFootprint = {'1': x_array_fake[123], '2':x_array_real[123], '3':x_array_fake[111] }
    print("I'm in skip msg")
    # return jsonify({'nextFootprint':nextFootprint})
    return "End of message reached"


# 3.4  Submit Message
@app.route("/eagle/submit-message", methods=['POST'])
def submit_message():
    global counter
    counter+=1
    data = request.get_json()
    if not isinstance(data.get('teamId'),str):
        abort(400, description='Bad Request: teamId must be a string')
    if not isinstance(data.get('decodedMsg'),str):
        abort(400, description='Bad Request: decodedMsg must be a string')
    
    # Note : submit_message can also return a string
    # Note : spectogram1, spectrogram2, spectrogram3 are numpy arrays
    nextFootprint = {'1':x_array_fake[123].tolist(), '2':x_array_real[213].tolist(), '3':x_array_fake[235].tolist()}
    print("I'm in submit msg")
    if counter==2:
        return "game ended in submit"
    else:
        return jsonify({'nextFootprint':nextFootprint})



# 3.5 End Game
@app.route("/eagle/end-game", methods=['POST'])
def end_game():
    data = request.get_json()
    if not isinstance(data.get('teamId'),str):
        abort(400, description='Bad Request: teamId must be a string')
    return_text = "Game ended successfully with a score of 10. New Highscore reached!"
    print("I'm in end game")
    return return_text


