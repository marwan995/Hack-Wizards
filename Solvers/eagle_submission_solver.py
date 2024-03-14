import numpy as np
# from LSBSteg import decode
import requests
import json
import joblib
import pickle
import librosa
import sys
sys.path.append('D:/3rd-cmp/HackTrick24/')        
 
# now we can import mod
from LSBSteg  import decode

loaded_model = None
with open('./model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

api_base_url = 'http://3.70.97.142:5000'
# api_base_url='http://127.0.0.1:5000' #sara's
team_id='WtOwYcX'
# team_id='hamada'


def init_eagle(team_id):
    init_eagle_api=api_base_url+'/eagle/start'
    data={'teamId':team_id}
    json_data = json.dumps(data)
    footprint=None
    response = requests.post(init_eagle_api, data=json_data,headers={'Content-Type': 'application/json'}, verify=False)
    if response.status_code == 200 or response.status_code == 201:
        result = response.json()
        footprint = result['footprint']
    else:
        print(f"Failed to hit endpoint in init eagle. Status code: {response.status_code}")
    '''
    In this fucntion you need to hit to the endpoint to start the game as an eagle with your team id.
    If a sucessful response is returned, you will recive back the first footprints.
    '''
    return footprint

def select_channel(footprint):
    '''
    According to the footprint you recieved (one footprint per channel)
    you need to decide if you want to listen to any of the 3 channels or just skip this message.
    Your goal is to try to catch all the real messages and skip the fake and the empty ones.
    Refer to the documentation of the Footprints to know more what the footprints represent to guide you in your approach.        
    '''
    is_real = 0
    # loaded_model=joblib.load('./Solvers/model.pkl')
    # for i in range(3):
    max_real = [0,0] 
    footprint['1']=np.array(footprint['1'])
    footprint['2']=np.array(footprint['2'])
    footprint['3']=np.array(footprint['3'])

    x1 = (footprint['1']*255).astype(np.uint8)
    x2 = (footprint['2']*255).astype(np.uint8)
    x3 = (footprint['3']*255).astype(np.uint8)

    spectral_flatness1 = librosa.feature.spectral_flatness(S=x1)[0].reshape(1,-1)[0]
    spectral_flatness2 = librosa.feature.spectral_flatness(S=x2)[0].reshape(1,-1)[0]
    spectral_flatness3 = librosa.feature.spectral_flatness(S=x3)[0].reshape(1,-1)[0]
    spectral_rolloff1 = librosa.feature.spectral_rolloff(S=x1)[0].reshape(1,-1)[0]
    spectral_rolloff2 = librosa.feature.spectral_rolloff(S=x2)[0].reshape(1,-1)[0]
    spectral_rolloff3 = librosa.feature.spectral_rolloff(S=x3)[0].reshape(1,-1)[0]
    # print(len( np.concatenate((spectral_flatness,spectral_rolloff))))
    # label1 = loaded_model.predict([ np.concatenate((spectral_flatness1,spectral_rolloff1))])[0]
    # label2 = loaded_model.predict([ np.concatenate((spectral_flatness2,spectral_rolloff2))])[0]
    # label3 = loaded_model.predict([ np.concatenate((spectral_flatness3,spectral_rolloff3))])[0]
    prob1=loaded_model.predict_proba([ np.concatenate((spectral_flatness1,spectral_rolloff1))])
    prob2=loaded_model.predict_proba([ np.concatenate((spectral_flatness2,spectral_rolloff2))])
    prob3=loaded_model.predict_proba([ np.concatenate((spectral_flatness3,spectral_rolloff3))])
    # print(prob1,prob2,prob3)
    label1_index = np.argmax(prob1)  
    label1 = 'real' if label1_index == 1  else 'fake'

    label2_index = np.argmax(prob2)  
    label2 = 'real' if label2_index == 1  else 'fake'

    label3_index = np.argmax(prob3)  
    label3 = 'real' if label3_index == 1  else 'fake'

    if prob1[0][0]==prob2[0][0]==prob3[0][0] and prob1[0][1]==prob2[0][1]==prob3[0][1]:
        return 0
    elif prob1[0][0]==prob2[0][0] and prob1[0][1]==prob2[0][1] :
        label1='fake'
        label2='fake'
    elif prob2[0][0]==prob3[0][0] and prob2[0][1]==prob3[0][1]:
        label2='fake'
        label3='fake'
    elif prob1[0][0]==prob3[0][0] and prob1[0][1]==prob3[0][1]:
        label1='fake'
        label3='fake'
    max_real = [prob1[0][1],1] if label1 =='real' and max_real[0] <= prob1[0][1] else max_real
    max_real = [prob2[0][1],2] if label2 =='real' and max_real[0] <= prob2[0][1] else max_real
    max_real = [prob3[0][1],3] if label3 =='real' and max_real[0] <= prob3[0][1] else max_real
    # print(label1,label2,label3)
    # print(max_real[1])

    return max_real[1]       

def skip_msg(team_id):
    '''
    If you decide to NOT listen to ANY of the 3 channels then you need to hit the end point skipping the message.
    If sucessful request to the end point , you will expect to have back new footprints IF ANY.
    '''
    skip_msg_api=api_base_url+'/eagle/skip-message'
    data={'teamId':team_id}
    json_data = json.dumps(data)
    next_footprint=None
    response = requests.post(skip_msg_api, data=json_data,headers={'Content-Type': 'application/json'}, verify=False)
    # print("skip res",response.text,response.content)
    if response.status_code == 200 or response.status_code == 201:
        try:
            # if 'application/json' in content_type:
            result = response.json()
            next_footprint = result['nextFootprint']
            return next_footprint
        except Exception as e:
            print(f"ba3att textttttttt: {e}")
            # elif 'text/plain' in content_type:
            result=response.text
            return result
    else:
        print(f"Failed to hit endpoint in skip msg. Status code: {response.status_code}")
        return "failed"
  
def request_msg(team_id, channel_id):
    '''
    If you decide to listen to any of the 3 channels then you need to hit the end point of selecting a channel to hear on (1,2 or 3)
    '''
    request_msg_api=api_base_url+'/eagle/request-message'
    data={'teamId':team_id, 'channelId':channel_id}
    json_data = json.dumps(data)
    message=None
    response = requests.post(request_msg_api, data=json_data,headers={'Content-Type': 'application/json'}, verify=False)
    # print("request msg res",response.text)
    if response.status_code == 200 or response.status_code == 201:
        result = response.json()
        message = result['encodedMsg']
    else:
        print(f"Failed to hit endpoint in request msg. Status code: {response.status_code}")
    return message
    

def submit_msg(team_id, decoded_msg):
    submit_msg_api=api_base_url+'/eagle/submit-message'
    data={'teamId':team_id,'decodedMsg':decoded_msg}
    json_data=json.dumps(data)
    response=requests.post(submit_msg_api,data=json_data,headers={'Content-Type': 'application/json'}, verify=False)
    # print("submit msg res",response.text)
    if response.status_code == 200 or response.status_code == 201:
        try:
        # if 'application/json' in content_type:
            result = response.json()
            next_footprint =result['nextFootprint']
            return next_footprint
        # elif 'text/plain' in content_type:
        except Exception as e:
            print(f"ba3att textttttttt: {e}")
            result=response.text
            return result
    else:
        print(f"Failed to hit endpoint in submit msg. Status code: {response.status_code}")
        return ""
    '''
    In this function you are expected to:
        1. Decode the message you requested previously
        2. call the api end point to send your decoded message  
    If sucessful request to the end point , you will expect to have back new footprints IF ANY.
    '''
    #the returned may be the next footprint or that the end of msg has been reached
    
    
    
  
def end_eagle(team_id):
    end_eagle_api=api_base_url+'/eagle/end-game'
    data={'teamId':team_id}
    json_data=json.dumps(data)
    response=requests.post(end_eagle_api,data=json_data,headers={'Content-Type': 'application/json'}, verify=False)
    print("end res",response.text)
    if response.status_code == 200 or response.status_code == 201:
        return_text = response.text
        print(return_text)
    else:
        print(f"Failed to hit endpoint in end eagle. Status code: {response.status_code}")
    
    '''
    Use this function to call the api end point of ending the eagle  game.
    Note that:
    1. Not calling this fucntion will cost you in the scoring function
    '''
    

def submit_eagle_attempt(team_id):
    footprint=init_eagle(team_id)
    counter = 0
    all_msg=""
    # print(type(footprint))
    while not isinstance(footprint, str):
        # try: 
            print(counter)
            channelID=select_channel(footprint)
            if channelID != 0:
                encoded_Msg=request_msg(team_id,channelID)
                encoded_Msg=np.array(encoded_Msg)
                all_msg+=decode(encoded_Msg)
                footprint=submit_msg(team_id,all_msg)
            else:
                footprint=skip_msg(team_id)
            counter = counter +1
        # except Exception as e:
            # print("ya5rabyyyyyy T^T : ",e)
            # break
    print(all_msg)
    end_eagle(team_id)
        
              
    
    '''
     Call this function to start playing as an eagle. 
     You should submit with your own team id that was sent to you in the email.
     Remeber you have up to 15 Submissions as an Eagle In phase1.
     In this function you should:
        1. Initialize the game as fox 
        2. Solve the footprints to know which channel to listen on if any.
        3. Select a channel to hear on OR send skip request.
        4. Submit your answer in case you listened on any channel
        5. End the Game
    '''
    


submit_eagle_attempt(team_id)

# fake = np.load('./fake.npz')
# real = np.load('./real.npz')
# x_array_fake = fake['x']
# x_array_real = real['x']
 
# print(select_channel({'1':x_array_real[200],'2':x_array_fake[100],'3':x_array_fake[20]}))