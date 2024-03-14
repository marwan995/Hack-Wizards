import requests
import numpy as np
# from LSBSteg import encode
from riddle_solvers import riddle_solvers
# import requests
# import numpy as np
# importing the sys module
import sys
import json

# appending the directory of mod.py
# in the sys.path list
sys.path.append('D:/3rd-cmp/HackTrick24/')

# now we can import mod
from LSBSteg import decode, encode

# from riddle_solvers import riddle_solvers

api_base_url = 'http://3.70.97.142:5000'
# api_base_url = 'http://127.0.0.1:5000'
team_id='WtOwYcX'
# team_id = "team123"


def init_fox(team_id):
    init_api = api_base_url+'/fox/start'
    data = {'teamId': team_id}
    # print(init_api)
    # print(data)
    result_message = None
    result_image = None
    json_data = json.dumps(data)
    response = requests.post(init_api, data=json_data, headers={
                             'Content-Type': 'application/json'}, verify=False)
    if response.status_code == 200 or response.status_code == 201:
        # Request was successful
        result = response.json()  # Assuming the response is JSON
        # print(result)
        result_message = result['msg']
        result_image = result['carrier_image']
    # Process the result here
    else:
        print(f"Failed to hit start/fox. Status code: {response.status_code}")

    '''
    In this fucntion you need to hit to the endpoint to start the game as a fox with your team id.
    If a sucessful response is returned, you will recive back the message that you can break into chunkcs
      and the carrier image that you will encode the chunk in it.
    '''
    return result_message, result_image


def generate_message_array(message, image_carrier, total_budget_global):
    fake_messages = ['', '']
    for i in range(2):
        if total_budget_global > 0:
            total_budget_global = total_budget_global-1
            fake_messages[i] = "this is fake message"

    # print(f"generate messages {total_budget_global}")
    image_carrier = np.array(image_carrier)

    image_encoded = encode(image_carrier.copy(), message)

    message_entities = ['E', 'E', 'E']
    # R F F
    if fake_messages[0] != '':
        message_entities[1] = 'F'
    else:
        message_entities[1] = 'E'

    if fake_messages[1] != '':
        message_entities[2] = 'F'
    else:
        message_entities[2] = 'E'

    message_entities[0] = 'R'

    img_fake1 = encode(image_carrier.copy(), fake_messages[0])
    img_fake2 = encode(image_carrier.copy(), fake_messages[1])
    # print(decode(img_fake1))
    # print(decode(img_fake2))
    message_array = [image_encoded, img_fake1, img_fake2]
    send_message(team_id, message_array, message_entities)

    '''
    In this function you will need to create your own startegy. That includes:
        1. How you are going to split the real message into chunkcs
        2. Include any fake chunks
        3. Decide what 3 chuncks you will send in each turn in the 3 channels & what is their entities (F,R,E)
        4. Encode each chunck in the image carrier  
    '''


def get_riddle(team_id, riddle_id):
    get_riddle_api = api_base_url+'/fox/get-riddle'
    data = {'teamId': team_id, 'riddleId': riddle_id}
    json_data = json.dumps(data)
    result_testcase = None
    response = requests.post(get_riddle_api, data=json_data, headers={
                             'Content-Type': 'application/json'}, verify=False)
    if response.status_code == 200 or response.status_code == 201:
        # Request was successful
        result = response.json()  # Assuming the response is JSON
        result_testcase = result['test_case']
        # print(f"result_testcase is {result_testcase}")
    # Process the result here
    else:
        print(f"Failed to hit getriddle. Status code: {response.status_code}")
    '''
    In this function you will hit the api end point that requests the type of riddle you want to solve.
    use the riddle id to request the specific riddle.
    Note that: 
        1. Once you requested a riddle you cannot request it again per game. 
        2. Each riddle has a timeout if you didnot reply with your answer it will be considered as a wrong answer.
        3. You cannot request several riddles at a time, so requesting a new riddle without answering the old one
          will allow you to answer only the new riddle and you will have no access again to the old riddle. 
    '''
    return result_testcase


def solve_riddle(team_id, solution):
    solve_riddle_api = api_base_url+'/fox/solve-riddle'

    data = {'teamId': team_id, 'solution': solution}
    # print(data)
    json_data = json.dumps(data)
    total_budget_global = 0
    response = requests.post(solve_riddle_api, data=json_data, headers={
                             'Content-Type': 'application/json'}, verify=False)
    if response.status_code == 200 or response.status_code == 201:
        # Request was successful
        result = response.json()
        budget_increase = result['budget_increase']
        # print(f"budget_increase{budget_increase}")
        total_budget = result['total_budget']
        total_budget_global = total_budget
        # print(f"total budget{total_budget_global}")
    else:
        print(
            f"Failed to hit solveriddle. Status code: {response.status_code}")

    '''
    In this function you will solve the riddle that you have requested. 
    You will hit the API end point that submits your answer.
    Use te riddle_solvers.py to implement the logic of each riddle.
    '''
    return total_budget_global


def send_message(team_id, messages, message_entities=['F', 'E', 'R']):
    send_message_api = api_base_url+'/fox/send-message'
    messages[0] = messages[0].tolist()
    messages[1] = messages[1].tolist()
    messages[2] = messages[2].tolist()
    data = {'teamId': team_id, 'messages': messages,
            'message_entities': message_entities}
    json_data = json.dumps(data)
    response = requests.post(send_message_api, data=json_data, headers={
                             'Content-Type': 'application/json'}, verify=False)
    if response.status_code == 200 or response.status_code == 201:
        # Request was successful
        result = response.json()

    else:
        print(f"Failed to hit send_msg. Status code: {response.status_code}")
    '''
    Use this function to call the api end point to send one chunk of the message. 
    You will need to send the message (images) in each of the 3 channels along with their entites.
    Refer to the API documentation to know more about what needs to be send in this api call. 
    '''


def end_fox(team_id):
    end_fox_api = api_base_url+'/fox/end-game'
    data = {'teamId': team_id}
    json_data = json.dumps(data)
    response = requests.post(end_fox_api, data=json_data, headers={
                             'Content-Type': 'application/json'}, verify=False)
    if response.status_code == 200 or response.status_code == 201:
        # Request was successful
        result = response
        # print(f"Game ended successfully.{result}")
    else:
        print(f"Failed to hit end_fox. Status code: {response.status_code}")
    '''
    Use this function to call the api end point of ending the fox game.
    Note that:
    1. Not calling this fucntion will cost you in the scoring function
    2. Calling it without sending all the real messages will also affect your scoring fucntion
      (Like failing to submit the entire message within the timelimit of the game).
    '''


def submit_fox_attempt(team_id):
    result_message, result_image = init_fox(team_id)
    total_budget_global = 0
    try:
        testcase_psHard=get_riddle(team_id, 'problem_solving_hard')
        solution_psHard=riddle_solvers['problem_solving_hard'](testcase_psHard)
        total_budget_global = solve_riddle(team_id, solution_psHard)

        testcase_secHard = get_riddle(team_id, 'sec_hard')
        solution_secHard = riddle_solvers['sec_hard'](testcase_secHard)
        total_budget_global = solve_riddle(team_id, solution_secHard)

        testcase_psEasy=get_riddle(team_id, 'problem_solving_easy')
        solution_psEasy=riddle_solvers['problem_solving_easy'](testcase_psEasy)
        total_budget_global = solve_riddle(team_id, solution_psEasy)

        testcase_psMedium=get_riddle(team_id, 'problem_solving_medium')
        solution_psMedium=riddle_solvers['problem_solving_medium'](testcase_psMedium)
        total_budget_global = solve_riddle(team_id, solution_psMedium)


        testcase_mlEasy = get_riddle(team_id, 'ml_easy')
        solution_mlEasy = riddle_solvers['ml_easy'](testcase_mlEasy)
        total_budget_global = solve_riddle(team_id, solution_mlEasy)

        testcase_cvMedium=get_riddle(team_id,'cv_medium')
        solution_cvMedium=riddle_solvers['cv_medium'](testcase_cvMedium)
        total_budget_global=solve_riddle(team_id,solution_cvMedium)

        testcase_mlMedium=get_riddle(team_id,'ml_medium')
        solution_mlMedium=riddle_solvers['ml_medium'](testcase_mlMedium)
        total_budget_global=solve_riddle(team_id,solution_mlMedium)

        
    except Exception as e:
        print ("yamamaaaaaaa" , e)
    print(total_budget_global)

    # solution_mlEasy=riddle_solvers['ml_easy'](testcase_mlEasy)

    # solve_riddle(team_id, solution_mlEasy)
    
    generate_message_array(result_message[0:6], result_image, total_budget_global)
    generate_message_array(result_message[6:12], result_image, total_budget_global-2)
    generate_message_array(result_message[12:20], result_image, total_budget_global-4)
    end_fox(team_id)
    '''
     Call this function to start playing as a fox. 
     You should submit with your own team id that was sent to you in the email.
     Remeber you have up to 15 Submissions as a Fox In phase1.
     In this function you should:
        1. Initialize the game as fox 
        inti_fox(team_id)
        2. Solve riddles
        testcase=get_riddle(team_id, riddle_id)
        testcase=get_riddle(team_id, riddle_id)
        testcase=get_riddle(team_id, riddle_id)
        testcase=get_riddle(team_id, riddle_id)
        testcase=get_riddle(team_id, riddle_id)
        solution=solve_(testcase)
        solve_riddle(team_id, solution) 
        3. Make your own Strategy of sending the messages in the 3 channels
        generate_message_array(message, image_carrier)
        4. Make your own Strategy of splitting the message into chunks
        
        5. Send the messages 
        6. End the Game
    Note that:
        1. You HAVE to start and end the game on your own. The time between the starting and ending the game is taken into the scoring function
        2. You can send in the 3 channels any combination of F(Fake),R(Real),E(Empty) under the conditions that
            2.a. At most one real message is sent
            2.b. You cannot send 3 E(Empty) messages, there should be atleast R(Real)/F(Fake)
        3. Refer To the documentation to know more about the API handling 
    '''


submit_fox_attempt(team_id)
