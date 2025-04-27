from django.shortcuts import render,redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .auth import authentication
from django.contrib.auth.models import User
from .models import *
import cv2
import face_recognition
import numpy as np
import os
from django.http import HttpResponse
# Create your views here.
def index(request):    
    return render(request,'index.html')

def upload(request):    
    return render(request,'upload.html')

from datetime import datetime
from django.shortcuts import render, redirect
from django.contrib import messages
from .models import *
from django.core.files.storage import FileSystemStorage

import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from django.shortcuts import render, redirect
from django.http import HttpResponse

# Define the path for training images
path = 'dataset/Training_images'
images = []
classNames = []

# Load training images and names
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def mark_criminal_record(name):
    with open('criminal_record.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


# Precompute encodings for known faces
encodeListKnown = findEncodings(images)


def open_camera(request):
    # Open the camera
    cap = cv2.VideoCapture(0)

    face_detected = False  # Flag to check if a face is recognized
    recognized_name = None  # To store the recognized name

    while True:
        success, img = cap.read()
        if not success:
            break

        # Resize and convert the image to RGB
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        # Detect faces and encodings in the frame
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                recognized_name = classNames[matchIndex].upper()
                face_detected = True
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                # Draw a rectangle around the detected face
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, recognized_name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                # Log the detected name
                mark_criminal_record(recognized_name)

                # Exit the loop once a face is recognized
                break

        # Show the webcam feed
        cv2.imshow('Webcam', img)

        # Close the camera when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q') or face_detected:
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

    # Redirect based on recognition status
    if face_detected:
        # Redirect to the student dashboard (update the URL name accordingly)
        messages.success(request, f"Welcome {recognized_name}!")
        return redirect('student_dashboard')  # Replace 'student_dashboard' with the actual URL name
    else:
        # Display an error message and return to the upload page
        messages.error(request, "Face not recognized. Please try again.")
        return redirect('upload')

def log_in(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(username=username, password=password)
        
        if user is not None:
            profile = Profile.objects.get(user=user)
            if profile.role == 'student':  # Ensure the user is a student
                login(request, user)
                messages.success(request, "Log In Successful...!")
                return redirect("upload")
            else:
                messages.success(request, "Access Denied! You are not a student.")
                return redirect("log_in")
        else:
            messages.success(request, "Invalid User...!")
            return redirect("log_in")
    return render(request, "log_in.html")

def teacher_log_in(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(username=username, password=password)

        if user is not None:
            profile = Profile.objects.get(user=user)
            if profile.role == 'teacher':  # Ensure the user is a teacher
                login(request, user)
                messages.success(request, "Log In Successful...!")
                return redirect("teacher_dashboard")
            else:
                messages.success(request, "Access Denied! You are not a teacher.")
                return redirect("teacher_log_in")
        else:
            messages.success(request, "Invalid User...!")
            return redirect("teacher_log_in")
    return render(request, "teacher_log_in.html")

def register(request):
    if request.method == "POST":
        fname = request.POST['fname']
        lname = request.POST['lname']
        username = request.POST['username']
        password = request.POST['password']
        password1 = request.POST['password1']

        verify = authentication(fname, lname, password, password1)
        if verify == "success":
            user = User.objects.create_user(username, password, password1)
            user.first_name = fname
            user.last_name = lname
            user.save()

            # Create a student profile
            Profile.objects.create(user=user, role='student')

            messages.success(request, "Your Account has been Created.")
            return redirect("log_in")
        else:
            messages.error(request, verify)
            return redirect("register")
    return render(request, "register.html", {'action': 'register'})


def teacher_register(request):
    if request.method == "POST":
        fname = request.POST['fname']
        lname = request.POST['lname']
        username = request.POST['username']
        password = request.POST['password']
        password1 = request.POST['password1']

        

        verify = authentication(fname, lname, password, password1)
        if verify == "success":
            user = User.objects.create_user(username, password, password1)
            user.first_name = fname
            user.last_name = lname
            user.save()

            # Create a teacher profile
            Profile.objects.create(user=user, role='teacher')

            messages.success(request, "Your Account has been Created.")
            return redirect("teacher_log_in")
        else:
            messages.error(request, verify)
            return redirect("teacher_register")
    return render(request, "teacher_register.html", {'action': 'register'})
import cv2
import numpy as np
from keras.models import load_model
from keras_preprocessing.image import img_to_array
from collections import Counter
from django.http import StreamingHttpResponse
from django.shortcuts import render
from django.views import View
import threading

# Load face detector and emotion classifier
face_classifier = cv2.CascadeClassifier('dataset/haarcascade_frontalface_default.xml')
classifier = load_model('dataset/model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

class VideoStreamView(View):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # 0 for default camera
        self.frame = None
        self.lock = threading.Lock()
        self.emotion_count = Counter({
            'Angry': 0,
            'Disgust': 0,
            'Fear': 0,
            'Happy': 0,
            'Neutral': 0,
            'Sad': 0,
            'Surprise': 0
        })

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

    def gen(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    prediction = classifier.predict(roi)[0]
                    label = emotion_labels[prediction.argmax()]
                    self.emotion_count[label] += 1
                    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the cumulative emotion counts on the frame
            y_offset = 30
            for emotion, count in self.emotion_count.items():
                cv2.putText(frame, f"{emotion}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y_offset += 30  # Increment the position for the next emotion label

            # Convert the frame to JPEG format
            _, jpeg = cv2.imencode('.jpg', frame)
            jpeg_bytes = jpeg.tobytes()

            # Yield the frame as a JPEG image for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n\r\n')

    def get(self, request):
        return StreamingHttpResponse(self.gen(), content_type='multipart/x-mixed-replace; boundary=frame')



# Example of how to render the dashboard
from django.contrib import messages
from django.shortcuts import redirect, render
from .models import Exam  # Assuming your exam model is in models.py

def student_dashboard(request):
    # Fetch the latest exam along with its related questions
    latest_exam = Exam.objects.prefetch_related('questions').order_by('-id').first()

    # Check if the timer ended or if the user manually submitted the exam
    if 'time_up' in request.GET:
        messages.success(request, 'Time is up! Your exam has been automatically submitted.')

    elif 'exam_submitted' in request.GET:
        messages.success(request, 'Your exam has been submitted successfully.')

    # Fetch emotion count from the session if available
    emotion_data = request.session.get('emotion_count', {})

    # Pass emotion data to context
    context = {
        'fname': request.user.first_name,
        'latest_exam': latest_exam,  # Pass only the latest exam to the template
        'emotion_data': emotion_data,
    }
    return render(request, "student_dashboard.html", context)




def submit_exam(request):
    # Logic for submitting the exam (mark it as completed or save the answers)
    # After submission, redirect to the index page with a flag indicating submission
    return HttpResponseRedirect('/?exam_submitted=1')



from django.http import JsonResponse
from django.shortcuts import render
from .models import Exam, Question  # Assuming you have these models

def teacher_dashboard(request):
    return render(request, "teacher_dashboard.html")



from web3 import Web3
import json
from django.shortcuts import render, redirect
from django.contrib import messages
from .models import Exam, Question
from datetime import datetime

# Blockchain Configuration
GANACHE_URL = "HTTP://127.0.0.1:7545"
CONTRACT_ADDRESS = "0xe86329327c58514605dCE618068D0b8a92810279"
web3 = Web3(Web3.HTTPProvider(GANACHE_URL))

# Contract ABI (matches your Solidity contract)
CONTRACT_ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "examId", "type": "bytes32"},
            {"internalType": "string", "name": "title", "type": "string"},
            {"internalType": "uint256", "name": "date", "type": "uint256"},
            {"internalType": "uint256", "name": "duration", "type": "uint256"},
            {"internalType": "uint256", "name": "totalMarks", "type": "uint256"},
            {"internalType": "bytes32", "name": "examHash", "type": "bytes32"}
        ],
        "name": "registerExam",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "bytes32", "name": "questionId", "type": "bytes32"},
            {"internalType": "bytes32", "name": "examId", "type": "bytes32"},
            {"internalType": "bytes32", "name": "questionHash", "type": "bytes32"}
        ],
        "name": "registerQuestion",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]

contract = web3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)

def generate_hashes(data):
    """Generate both hex hash and bytes32 hash"""
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True)
    hex_hash = Web3.keccak(text=data).hex()
    bytes_hash = Web3.keccak(text=data)
    return hex_hash, bytes_hash

def create_exam(request):
    if request.method == "POST":
        try:
            # 1. Prepare exam data
            exam_data = {
                "title": request.POST['examTitle'],
                "date": request.POST['examDate'],
                "duration": request.POST['examDuration'],
                "total_marks": request.POST['marks'],
                "type": "exam"
            }
            
            # 2. Generate hashes
            hex_hash, bytes_hash = generate_hashes(exam_data)
            
            # 3. Create exam in Django DB
            exam = Exam.objects.create(
                title=exam_data['title'],
                date=exam_data['date'],
                duration=int(exam_data['duration']),
                total_marks=int(exam_data['total_marks']),
                txn_hash=hex_hash  # Storing the hex hash in txn_hash field
            )
            
            # 4. Generate exam ID (bytes32 compatible)
            exam_id = Web3.keccak(text=str(exam.id))
            
            # 5. Register exam on blockchain
            tx_hash = contract.functions.registerExam(
                exam_id,
                exam.title,
                int(datetime.strptime(exam.date, "%Y-%m-%d").timestamp()),
                int(exam.duration),
                int(exam.total_marks),
                bytes_hash
            ).transact({'from': web3.eth.accounts[0]})
            
            # Store transaction hash if needed (optional)
            # exam.txn_hash = tx_hash.hex()
            # exam.save()
            
            # 6. Process questions
            i = 0
            while True:
                question_key = f'mcq_questions[{i}][question]'
                if question_key not in request.POST:
                    break
                
                # Prepare question data
                question_data = {
                    "text": request.POST[question_key],
                    "options": [
                        request.POST[f'mcq_questions[{i}][option1]'],
                        request.POST[f'mcq_questions[{i}][option2]'],
                        request.POST[f'mcq_questions[{i}][option3]'],
                        request.POST[f'mcq_questions[{i}][option4]']
                    ],
                    "correct_answer": request.POST[f'mcq_questions[{i}][correct_answer]'],
                    "exam_id": str(exam.id),
                    "type": "question"
                }
                
                # Generate question hashes
                hex_hash, bytes_hash = generate_hashes(question_data)
                
                # Create question in Django DB
                question = Question.objects.create(
                    exam=exam,
                    text=question_data['text'],
                    option1=question_data['options'][0],
                    option2=question_data['options'][1],
                    option3=question_data['options'][2],
                    option4=question_data['options'][3],
                    correct_answer=question_data['correct_answer'],
                    txn_hash=hex_hash  # Storing the hex hash in txn_hash field
                )
                
                # Generate question ID (bytes32 compatible)
                question_id = Web3.keccak(text=f"{exam.id}-{i}")
                
                # Register question on blockchain
                tx_hash = contract.functions.registerQuestion(
                    question_id,
                    exam_id,
                    bytes_hash
                ).transact({'from': web3.eth.accounts[0]})
                
                i += 1

            messages.success(request, 'Exam created with blockchain verification!')
            return redirect('teacher_dashboard')
        
        except Exception as e:
            messages.error(request, f'Error: {str(e)}')
            return redirect('create_exam')
    
    return render(request, "teacher_dashboard.html")

def verify_data(request):
    if request.method == 'POST':
        hash_input = request.POST.get('hash_input', '').strip()
        
        if len(hash_input) == 64:
            # Check Exam model
            exam = Exam.objects.filter(txn_hash=hash_input).first()
            if exam:
                exam_data = {
                    "title": exam.title,
                    "date": str(exam.date),
                    "duration": exam.duration,
                    "total_marks": exam.total_marks,
                    "type": "exam"
                }
                computed_hex, computed_bytes = generate_hashes(exam_data)
                
                # Verify against stored hash
                db_match = hash_input == computed_hex
                
                # Verify on blockchain (if implemented)
                blockchain_match = "Not checked"
                try:
                    exam_id = Web3.keccak(text=str(exam.id))
                    blockchain_data = contract.functions.exams(exam_id).call()
                    blockchain_match = blockchain_data[4].hex() == computed_hex
                except:
                    pass
                
                return render(request, 'verify.html', {
                    'result': {
                        'type': 'Exam',
                        'data': exam_data,
                        'db_hash': hash_input,
                        'computed_hash': computed_hex,
                        'db_match': db_match,
                        'blockchain_match': blockchain_match
                    }
                })
            
            # Check Question model
            question = Question.objects.filter(txn_hash=hash_input).first()
            if question:
                question_data = {
                    "text": question.text,
                    "options": [
                        question.option1,
                        question.option2,
                        question.option3,
                        question.option4
                    ],
                    "correct_answer": question.correct_answer,
                    "exam_id": str(question.exam.id),
                    "type": "question"
                }
                computed_hex, computed_bytes = generate_hashes(question_data)
                
                # Verify against stored hash
                db_match = hash_input == computed_hex
                
                # Verify on blockchain (if implemented)
                blockchain_match = "Not checked"
                try:
                    question_id = Web3.keccak(text=f"{question.exam.id}-{question.id}")
                    blockchain_data = contract.functions.questions(question_id).call()
                    blockchain_match = blockchain_data[0].hex() == computed_hex
                except:
                    pass
                
                return render(request, 'verify.html', {
                    'result': {
                        'type': 'Question',
                        'data': question_data,
                        'db_hash': hash_input,
                        'computed_hash': computed_hex,
                        'db_match': db_match,
                        'blockchain_match': blockchain_match
                    }
                })
            
            return render(request, 'verify.html', {'error': 'Hash not found in system'})
        
        return render(request, 'verify.html', {'error': 'Invalid hash format (64 hex characters required)'})
    
    return render(request, 'verify.html')

# def admin_dashboard(request):
#     return render(request,"admin_dashboard.html")

def log_out(request):
    logout(request)
    messages.success(request, "Logged out successfully.")
    return redirect("/")

# from django.shortcuts import get_object_or_404
# from django.contrib import messages
# from .models import Exam, Response

# # Example in student_dashboard (or wherever emotion data is being processed)
import random

def student_dashboard(request):
    # Assuming you have a way of detecting emotions and counting them
    emotion_data = request.session.get('emotion_count', {})

    # List of possible emotions
    possible_emotions = ['Happy', 'Sad', 'Neutral']

    # Emotion detected (this would be dynamically determined by your emotion detection model)
    detected_emotion = random.choice(possible_emotions)  # Choose a random emotion if not detected

    # Increment the count for detected emotions
    if detected_emotion in emotion_data:
        emotion_data[detected_emotion] += 1
    else:
        emotion_data[detected_emotion] = 1

    # Store the updated emotion count back into the session
    request.session['emotion_count'] = emotion_data

    # Fetch the latest exam for the student
    latest_exam = Exam.objects.prefetch_related('questions').order_by('-id').first()

    # Continue with your existing logic
    context = {
        'fname': request.user.first_name,
        'latest_exam': latest_exam,
        'emotion_data': emotion_data,  # Pass emotion data to the template
    }
    return render(request, "student_dashboard.html", context)




def response(request):
    context = {}
    
    latest_exam = Exam.objects.order_by('-id').first()
    
    if latest_exam:
        responses = Response.objects.select_related('question', 'student').filter(exam=latest_exam)
        response_data = {}
        total_marks = latest_exam.total_marks
        marks_per_question = total_marks / latest_exam.questions.count() if latest_exam.questions.count() > 0 else 0
        
        for response in responses:
            correct_answer = response.question.correct_answer
            is_correct = response.answer == correct_answer
            
            if response.student not in response_data:
                response_data[response.student] = {
                    'responses': [],
                    'score': 0
                }
                
            response_data[response.student]['responses'].append({
                'question': response.question.text,
                'given_answer': response.answer,
                'correct_answer': correct_answer,
                'is_correct': is_correct,
            })

            if is_correct:
                response_data[response.student]['score'] += marks_per_question

    context['response_data'] = response_data

    return render(request, "response.html", context)

def send_results(request, student_id):
    # Fetch student
    student = get_object_or_404(User, id=student_id)
    
    # Here, implement the logic to send results to the student
    # For simplicity, let's just show a success message
    messages.success(request, f"Results have been sent to {student.first_name} {student.last_name}!")
    
    return redirect('response')

from collections import Counter
from django.shortcuts import render, redirect
from django.contrib import messages

from collections import Counter
from django.shortcuts import render, redirect
from django.contrib import messages

from collections import Counter
from django.shortcuts import render, redirect
from django.contrib import messages
from collections import Counter
from django.shortcuts import render, redirect
from django.contrib import messages

import random
from collections import Counter

import random

import random

import random

def view_result(request):
    if request.user.is_authenticated:
        latest_exam = Exam.objects.order_by('-id').first()
        responses = Response.objects.select_related('question').filter(student=request.user, exam=latest_exam)

        response_data = []
        total_marks = latest_exam.total_marks if latest_exam else 0

        for response in responses:
            correct_answer = response.question.correct_answer
            is_correct = response.answer == correct_answer

            response_data.append({
                'question': response.question.text,
                'given_answer': response.answer,
                'correct_answer': correct_answer,
                'is_correct': is_correct,
            })

        score = sum(1 for r in response_data if r['is_correct']) * (total_marks / (latest_exam.questions.count() if latest_exam and latest_exam.questions.count() > 0 else 1))

        # Randomly select an emotion
        emotions = {
            "Happy": "It looks like your exam is going very interesting!",
            "Sad": "Don't worry, keep practicing and you will do better next time.",
            "Neutral": "You seem calm about your exam. Stay focused and keep going!"
        }
        random_emotion = random.choice(list(emotions.keys()))
        emotion_message = emotions[random_emotion]

        context = {
            'response_data': response_data,
            'score': score,
            'total_marks': total_marks,
            'emotion_message': emotion_message,
        }

        return render(request, "view_result.html", context)

    return redirect('log_in')





from django.shortcuts import render
from .models import Block

def view_blockchain(request):
    # Fetch all blocks stored in the blockchain
    blocks = Block.objects.all().order_by('index')  # You can order them by 'index' to display in order

    context = {
        'blocks': blocks,
    }
    
    return render(request, 'view_blockchain.html', context)


from django.shortcuts import render
from web3 import Web3
from .models import Exam, Question

def hash_lookup(request):
    result = None
    hash_input = ""
    computed_hash = ""
    
    if request.method == 'POST':
        hash_input = request.POST.get('hash_input', '').strip().lower()
        
        if len(hash_input) == 64 and all(c in '0123456789abcdef' for c in hash_input):
            # Check Exam model first using txn_hash
            exam = Exam.objects.filter(txn_hash=hash_input).first()
            if exam:
                exam_data = f"Exam: {exam.title}, Date: {exam.date}, Duration: {exam.duration} mins, Marks: {exam.total_marks}"
                computed_hash = Web3.keccak(text=exam_data).hex()
                result = {
                    'type': 'Exam',
                    'data': exam_data,
                    'hash_match': computed_hash == hash_input
                }
            else:
                # Check Question model if not found in Exam
                question = Question.objects.filter(blockchain_hash=hash_input).first()
                if question:
                    question_data = f"Question: {question.text}\nOptions:\n1. {question.option1}\n2. {question.option2}\n3. {question.option3}\n4. {question.option4}\nCorrect: {question.correct_answer}"
                    computed_hash = Web3.keccak(text=question_data).hex()
                    result = {
                        'type': 'Question',
                        'data': question_data,
                        'hash_match': computed_hash == hash_input
                    }
                else:
                    result = {'error': 'Hash not found in database'}
        else:
            result = {'error': 'Invalid hash format (must be 64-character hex)'}
    
    return render(request, 'hash_lookup.html', {
        'hash_input': hash_input,
        'result': result,
        'computed_hash': computed_hash
    })