import os
import numpy as np
import cv2
import threading
from collections import defaultdict
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind, chi2_contingency, entropy
from statsmodels.stats.weightstats import ztest
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

# Global variable
stop_flag = threading.Event()

# Load pre-trained sentiment analysis model
emotion_recognition = pipeline('sentiment-analysis')

# train dataset
train_dir = 'images/train/'
validation_dir = 'images/validation/'

# Picture generation
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical'
)

# build model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train model
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator
)

# 保存训练历史记录
with open('training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)

# save model structure
model_json = model.to_json()
with open("custom_model.json", "w") as json_file:
    json_file.write(model_json)
# save model weight
model.save_weights("custom_model.weights.h5")

# Function to plot training history
def plot_training_history(history):
    plt.figure(figsize=(14, 6))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    plt.show()


# Load training history from file
with open('training_history.pkl', 'rb') as file:
    history = pickle.load(file)

# Plot training history
plot_training_history(history)


# Load custom model
def load_custom_model():
    json_file = open('custom_model.json', 'r')
    loaded_model_json = json_file.read()                        
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("custom_model.weights.h5")
    return loaded_model

# Function for facial emotion recognition
def detect_emotion_from_face(frame, custom_model):
    try:
        if frame is None or frame.size == 0:
            raise ValueError("Invalid input image")
        
        face_img = cv2.resize(frame, (48, 48))
        face_img = face_img.astype("float") / 255.0
        face_img = img_to_array(face_img)
        face_img = np.expand_dims(face_img, axis=0)

        predictions = custom_model.predict(face_img)
        max_index = np.argmax(predictions[0])
        emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        emotion = emotions[max_index]
        confidence = predictions[0][max_index]
        return emotion, confidence
    except Exception as e:
        print(f"Error in detecting emotion: {e}")
        return None, 0

# Emotion label mapping
def map_nlp_emotion(label):
    mapping = {
        'POSITIVE': 'happy',
        'NEGATIVE': 'sad',
        'NEUTRAL': 'neutral'
    }
    return mapping.get(label.upper(), 'neutral')

# Initialize emotion state statistics
state_counts = defaultdict(int)
joint_state_counts = defaultdict(int)
possible_states = ['happy', 'sad', 'neutral', 'angry', 'surprise', 'fear', 'disgust']
face_scores = []
text_scores = []

def analyze_emotions(frame_count=10):
    frame_emotions = []
    for _ in range(frame_count):
        if stop_flag.is_set():
            break
        ret, frame = cap.read()
        if not ret:
            break
        
        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            print("No face detected")
            continue

        (x, y, w, h) = faces[0]
        face = frame[y:y+h, x:x+w]
        
        # Detect face emotion
        face_emotion, face_confidence = detect_emotion_from_face(face, custom_model)
        if face_emotion:
            frame_emotions.append((face_emotion, face_confidence))
        
        # Display video
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag.set()
            break
    
    if frame_emotions:
        avg_emotion_confidence = defaultdict(float)
        for emotion, confidence in frame_emotions:
            avg_emotion_confidence[emotion] += confidence / len(frame_emotions)
        
        dominant_emotion = max(avg_emotion_confidence, key=avg_emotion_confidence.get)
        confidence = avg_emotion_confidence[dominant_emotion]
        
        print(f"Average face emotion: {dominant_emotion} ({confidence*100:.2f}%)")
        
        return dominant_emotion, confidence
    return None, 0

def main_loop():
    while not stop_flag.is_set():
        user_input = input("Please enter a sentence that expresses your current emotion (type 'q' to quit): ")
        if user_input.lower() == 'q':
            stop_flag.set()
            break
        
        # Analyze sentiment of user input
        text_emotion = emotion_recognition(user_input)[0]
        mapped_text_emotion = map_nlp_emotion(text_emotion['label'])
        
        # Detect face emotion
        face_emotion, face_confidence = analyze_emotions()
        
        # Compare face and text emotions
        print(f"Face emotion: {face_emotion} ({face_confidence*100:.2f}%)")
        print(f"Text emotion: {mapped_text_emotion} ({text_emotion['score']*100:.2f}%)")
        
        if face_emotion in possible_states and mapped_text_emotion in possible_states:
            state_counts[(face_emotion, mapped_text_emotion)] += 1
            for prev_state in state_counts:
                joint_state_counts[(prev_state, (face_emotion, mapped_text_emotion))] += 1

        if face_confidence > 0:
            face_scores.append(face_confidence)
        if text_emotion['score'] > 0:
            text_scores.append(text_emotion['score'])

    if face_scores and text_scores:
        pca = PCA(n_components=1)
        pca_result = pca.fit_transform(np.array([face_scores, text_scores]).T)
        print(f"PCA result: {pca_result}")
        
        z_stat, p_val = ztest(face_scores, value=np.mean(text_scores))
        print(f"Z-test statistic: {z_stat}, p-value: {p_val}")
        
        t_stat, t_p_val = ttest_ind(face_scores, text_scores)
        print(f"T-test statistic: {t_stat}, p-value: {t_p_val}")
        
        if all(x > 0 for x in face_scores + text_scores):
            chi2_stat, chi2_p_val, _, _ = chi2_contingency([face_scores, text_scores])
            print(f"Chi-square test statistic: {chi2_stat}, p-value: {chi2_p_val}")
        else:
            print("Chi-square test cannot be performed: frequency values contain zero")
        
        face_entropy = entropy(face_scores)
        text_entropy = entropy(text_scores)
        print(f"Face emotion entropy: {face_entropy}, Text emotion entropy: {text_entropy}")
        
        total_transitions = sum(joint_state_counts.values())
        markov_probabilities = {k: v / total_transitions for k, v in joint_state_counts.items()}
        print("Markov joint state probability matrix:")
        print(markov_probabilities)

        total_counts = sum(state_counts.values())
        joint_vector_probabilities = {k: v / total_counts for k, v in state_counts.items()}
        print("Joint vectorized observation state probability:")
        print(joint_vector_probabilities)

        # Visualize results
        visualize_results(pca_result, face_scores, text_scores, markov_probabilities, joint_vector_probabilities)

def visualize_results(pca_result, face_scores, text_scores, markov_probabilities, joint_vector_probabilities):
    # Convert joint probabilities to DataFrame for heatmap
    joint_df = pd.DataFrame(list(joint_vector_probabilities.items()), columns=['States', 'Probability'])
    joint_df[['Face Emotion', 'Text Emotion']] = pd.DataFrame(joint_df['States'].tolist(), index=joint_df.index)
    joint_pivot = joint_df.pivot(index='Face Emotion', columns='Text Emotion', values='Probability')

    # Plotting face and text emotion scores
    results_df = pd.DataFrame({
        'Face Scores': face_scores,
        'Text Scores': text_scores,
        'PCA1': pca_result[:, 0]
    })

    plt.figure(figsize=(14, 6))
    sns.scatterplot(data=results_df, x='Face Scores', y='Text Scores')
    plt.title('Face Emotion vs Text Emotion Distribution')
    plt.xlabel('Face Emotion Scores')
    plt.ylabel('Text Emotion Scores')
    plt.show()

    plt.figure(figsize=(14, 6))
    sns.scatterplot(data=results_df, x='PCA1', y='Face Scores')
    plt.title('PCA Result')
    plt.xlabel('PCA 1')
    plt.ylabel('Face Scores')
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(joint_pivot, annot=True, cmap='coolwarm', cbar=True)
    plt.title('Joint State Probability Matrix')
    plt.show()

# Get video input
cap = cv2.VideoCapture(0)

# Lower camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load custom model
custom_model = load_custom_model()

# Start emotion analysis thread
emotion_thread = threading.Thread(target=main_loop)
emotion_thread.start()

# Main thread for displaying video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or stop_flag.is_set():
        break
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_flag.set()
        break

cap.release()
cv2.destroyAllWindows()
emotion_thread.join()

