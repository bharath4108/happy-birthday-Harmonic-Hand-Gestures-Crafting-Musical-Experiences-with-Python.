import cv2
import pygame
import numpy as np
import time
import mediapipe as mp

# Initialize pygame mixer
pygame.mixer.init()

# Frequencies for the musical notes
NOTE_FREQS = {
    "C4": 261.63,
    "D4": 293.66,
    "E4": 329.63,
    "F4": 349.23,
    "G4": 392.00,
    "A4": 440.00,
    "B4": 493.88,
    "C5": 523.25,
    "A#4": 466.16
}

HAPPY_BIRTHDAY_SEQ = [
    "C4", "C4", "D4", "C4", "F4", "E4",
    "C4", "C4", "D4", "C4", "G4", "F4",
    "C4", "C4", "C5", "A4", "F4", "E4", "D4",
    "A#4", "A#4", "A4", "F4", "G4", "F4"
]


def generate_sine_wave(freq, duration, volume=0.5, rate=44100):
    t = np.linspace(0, duration, int(rate * duration), endpoint=False)
    mono_signal = (32767 * volume * np.sin(2 * np.pi * freq * t)).astype(np.int16)

    # Convert mono to stereo and ensure it's C-contiguous
    stereo_signal = np.ascontiguousarray(np.vstack([mono_signal, mono_signal]).T)

    return stereo_signal


# Create the notes using pygame
notes = {}
for name, freq in NOTE_FREQS.items():
    samples = generate_sine_wave(freq, 1.5)  # 1.5 seconds per note
    note = pygame.sndarray.make_sound(samples)
    note.set_volume(0.9)
    notes[name] = note

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

note_idx = 0
last_played = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get the hand landmarks
    results = hands.process(rgb_frame)

    # Draw the hand landmarks on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if time.time() - last_played > 1.5:
                cv2.putText(frame, 'Playing Note', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                notes[HAPPY_BIRTHDAY_SEQ[note_idx]].play()
                note_idx += 1
                if note_idx == len(HAPPY_BIRTHDAY_SEQ):
                    note_idx = 0
                last_played = time.time()

    cv2.imshow('Happy Birthday Hand Gesture Music', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
