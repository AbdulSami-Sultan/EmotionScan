

def get_lighting_color(emotion):
    emotion_colors = {
        0: (255, 255, 0),  # Happy: Yellow
        1: (0, 0, 255),    # Sad: Blue
        2: (255, 0, 0),    # Angry: Red
        3: (255, 255, 255) # Neutral: White
    }
    return emotion_colors.get(emotion, (255, 255, 255))