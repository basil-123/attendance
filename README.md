# Smart Attendance System using Face Recognition

This project is an automated attendance system that uses **Face Recognition** to mark attendance and stores the data in a **Supabase (PostgreSQL)** database. It utilizes Deep Learning models (RetinaFace/FaceNet) to detect and verify faces from video feeds.

> **Note:** This project is based on/adapted from the original work by **[]**.

## 🚀 Features
- **Face Detection & Recognition:** Uses `DeepFace` (RetinaFace/FaceNet) for high-accuracy recognition.
- **Real-time Processing:** Processes video input to identify registered users.
- **Cloud Database:** Syncs attendance logs instantly with Supabase.
- **Secure:** Uses environment variables to protect database credentials.

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Computer Vision:** OpenCV, DeepFace
- **Database:** Supabase (PostgreSQL)
- **Dependency Management:** Pip / Pipenv

## 📂 Project Structure
```text
attendance/
├── saved_models/      # (Ignored) Pre-trained models
├── basilvideo4.py     # Main script for attendance
├── requirements.txt   # (Optional) List of dependencies
├── .gitignore         # Config to ignore large files
└── README.md          # Project documentation
