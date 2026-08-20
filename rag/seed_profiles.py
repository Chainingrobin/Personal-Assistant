# rag/seed_profiles.py
from rag.ingest import ingest_document

PROFILES = [
    # ─── Robin ──────────────────────────────────────────────────────────
    {"doc_id": "robin_course_1", "content": "Robin is enrolled in Deep Learning and Computer Vision this semester as part of the Media Engineering and Technology program.",
     "metadata": {"user_id": "robin", "type": "course"}},
    {"doc_id": "robin_course_2", "content": "Robin is taking an Embedded Systems and Sustainable Smart Sensor Systems course, which is where this Jarvis assistant project comes from.",
     "metadata": {"user_id": "robin", "type": "course"}},
    {"doc_id": "robin_milestone_thesis", "content": "Robin's bachelor thesis is titled 'Sub-pixel Info for Multi-image Super-Resolution', applying deep learning MISR to microscopic semiconductor chip imaging, supervised by Dr. Hisham Hassaballah Othman and Dr. Haitham Omran.",
     "metadata": {"user_id": "robin", "type": "milestone"}},
    {"doc_id": "robin_milestone_defense", "content": "Robin is aiming to defend the thesis before the end of the academic year.",
     "metadata": {"user_id": "robin", "type": "milestone"}},
    {"doc_id": "robin_pref_study", "content": "Robin prefers not to study for more than 2 continuous hours without taking a break.",
     "metadata": {"user_id": "robin", "type": "preference"}},
    {"doc_id": "robin_pref_feedback", "content": "Robin prefers direct, concise feedback rather than lengthy softened explanations.",
     "metadata": {"user_id": "robin", "type": "preference"}},
    {"doc_id": "robin_hobby_retro", "content": "Robin enjoys retro gaming, CRT monitor collecting, and emulation as a way to unwind after long study sessions.",
     "metadata": {"user_id": "robin", "type": "hobby"}},
    {"doc_id": "robin_hobby_reading", "content": "Robin likes reading during short breaks rather than checking a phone.",
     "metadata": {"user_id": "robin", "type": "hobby"}},
    {"doc_id": "robin_constraint_deadline", "content": "Robin has a graded course project deadline that takes priority over personal side projects like the coding assistant setup.",
     "metadata": {"user_id": "robin", "type": "constraint"}},

    # ─── Youssef ────────────────────────────────────────────────────────
    {"doc_id": "youssef_internship_1", "content": "Youssef is currently interning part-time, focused on backend development work.",
     "metadata": {"user_id": "youssef", "type": "internship"}},
    {"doc_id": "youssef_course_1", "content": "Youssef is enrolled in the same Embedded Systems course and is a teammate on the Jarvis assistant project.",
     "metadata": {"user_id": "youssef", "type": "course"}},
    {"doc_id": "youssef_milestone_1", "content": "Youssef is working toward completing his internship deliverables alongside coursework this semester.",
     "metadata": {"user_id": "youssef", "type": "milestone"}},
    {"doc_id": "youssef_pref_study", "content": "Youssef prefers short, frequent breaks roughly every 45 minutes rather than long uninterrupted sessions.",
     "metadata": {"user_id": "youssef", "type": "preference"}},
    {"doc_id": "youssef_pref_notif", "content": "Youssef prefers a quiet notification (visual/OLED display) over an audible alert when a break suggestion triggers.",
     "metadata": {"user_id": "youssef", "type": "preference"}},
    {"doc_id": "youssef_hobby_1", "content": "Youssef likes going for a short walk or listening to music as a break activity.",
     "metadata": {"user_id": "youssef", "type": "hobby"}},
    {"doc_id": "youssef_constraint_time", "content": "Youssef has limited availability in the evenings due to the internship schedule.",
     "metadata": {"user_id": "youssef", "type": "constraint"}},
]

if __name__ == "__main__":
    for p in PROFILES:
        ingest_document(p["doc_id"], p["content"], p["metadata"])
    print(f"\nSeeded {len(PROFILES)} facts across {len(set(p['metadata']['user_id'] for p in PROFILES))} users.")
