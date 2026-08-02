from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

doc = SimpleDocTemplate("test_data/project_brief.pdf", pagesize=letter)
styles = getSampleStyleSheet()
story = [
    Paragraph("Jarvis Pi Assistant - Sprint 2 Brief", styles["Title"]),
    Spacer(1, 12),
    Paragraph("Sprint 2 focuses on replacing the mocked Gmail and Google Calendar tools "
              "with real API calls using per-user OAuth tokens. The Raspberry Pi 5 demo "
              "is due in two weeks.", styles["Normal"]),
    Spacer(1, 12),
    Paragraph("Key risk: qwen2.5:1.5b required multiple retries to reliably trigger tool "
              "calls, so qwen2.5:3b remains the pinned routing model.", styles["Normal"]),
]
doc.build(story)
print("PDF created successfully.")
