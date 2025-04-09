# tests/test_text_generator.py

from app.services.text_generator import GPT2TextGenerator

def test_generation():
    generator = GPT2TextGenerator()
    text = generator.generate_text("Eren is building", max_length=20)
    assert isinstance(text, str)
    assert len(text) > 0