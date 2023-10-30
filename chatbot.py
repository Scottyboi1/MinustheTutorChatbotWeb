import os
from flask import Flask, render_template, request, jsonify
from vertexai.language_models import TextGenerationModel

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "cred.json"

app = Flask(__name__)

model = TextGenerationModel.from_pretrained("text-bison")

class Chatbot:
    def get_response(self, user_input, subject, grade_level):
        elementary_prompt = f"Explain in detail how to solve this {subject} question for elementary school students: {user_input}"
        middle_prompt = f"Provide a comprehensive explanation for this {subject} question suitable for middle school students: {user_input}"
        high_school_prompt = f"Give a detailed answer to this {subject} question at the high school level: {user_input}"

        prompt = {
            "Elementary School": elementary_prompt,
            "Middle School": middle_prompt,
            "High School": high_school_prompt,
        }.get(grade_level, user_input)

        parameters = {
            "candidate_count": 1,
            "max_output_tokens": 1024,
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40
        }
        response = model.predict(prompt, **parameters)
        return response.text


chatbot_instance = Chatbot()

@app.route('/')
def chatbot():
    return render_template('chatbot.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    subject = request.form['subject']
    grade_level = request.form['grade_level']

    response = chatbot_instance.get_response(user_input, subject, grade_level)

    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)
