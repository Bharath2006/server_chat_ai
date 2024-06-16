import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask application
app = Flask(__name__)
CORS(app)

# Book data with summaries
books_data = [
    {"title": "Harry Potter series by J.K. Rowling", "description": "Harry Potter is a series of fantasy novels written by British author J. K. Rowling. The novels chronicle the lives of a young wizard, Harry Potter, and his friends Hermione Granger and Ron Weasley, all of whom are students at Hogwarts School of Witchcraft and Wizardry. The main story arc concerns Harry's struggle against Lord Voldemort, a dark wizard who intends to become immortal, overthrow the wizard governing body known as the Ministry of Magic, and subjugate all wizards and Muggles.", "summary": "A young wizard, Harry Potter, and his friends fight against the dark wizard Lord Voldemort."},
    {"title": "The Hobbit by J.R.R. Tolkien", "description": "The Hobbit is a fantasy novel by J.R.R. Tolkien. It follows the adventures of Bilbo Baggins, a hobbit who enjoys a peaceful, pastoral life. Bilbo is thrust into an epic quest to reclaim the lost Kingdom of Erebor from the fearsome dragon Smaug. Bilbo's journey takes him from his rural surroundings into a world of magical creatures and dangerous territories.", "summary": "Bilbo Baggins embarks on a quest to reclaim a lost kingdom from the dragon Smaug."},
    {"title": "The Catcher in the Rye by J.D. Salinger", "description": "The Catcher in the Rye is a novel by J.D. Salinger. It is narrated by Holden Caulfield, a teenager who has just been expelled from his prep school. Disillusioned and lonely, Holden goes on a journey through New York City, grappling with complex feelings of angst and alienation.", "summary": "Holden Caulfield, a disillusioned teenager, wanders New York City."},
    {"title": "1984 by George Orwell", "description": "1984 is a dystopian social science fiction novel by George Orwell. It presents a future totalitarian society dominated by Big Brother, where individualism and independent thinking are persecuted. The protagonist, Winston Smith, works for the Party rewriting history, but secretly he despises the regime and dreams of rebellion.", "summary": "In a totalitarian society, Winston Smith rebels against the oppressive regime."},
    {"title": "To Kill a Mockingbird by Harper Lee", "description": "To Kill a Mockingbird is a novel by Harper Lee. The story is set in the 1930s in the racially charged atmosphere of the Southern United States. It is narrated by Scout Finch, a young girl whose father, Atticus Finch, is a lawyer defending a black man falsely accused of raping a white woman. The novel addresses themes of racial injustice and moral growth.", "summary": "A young girl, Scout Finch, observes her father defend a black man falsely accused of rape."},
    {"title": "Pride and Prejudice by Jane Austen", "description": "Pride and Prejudice is a romantic novel by Jane Austen. It follows the life of Elizabeth Bennet as she deals with issues of manners, upbringing, morality, and marriage. Elizabeth's primary conflict is with the proud and wealthy Mr. Darcy, but as the story progresses, she discovers the complexity of his character.", "summary": "Elizabeth Bennet navigates societal expectations and her feelings for Mr. Darcy."},
    {"title": "The Great Gatsby by F. Scott Fitzgerald", "description": "The Great Gatsby is a novel by F. Scott Fitzgerald. It is narrated by Nick Carraway, who tells the story of Jay Gatsby, a mysterious millionaire known for his lavish parties. The novel explores themes of decadence, idealism, resistance to change, social upheaval, and excess, creating a portrait of the Jazz Age or the Roaring Twenties.", "summary": "Nick Carraway narrates the story of Jay Gatsby, a millionaire with a mysterious past."},
    {"title": "Moby-Dick by Herman Melville", "description": "Moby-Dick is a novel by Herman Melville. It follows the quest of Ishmael, a sailor aboard the whaling ship Pequod, commanded by Captain Ahab. Ahab is obsessed with seeking revenge on Moby Dick, a giant white sperm whale that had previously destroyed Ahab's ship and severed his leg.", "summary": "Captain Ahab obsessively hunts the giant white whale Moby Dick."},
    {"title": "Jane Eyre by Charlotte Brontë", "description": "Jane Eyre is a novel by Charlotte Brontë. It is a bildungsroman that follows the experiences of its eponymous heroine, including her growth to adulthood and her love for Mr. Rochester, the brooding master of Thornfield Hall. The novel addresses themes of class, sexuality, religion, and feminism.", "summary": "Jane Eyre grows up and falls in love with the brooding Mr. Rochester."},
    {"title": "The Lord of the Rings by J.R.R. Tolkien", "description": "The Lord of the Rings is an epic fantasy novel by J.R.R. Tolkien. It is set in Middle-earth and follows the quest to destroy the One Ring, which was forged by the Dark Lord Sauron. The story follows the fellowship of nine members, including hobbits Frodo Baggins and Samwise Gamgee, as they undertake a perilous journey to Mount Doom to destroy the ring and save Middle-earth from Sauron's tyranny.", "summary": "Frodo Baggins and his friends embark on a quest to destroy the One Ring."}
]

# Convert to DataFrame for easier manipulation
df_books = pd.DataFrame(books_data)

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer().fit(df_books['description'])

def generate_answer(question, df_books, vectorizer):
    """
    Generates an answer based on the question using the provided book descriptions.
    """
    # Check if the question is empty
    if not question.strip():
        return "The question cannot be empty. Please ask a valid question."
    
    question_vec = vectorizer.transform([question])
    description_vecs = vectorizer.transform(df_books['description'])
    similarities = cosine_similarity(question_vec, description_vecs).flatten()
    
    # Find the book with the highest similarity
    max_sim_index = similarities.argmax()
    most_relevant_book = df_books.iloc[max_sim_index]
    
    # Check if the highest similarity score is above a certain threshold
    if similarities[max_sim_index] < 0.1:  # threshold can be adjusted
        return "Sorry, I couldn't find a book that matches your question."
    
    # Return the summary of the most relevant book
    return most_relevant_book['summary']

# Endpoint to handle user questions
@app.route('/ask', methods=['POST'])
def ask_question():
    if request.method == 'POST':
        question = request.json.get('question', '')
        
        # Generate an answer based on the provided data
        answer = generate_answer(question, df_books, vectorizer)
        
        return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
