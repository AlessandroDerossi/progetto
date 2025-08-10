from flask_login import UserMixin

# Classe User (programmazione oggetti) per Flask-Login
class User(UserMixin):
    def __init__(self, user_id, username, email):
        super().__init__()
        self.id = user_id
        self.username = username
        self.email = email