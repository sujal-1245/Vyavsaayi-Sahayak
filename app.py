from flask import Flask, render_template, url_for, redirect, flash, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError, Regexp
from flask_bcrypt import Bcrypt
from flask_migrate import Migrate
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import string
import matplotlib.pyplot as plt
import io
import base64
from werkzeug.utils import secure_filename
import os
from deep_translator import GoogleTranslator
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']= False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
migrate = Migrate(app, db)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'signin'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    email = db.Column(db.String(120), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)


class RegisterForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired(), 
                                             Regexp(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$', message="Invalid email format")], 
                                             render_kw={"placeholder": "Email"})
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField('Sign Up')

    def validate_email(self, email):
        existing_user_email = User.query.filter_by(email=email.data).first()
        if existing_user_email:
            raise ValidationError('That email is already in use. Please choose a different one.')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(username=username.data).first()
        if existing_user_username:
            raise ValidationError('That username already exists. Please choose a different one.')

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired(), 
                                             Regexp(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$', message="Invalid email format")], 
                                             render_kw={"placeholder": "Email"})
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField('Sign In')



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/signin', methods=['GET', 'POST'])
def signin():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for('home'))
        elif user is None:
            flash('This user does not exist. Please sign up.', 'danger')
        else:
            flash('Incorrect credentials. Please try again.', 'danger')
    
    return render_template('sign-in.html', form=form)



@app.route('/loancalculator', methods=['GET', 'POST'])
def loancalculator():
    return render_template('Loan_Calculator.html')

@app.route('/profit', methods=['GET', 'POST'])
def profit():
    return render_template('Analyse_Profit.html')

@app.route('/summarize_docs')
def summarize_docs():
    return render_template('Summarize_DOCs.html') 


# Database Model
class UserData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False, unique=True)
    phone = db.Column(db.String(15), nullable=False)
    address = db.Column(db.String(200), nullable=False)
    education = db.Column(db.String(200), nullable=False)
    occupation = db.Column(db.String(100), nullable=False)
    photo = db.Column(db.String(200), nullable=True)  # Store photo file path

# Create tables
with app.app_context():
    db.create_all()

@app.route('/profile')
def profile():
    user = UserData.query.first()  # Fetch the first user from the database
    if user is None:
        # Return a message or redirect to a different page if user is not found
        return "No user data available. Please register or create user data first."
    return render_template('profile.html', user=user)


@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    user = UserData.query.first()  # Fetch the first user from the database
    if user is None:
        # Return a message or redirect to a different page if user is not found
        return "No user data available to edit."
    
    if request.method == 'POST':
        user.name = request.form['name']
        user.email = request.form['email']
        user.phone = request.form['phone']
        user.address = request.form['address']
        user.education = request.form['education']
        user.occupation = request.form['occupation']

        # Handle Photo Upload
        if 'photo' in request.files:
            file = request.files['photo']
            if file and file.filename:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                user.photo = filepath

        db.session.commit()
        return redirect(url_for('profile'))
    
    return render_template('edit_profile.html', user=user)

@app.route('/populate_user_data')
def populate_user_data():
    # Create a list of default user data to populate the table
    users_data = [
        {
            'name': 'John Doe',
            'email': 'john.doe@example.com',
            'phone': '1234567890',
            'address': '123 Main St, Springfield',
            'education': 'Bachelors in Computer Science',
            'occupation': 'Software Developer'
        },
        {
            'name': 'Jane Smith',
            'email': 'jane.smith@example.com',
            'phone': '9876543210',
            'address': '456 Elm St, Springfield',
            'education': 'Masters in Business Administration',
            'occupation': 'Marketing Manager'
        },
        {
            'name': 'Robert Brown',
            'email': 'robert.brown@example.com',
            'phone': '1122334455',
            'address': '789 Oak St, Springfield',
            'education': 'PhD in Electrical Engineering',
            'occupation': 'Research Scientist'
        }
    ]

    # Loop through the list and add each user to the database
    for user_data in users_data:
        user = UserData(
            name=user_data['name'],
            email=user_data['email'],
            phone=user_data['phone'],
            address=user_data['address'],
            education=user_data['education'],
            occupation=user_data['occupation']
        )
        db.session.add(user)

    # Commit changes to the database
    db.session.commit()

    # Return a message indicating that the table has been populated
    return "User data has been populated successfully!"


@app.route('/recommendations', methods=['GET', 'POST'])
def recommendations():
    return render_template('recommendations.html')


@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('signin'))


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()

    if form.validate_on_submit():
        existing_user_email = User.query.filter_by(email=form.email.data).first()
        existing_user_username = User.query.filter_by(username=form.username.data).first()

        if existing_user_email:
            flash('That email is already in use. Please choose a different one.', 'danger')
        elif existing_user_username:
            flash('That username already exists. Please choose a different one.', 'danger')
        else:
            hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
            new_user = User(email=form.email.data, username=form.username.data, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            flash('Account created successfully!', 'success')
            return redirect(url_for('signin'))

    return render_template('sign-up.html', form=form)

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Enhanced summarization function
def summarize_text(text, max_sentences=5):
    if not text.strip():
        return "Input text is empty. Please provide valid text."

    # Tokenize sentences and words
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [word for word in words if word not in stop_words and word not in string.punctuation]

    # Count word frequencies
    word_frequencies = Counter(words)
    high_priority_words = ["important", "objective", "key", "summary", "conclusion", "goal", "focus"]

    # Weight high-priority words more
    for word in high_priority_words:
        if word in word_frequencies:
            word_frequencies[word] += 3

    # Score sentences based on word frequencies
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_frequencies[word]

    # Rank sentences by score
    ranked_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)

    # Return the top sentences as summary
    important_points = ranked_sentences[:max_sentences]
    summary = "\n".join([f"- {sentence.strip()}" for sentence in important_points])

    return summary

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        with open(pdf_file, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"

# Translate text function
def translate_text(input_text, target_language):
    """
    Translate the given text into the target language using GoogleTranslator.
    
    Parameters:
        input_text (str): The text to translate.
        target_language (str): The language code for the target language.
    
    Returns:
        str: The translated text.
    """
    # Translate the text using deep_translator
    translated = GoogleTranslator(source='auto', target=target_language).translate(input_text)
    return translated


# Load and preprocess the dataset
df = pd.read_csv('loan_recommendation.csv')

# Handle missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# Encode categorical variables
categorical_cols = ['home_ownership', 'sanitary_availability', 'water_availabity', 'Recommended Loan']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Normalize numerical columns
scaler = StandardScaler()
numerical_cols = ['age', 'annual_income', 'monthly_expenses', 'house_area', 'occupants_count', 'loan_amount', 'young_dependents']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Features and target
X = df[['age', 'annual_income', 'monthly_expenses', 'home_ownership', 'house_area',
        'sanitary_availability', 'water_availabity', 'occupants_count', 'loan_amount', 'young_dependents']]
y = df['Recommended Loan']

# Train/test split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Route to handle PDF file upload, summarization, and translation
@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    file = request.files['pdf_file']
    target_language = request.form.get('language')  # Get target language from frontend
    if file:
        file.save('uploaded_pdf.pdf')  # Save file temporarily
        extracted_text = extract_text_from_pdf('uploaded_pdf.pdf')
        summary = summarize_text(extracted_text)
        translated_summary = translate_text(summary, target_language)  # Translate the summary
        return jsonify({
            'extracted_text': extracted_text,
            'summary': summary,
            'translated_summary': translated_summary
        })
    return jsonify({'error': 'No file uploaded'}), 400


# Define the Product model
class Products(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    sales = db.Column(db.Integer, nullable=False)

# Route to display the form page and add products
@app.route('/add_product', methods=['GET', 'POST'])
def add_product():
    if request.method == 'POST':
        product_name = request.form['product_name']
        sales = int(request.form['sales'])
        
        # Add the product data to the database
        new_product = Products(name=product_name, sales=sales)
        db.session.add(new_product)
        db.session.commit()
        
        # Redirect to the home page after adding the product
        return redirect(url_for('graph'))
    
    return render_template('add_product.html')

@app.route('/delete_product/<int:product_id>', methods=['GET'])
def delete_product(product_id):
    # Find the product by its ID
    product = Products.query.get_or_404(product_id)

    # Delete the product from the database
    db.session.delete(product)
    db.session.commit()

    # Redirect to the home page after deletion
    return redirect(url_for('graph'))



# Route for home page, which includes the graph
@app.route('/graph')
def graph():
    # Ensure app context is active while querying the database
    with app.app_context():
        products = Products.query.all()

        # Create the graph
        product_names = [product.name for product in products]
        product_sales = [product.sales for product in products]
        
        # Create a matplotlib plot
        fig, ax = plt.subplots()
        ax.bar(product_names, product_sales, color='skyblue')

        ax.set_xlabel('Product Name')
        ax.set_ylabel('Number of Items Sold')
        ax.set_title('Product vs Sales')

        # Save the plot to a BytesIO object and convert it to base64
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('Analyse_Profit.html', plot_url=plot_url, products=products)

class Sales(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    month = db.Column(db.String(20), nullable=False, unique=True)
    sales = db.Column(db.Float, nullable=False)

    def __repr__(self):
        return f"Sales('{self.month}', '{self.sales}')"

@app.route('/populate_sales')
def populate_sales():
    # Only populate if the table is empty
    if Sales.query.count() == 0:
        months = [
            ('Jan', 1000),
            ('Feb', 1200),
            ('Mar', 900),
            ('Apr', 1500),
            ('May', 1100),
            ('Jun', 1400),
            ('Jul', 1300),
            ('Aug', 1600),
            ('Sep', 1250),
            ('Oct', 1700),
            ('Nov', 1800),
            ('Dec', 2000)
        ]

        for month, sales in months:
            new_sale = Sales(month=month, sales=sales)
            db.session.add(new_sale)
        
        db.session.commit()
        return jsonify({"message": "Sales data populated successfully!"})

    return jsonify({"message": "Sales table already populated!"})

@app.route('/dashboard')
@login_required
def dashboard():
    # Fetch sales data from the database
    sales_data = Sales.query.all()

    # Prepare data for the chart (months and sales values)
    months = [sale.month for sale in sales_data]
    sales = [sale.sales for sale in sales_data]

    return render_template('index.html', months=months, sales=sales)

@app.route('/get_sales_data')
def get_sales_data():
    # Query the sales data from the database
    sales_data = db.session.query(Sales).all()
    months = [sale.month for sale in sales_data]
    sales = [sale.sales for sale in sales_data]

    # Return the sales data as JSON
    return jsonify({'months': months, 'sales': sales})

@app.route('/growbusiness', methods=['GET', 'POST'])
def growbusiness():
    return render_template('growbusiness.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse form data
        data = request.json

        # Prepare input data
        user_input = [
            float(data['age']),
            float(data['annual_income']),
            float(data['monthly_expenses']),
            1 if data['home_ownership'].lower() == 'yes' else 0,
            float(data['house_area']),
            1 if data['sanitary_availability'].lower() == 'yes' else 0,
            1 if data['water_availability'].lower() == 'yes' else 0,
            int(data['occupants_count']),
            float(data['loan_amount']),
            int(data['young_dependents'])
        ]

        # Scale numerical features
        numerical_features = [user_input[i] for i in range(len(numerical_cols))]
        numerical_features_scaled = scaler.transform([numerical_features])
        user_input_scaled = [
            numerical_features_scaled[0][0],
            numerical_features_scaled[0][1],
            numerical_features_scaled[0][2],
            user_input[3],
            numerical_features_scaled[0][3],
            user_input[5],
            user_input[6],
            numerical_features_scaled[0][4],
            numerical_features_scaled[0][5],
            numerical_features_scaled[0][6]
        ]
        user_input_scaled = [user_input_scaled]

        # Make prediction
        prediction = model.predict(user_input_scaled)
        recommended_loan = label_encoders['Recommended Loan'].inverse_transform(prediction)

        return jsonify({'recommended_loan': recommended_loan[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Ensure that the database tables are created when the app starts
with app.app_context():
    db.create_all()


if __name__ == "__main__":
    app.run(debug=True)