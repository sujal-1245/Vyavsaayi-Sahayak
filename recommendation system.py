from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Define the Product model
class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    sales = db.Column(db.Integer, nullable=False)

# Route to display the form page and add products
@app.route('/add_product', methods=['GET', 'POST'])
def add_product():
    if request.method == 'POST':
        product_name = request.form.get('product_name')
        sales = request.form.get('sales')
        
        if not product_name or not sales.isdigit():
            return redirect(url_for('add_product'))
        
        new_product = Product(name=product_name, sales=int(sales))
        db.session.add(new_product)
        db.session.commit()
        
        return redirect(url_for('index'))
    
    return render_template('add_product.html')

# Route to delete a product
@app.route('/delete_product/<int:product_id>', methods=['POST'])
def delete_product(product_id):
    product = Product.query.get_or_404(product_id)
    db.session.delete(product)
    db.session.commit()
    return redirect(url_for('index'))

# Route to edit a product
@app.route('/edit_product/<int:product_id>', methods=['GET', 'POST'])
def edit_product(product_id):
    product = Product.query.get_or_404(product_id)
    
    if request.method == 'POST':
        product_name = request.form.get('product_name')
        sales = request.form.get('sales')
        
        if not product_name or not sales.isdigit():
            return redirect(url_for('edit_product', product_id=product_id))
        
        product.name = product_name
        product.sales = int(sales)
        db.session.commit()
        return redirect(url_for('index'))
    
    return render_template('edit_product.html', product=product)

# Route for home page, which includes the graph
@app.route('/')
def index():
    products = Product.query.all()
    product_names = [product.name for product in products]
    product_sales = [product.sales for product in products]
    
    fig, ax = plt.subplots()
    ax.bar(product_names, product_sales, color='skyblue')
    ax.set_xlabel('Product Name')
    ax.set_ylabel('Number of Items Sold')
    ax.set_title('Product vs Sales')
    
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    
    return render_template('index.html', plot_url=plot_url, products=products)

# Ensure that the database tables are created when the app starts
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)
