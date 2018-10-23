from flask import Flask
 
app = Flask(__name__)
 
@app.route('/')
def hello_whale():
    return 'give me whales or give me death'


 
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
