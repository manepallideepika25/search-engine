
from flask import Flask, render_template, request
from search import SearchEngine

app = Flask(__name__)
search_engine = SearchEngine()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    query = request.args.get('q', '')
    results = search_engine.search(query)
    return render_template('results.html', results=results, query=query)

if __name__ == '__main__':
    app.run(debug=True)
