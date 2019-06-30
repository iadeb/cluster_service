from flask import Flask, send_file, make_response
from plot import do_plot
app = Flask(__name__)

@app.route('/plots/breast_cancer_data/correlation_matrix', methods=['GET'])
def correlation_matrix():
    bytes_obj = do_plot()
    
    return send_file(bytes_obj,
                     attachment_filename='plot.png',
                     mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=False)