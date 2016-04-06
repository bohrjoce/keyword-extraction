from flask import *
from werkzeug import secure_filename
from methods import *

main = Blueprint('main', __name__, template_folder='templates')
ALLOWED_EXTENSIONS = set(['txt'])

def allowed_file(filename):
    return '.' in filename and \
    	filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@main.route('/', methods=['POST', 'GET'])
def main_route():
	rake_tr_keywords = []

	f_content = ''
	file_entered = False
	if request.method == 'POST':
		file = request.files['file']
		if file:
			f_content = file.read()
			file_entered = True
			rake_tr_keywords = rake_tr.main(f_content)

	options = {
		"fileEntered": file_entered,
		"fileContent": f_content,
		"rake_tr_keywords": rake_tr_keywords
	}

	return render_template("index.html", **options)

