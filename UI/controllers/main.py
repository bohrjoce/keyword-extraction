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
	svd_keywords = []
	cluster_keywords = []
	rake_keywords = []

	f_content = ''
	file_entered = False
	if request.method == 'POST':
		file = request.files['file']
		# print file.filename
		if file:
			f_content = file.read()
			file_entered = True
			rake_tr_keywords = rake_tr.main(f_content)
			svd_keywords = svd(f_content)
			# tokens, data, mapping_back = feature_extract.get_rakeweight_data(f_content)
			# print tokens
			# print data
			# print mapping_back
			# cluser_keywords = kcluster(mapping_back, 5, data, tokens)
			cluster_keywords = getCluster(f_content)
			print cluster_keywords
			rake_keywords = getRake(f_content)
			

	if request.method == 'GET':
		file_entered = False

	print "file entered: " + str(file_entered)
	options = {
		"file_entered": file_entered,
		"fileContent": f_content,
		"rake_keywords": rake_keywords,
		"svd_keywords": svd_keywords,
		"rake_tr_keywords": rake_tr_keywords,
		"cluster_keywords": cluster_keywords
	}

	return render_template("index.html", **options)

