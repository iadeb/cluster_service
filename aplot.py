from flask import Flask, send_file, make_response, request, render_template, jsonify
import sys
import module_test
app = Flask(__name__)



@app.route('/select_camp/<string:camp_type>/<string:camp_name>/')
@app.route('/select_camp/<string:camp_type>/<string:camp_name>/<string:objective>/')
def main_func(camp_type, camp_name, objective=None):

	global clus_bytes_obj
	global clus_data

	clus_bytes_obj, clus_data = module_test.all_run(camp_type, camp_name, objective)

	return jsonify("success")

	

	#return redirect('/cluster')


@app.route('/cluster')
def correlation_matrix():
    


 # 	clus_num = request.args.get('cluster_num', default=9)
	# kmax = request.args.get('kmax', default=8)
	# accuracy = request.args.get('acc', default=0.80)

	

	# if clus_num == 9:
	# 	bytes_obj = clus_bytes_obj

	# else:
	# 	bytes_obj = device_bytes_obj
    
    return send_file(clus_bytes_obj,
                     attachment_filename='plot.png',
                     mimetype='image/png')


@app.route('/device_graph/<int:clusid>/')
def device_print(clusid):

	#clus_num = request.args.get('cluster_num', default=0)
	device_plot = module_test.device_process(clus_data, clusid)
	#print(device_plot)
    
	return send_file(device_plot, attachment_filename='plot.png', mimetype='image/png')
	#return device_plot
	#return render_template('test.html', plot_url=plot_url)



@app.route('/device_data/<int:clusid>/')
def device_dataframe(clusid):

	#clus_num = request.args.get('cluster_num', default=0)
	datafr = module_test.device_data(clus_data, clusid)
	print(datafr)
    
	#return send_file(device_plot, attachment_filename='plot.png', mimetype='image/png')
	#return device_plot
	#return render_template('test.html', plot_url=plot_url)
	#return jsonify(datafr)
	return render_template("test.html",  data=datafr.to_html())






@app.route('/day_graph/<int:clusid>/<string:device_type>')
def day_print(clusid, device_type):
	#clus_num = request.args.get('cluster_num', default=0)
	device_plot = module_test.day_of_week_process(clus_data, clusid, device_type)
	#print(device_plot)
    
	return send_file(device_plot, attachment_filename='plot.png', mimetype='image/png')
	#return device_plot
	#return render_template('test.html', plot_url=plot_url)




@app.route('/day_data/<int:clusid>/<string:device_type>')
def day_dataframe(clusid, device_type):
	#clus_num = request.args.get('cluster_num', default=0)
	datafr = module_test.day_of_week_data(clus_data, clusid, device_type)
	print(datafr)
    
	return render_template("test.html",  data=datafr.to_html())




@app.route('/hour_graph/<int:clusid>/<string:device_type>/<string:week_day>')
def hour_print(clusid, device_type, week_day):
	#clus_num = request.args.get('cluster_num', default=0)
	device_plot = module_test.hour_of_day_process(clus_data, clusid, device_type, week_day)
	#print(device_plot)
    
	return send_file(device_plot, attachment_filename='plot.png', mimetype='image/png')
	#return device_plot
	#return render_template('test.html', plot_url=plot_url)



@app.route('/hour_data/<int:clusid>/<string:device_type>/<string:week_day>')
def hour_dataframe(clusid, device_type, week_day):
	#clus_num = request.args.get('cluster_num', default=0)
	datafr = module_test.hour_of_day_data(clus_data, clusid, device_type, week_day)
	print(datafr)
    
	return render_template("test.html",  data=datafr.to_html())





if __name__ == '__main__':
    app.run(debug=False)