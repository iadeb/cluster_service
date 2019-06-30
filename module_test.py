import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from IPython.display import display # Allows the use of display() for DataFrames
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# Pretty display for notebooks
#%matplotlib inline
#import os
#print(os.listdir("../input"))


###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
###########################################


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

# Apply PCA by fitting the good data with the same number of dimensions as features
from sklearn.decomposition import PCA
import io
#import StringIO
import base64



def pca_results(good_data, pca):
	'''
	Create a DataFrame of the PCA results
	Includes dimension feature weights and explained variance
	Visualizes the PCA results
	'''

	# Dimension indexing
	dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

	# PCA components
	components = pd.DataFrame(np.round(pca.components_, 4), columns = list(good_data.keys()))
	components.index = dimensions

	# PCA explained variance
	ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
	variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
	variance_ratios.index = dimensions

	# Create a bar plot visualization
	fig, ax = plt.subplots(figsize = (14,8))

	# Plot the feature weights as a function of the components
	components.plot(ax = ax, kind = 'bar');
	ax.set_ylabel("Feature Weights")
	ax.set_xticklabels(dimensions, rotation=0)


	# Display the explained variance ratios
	for i, ev in enumerate(pca.explained_variance_ratio_):
		ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))

	# Return a concatenated DataFrame
	return pd.concat([variance_ratios, components], axis = 1)

def cluster_results(reduced_data, preds, centers):
	'''
	Visualizes the PCA-reduced cluster data in two dimensions
	Adds cues for cluster centers and student-selected sample data
	'''

	predictions = pd.DataFrame(preds, columns = ['Cluster'])
	plot_data = pd.concat([predictions, reduced_data], axis = 1)

	# Generate the cluster plot
	fig, ax = plt.subplots(figsize = (14,8))

	# Color map
	cmap = cm.get_cmap('gist_rainbow')

	# Color the points based on assigned cluster
	for i, cluster in plot_data.groupby('Cluster'):   
	    cluster.plot(ax = ax, kind = 'scatter', x = 'Dimension 1', y = 'Dimension 2', \
	                 color = cmap((i)*1.0/(len(centers)-1)), label = 'Cluster %i'%(i), s=30);



	# Plot centers with indicators
	for i, c in enumerate(centers):
	    ax.scatter(x = c[0], y = c[1], color = 'white', edgecolors = 'black', \
	               alpha = 1, linewidth = 2, marker = 'o', s=200);
	    ax.scatter(x = c[0], y = c[1], marker='$%d$'%(i), alpha = 1, s=100);

	# Plot transformed sample points 
	#ax.scatter(x = pca_samples[:,0], y = pca_samples[:,1], \
	 #          s = 150, linewidth = 4, color = 'black', marker = 'x');

	# Set plot title
	ax.set_title("Cluster Learning on PCA-Reduced Data - Centroids Marked by Number\nTransformed Sample Data Marked by Black Cross");

	#plt.show()

	bytes_image = io.BytesIO()
	plt.savefig(bytes_image, format='png')
	bytes_image.seek(0)


	return bytes_image



def biplot(good_data, reduced_data, pca):
    '''
    Produce a biplot that shows a scatterplot of the reduced
    data and the projections of the original features.
    
    good_data: original data, before transformation.
               Needs to be a pandas dataframe with valid column names
    reduced_data: the reduced data (the first two dimensions are plotted)
    pca: pca object that contains the components_ attribute

    return: a matplotlib AxesSubplot object (for any additional customization)
    
    This procedure is inspired by the script:
    https://github.com/teddyroland/python-biplot
    '''

    fig, ax = plt.subplots(figsize = (14,8))
    # scatterplot of the reduced data    
    ax.scatter(x=reduced_data.loc[:, 'Dimension 1'], y=reduced_data.loc[:, 'Dimension 2'], 
        facecolors='b', edgecolors='b', s=70, alpha=0.5)
    
    feature_vectors = pca.components_.T

    # we use scaling factors to make the arrows easier to see
    arrow_size, text_pos = 7.0, 8.0,

    # projections of the original features
    for i, v in enumerate(feature_vectors):
        ax.arrow(0, 0, arrow_size*v[0], arrow_size*v[1], 
                  head_width=0.2, head_length=0.2, linewidth=2, color='red')
        ax.text(v[0]*text_pos, v[1]*text_pos, good_data.columns[i], color='black', 
                 ha='center', va='center', fontsize=18)

    ax.set_xlabel("Dimension 1", fontsize=14)
    ax.set_ylabel("Dimension 2", fontsize=14)
    ax.set_title("PC plane with original feature projections.", fontsize=16);
    return ax
    

def channel_results(reduced_data, outliers, pca_samples):
	'''
	Visualizes the PCA-reduced cluster data in two dimensions using the full dataset
	Data is labeled by "Channel" and cues added for student-selected sample data
	'''

	# Check that the dataset is loadable
	try:
	    full_data = pd.read_csv("new_device.csv")
	except:
	    print("Dataset could not be loaded. Is the file missing?")       
	    return False

	# Create the Channel DataFrame
	channel = pd.DataFrame(full_data['Channel'], columns = ['Channel'])
	channel = channel.drop(channel.index[outliers]).reset_index(drop = True)
	labeled = pd.concat([reduced_data, channel], axis = 1)
	
	# Generate the cluster plot
	fig, ax = plt.subplots(figsize = (14,8))

	# Color map
	cmap = cm.get_cmap('gist_rainbow')

	# Color the points based on assigned Channel
	labels = ['Hotel/Restaurant/Cafe', 'Retailer']
	grouped = labeled.groupby('Channel')
	for i, channel in grouped:   
	    channel.plot(ax = ax, kind = 'scatter', x = 'Dimension 1', y = 'Dimension 2', \
	                 color = cmap((i-1)*1.0/2), label = labels[i-1], s=30);
	    
	# Plot transformed sample points   
	for i, sample in enumerate(pca_samples):
		ax.scatter(x = sample[0], y = sample[1], \
	           s = 200, linewidth = 3, color = 'black', marker = 'o', facecolors = 'none');
		ax.scatter(x = sample[0]+0.25, y = sample[1]+0.3, marker='$%d$'%(i), alpha = 1, s=125);

	# Set plot title
	ax.set_title("PCA-Reduced Data Labeled by 'Channel'\nTransformed Sample Data Circled");



#########################################################################################################################################################





# Applying K means
def sil_coeff(no_clusters, reduced_data):
    # Apply your clustering algorithm of choice to the reduced data 
    clusterer_1 = KMeans(n_clusters=no_clusters, random_state=0 )
    clusterer_1.fit(reduced_data)
    
    # Predict the cluster for each data point
    preds_1 = clusterer_1.predict(reduced_data)
    
    # Find the cluster centers
    centers_1 = clusterer_1.cluster_centers_
    
    # Predict the cluster for each transformed sample data point
    #sample_preds_1 = clusterer_1.predict(pca_samples)
    
    # Calculate the mean silhouette coefficient for the number of clusters chosen
    score = silhouette_score(reduced_data, preds_1)
    
    print("silhouette coefficient for `{}` clusters => {:.4f}".format(no_clusters, score))
    



def all_run( csv_name, camp_name, objective):
	parse_date = lambda val : pd.datetime.strptime(val, '%Y-%m-%d %H:%M:%S')

	#with gzip.open('train.gz') as f:
	train = pd.read_csv(csv_name+".csv", parse_dates = ['date_stats_start'], date_parser = parse_date, usecols =["campaign_id", "date_stats_start", "timestamp", "spend", "unique_clicks", "impression_device", "account_name", "objective"])


	#train.head()

	tempo_data = train.loc[train['account_name'] == camp_name]
	# tempoo_data = tempo_data.loc[tempo_data['objective'] == 'LEAD_GENERATION']
	# tempooo_data = tempo_data.loc[tempo_data['objective'] == 'LINK_CLICKS']

	# frames = [tempoo_data, tempooo_data]

	# temp_data = pd.concat(frames)

	if objective == None:
		temp_data = tempo_data	
	else:
		tempoo_data = tempo_data.loc[tempo_data['objective'] == objective]
		temp_data = tempoo_data


	#should always be (spends+timestamp+impression_device) combination to fit on (unique_clicks)


	#camp_data = temp_data.dropna()

	#spends_data = temp_data2.dropna()

	# label_encoder = LabelEncoder()
	# label_encoder = label_encoder.fit(temp_data.impression_device)
	# label_encoded_y = label_encoder.transform(temp_data.impression_device)

	# temp_data['impression_device_encoded'] = label_encoded_y

	#spends_data = temp_data2.dropna()
	temp_data['hour_of_day'] = temp_data.date_stats_start.apply(lambda x: x.hour)
	temp_data['day_of_week'] = temp_data['date_stats_start'].apply(lambda val: val.weekday_name)



	temp_backup = temp_data

	# label_encoder = LabelEncoder()
	# label_encoder = label_encoder.fit(temp_data.day_of_week)
	# label_encoded_y = label_encoder.transform(temp_data.day_of_week)

	# temp_data['day_of_week_encoded'] = label_encoded_y

	temp_device = pd.get_dummies(temp_data['impression_device'], prefix = 'device')
	temp_week = pd.get_dummies(temp_data['day_of_week'], prefix = 'week')
	temp_hour = pd.get_dummies(temp_data['hour_of_day'], prefix = 'hour')

	temp_data = pd.concat([temp_data, temp_device], axis=1)
	temp_data = pd.concat([temp_data, temp_week], axis=1)
	temp_data = pd.concat([temp_data, temp_hour], axis=1)

	temp_data.drop(['campaign_id', 'date_stats_start', 'timestamp', 'objective', 'account_name', 'impression_device', 'day_of_week', 'hour_of_day'], axis = 1, inplace = True)

	data = temp_data.dropna()


	data = data.reset_index()
	data.drop(['index'], axis = 1, inplace = True)





	pca = PCA(n_components=2).fit(data)
	reduced_data = pca.transform(data)
	reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])
	#reduced_data



	clusters_range = range(2,11)
	for i in clusters_range:
	    sil_coeff(i, reduced_data)





	# 1. K Means Visualization 
	# Display the results of the clustering from implementation for 2 clusters
	clusterer = KMeans(n_clusters = 6)
	clusterer.fit(reduced_data)
	preds = clusterer.predict(reduced_data)
	centers = clusterer.cluster_centers_
	#sample_preds = clusterer.predict(pca_samples)

	cluster_plot = cluster_results(reduced_data, preds, centers)
	#cluster_results(reduced_data, preds, centers)



	predictions = pd.DataFrame(preds, columns = ['Cluster'])





	temp_backup.drop(['campaign_id', 'date_stats_start', 'timestamp', 'objective', 'account_name'], axis = 1, inplace = True)

	init_data = temp_backup.dropna()
	init_data = init_data.reset_index()
	init_data.drop(['index'], axis = 1, inplace = True)

	cluster_data = pd.concat([predictions, init_data], axis = 1)

	




	return cluster_plot, cluster_data
	#return cluster_data




#cluster_data




# clus1 = cluster_data[cluster_data['Cluster'] == 1]
# clus2 = cluster_data[cluster_data['Cluster'] == 2]
# clus3 = cluster_data[cluster_data['Cluster'] == 3]
# clus4 = cluster_data[cluster_data['Cluster'] == 4]
# clus5 = cluster_data[cluster_data['Cluster'] == 5]
# clus6 = cluster_data[cluster_data['Cluster'] == 0]
# clus6['Cluster'] = clus6.Cluster.apply(lambda x: x+6)





###visualize cluster
#print(clus1)
#print(clus2)
#print(clus3)
#print(clus4)
#print(clus5)
#print(clus6)




#print(clus4)

# print(len(clus4))


# if len(clus4)>3:
# 	clus4_ratio = clus4.groupby(['day_of_week']).sum().reset_index()
# 	clus4_ratio['ratio'] = clus4_ratio['unique_clicks']/clus4_ratio['spend']
# 	display(clus4_ratio)
# 	sns.barplot(x="day_of_week", y="ratio", data=clus4_ratio)
# 	plt.show()


def device_process(cluster_data, cluster_num):


	clus = cluster_data[cluster_data['Cluster'] == cluster_num]

	device_image = 0
	plot_url = ''

	if len(clus)>1:
		clus_ratio = clus.groupby(['impression_device']).sum().reset_index()
		clus_ratio['ratio'] = clus_ratio['unique_clicks']/clus_ratio['spend']
		print(clus_ratio)
		#sns.barplot(x="impression_device", y="ratio", data=clus_ratio)



		#ax = clus_ratio[['ratio']].plot(kind='bar', title ="V comp", figsize=(15, 10), legend=True, fontsize=12)
		ax = clus_ratio.plot(x='impression_device', y='ratio', kind='bar', figsize=(15, 10), fontsize=9)
		ax.set_xlabel("impression_device", fontsize=12)
		ax.set_ylabel("ratio", fontsize=12)
		#plt.show()
		#plt.show()

		device_image = io.BytesIO()
		plt.savefig(device_image, format='png')
		device_image.seek(0)

		#plot_url = base64.b64encode(device_image.getvalue())
	else:
		device_image = None

	return device_image
	#return plot_url




def device_data(cluster_data, cluster_num):
	
	clus_ratio = pd.DataFrame()

	clus = cluster_data[cluster_data['Cluster'] == cluster_num]

	if len(clus)>1:
		clus_ratio = clus.groupby(['impression_device']).sum().reset_index()
		clus_ratio['ratio'] = clus_ratio['unique_clicks']/clus_ratio['spend']
		#print(clus_ratio)
	
	return clus_ratio


# def cluster_check(cluster_data, cluster_num):

# 	device_plot = call_device(cluster_data, cluster_num)

# 	return device_plot



def day_of_week_process(cluster_data, cluster_num, device_type):


	day_image = 0


	clus = cluster_data[cluster_data['Cluster'] == cluster_num]
	clus_device = clus.loc[(clus['impression_device'] == device_type)]

	if len(clus)>1:
		
		clus_ratio = clus_device.groupby(['day_of_week']).sum().reset_index()
		clus_ratio['ratio'] = clus_ratio['unique_clicks']/clus_ratio['spend']
		print(clus_ratio)

		ax = clus_ratio.plot(x='day_of_week', y='ratio', kind='bar', figsize=(15, 10), fontsize=9)
		ax.set_xlabel("day_of_week", fontsize=12)
		ax.set_ylabel("ratio", fontsize=12)

		#sns.barplot(x="day_of_week", y="ratio", data=clus_ratio)
		#plt.show()

		day_image = io.BytesIO()
		plt.savefig(day_image, format='png')
		day_image.seek(0)

	else:
		day_image = None

	return day_image



def day_of_week_data(cluster_data, cluster_num, device_type):
	
	clus_ratio = pd.DataFrame()

	clus = cluster_data[cluster_data['Cluster'] == cluster_num]
	clus_device = clus.loc[(clus['impression_device'] == device_type)]

	if len(clus)>1:
		clus_ratio = clus_device.groupby(['day_of_week']).sum().reset_index()
		clus_ratio['ratio'] = clus_ratio['unique_clicks']/clus_ratio['spend']
		#print(clus_ratio)
	
	return clus_ratio





#call_day_week('android_smartphone')


def hour_of_day_process(cluster_data, cluster_num, device_type, day_of_week):
	

	#try:
	hour_image = 0
	clus = cluster_data[cluster_data['Cluster'] == cluster_num]
	clus_device = clus.loc[(clus['impression_device'] == device_type)]
	clus_day_week = clus_device.loc[clus_device['day_of_week'] == day_of_week]
	
	if len(clus_day_week)>1:
		
		# clus4_sun = clus4_device.loc[clus4_device['day_of_week'] == 'Sunday']

		# clus4_mon = clus4_device.loc[clus4_device['day_of_week'] == 'Monday']
		# clus4_tue = clus4_device.loc[clus4_device['day_of_week'] == 'Tuesday']


		clus_ratio = clus_day_week.groupby(['hour_of_day']).sum().reset_index()
		clus_ratio['ratio'] = clus_ratio['unique_clicks']/clus_ratio['spend']
		print(clus_ratio)

		ax = clus_ratio.plot(x='hour_of_day', y='ratio', kind='bar', figsize=(15, 10), fontsize=9)
		ax.set_xlabel("hour_of_day", fontsize=12)
		ax.set_ylabel("ratio", fontsize=12)

		#sns.barplot(x="day_of_week", y="ratio", data=clus_ratio)
		#plt.show()

		hour_image = io.BytesIO()
		plt.savefig(hour_image, format='png')
		hour_image.seek(0)

	else:
		hour_image = None

	return hour_image
		
    


def hour_of_day_data(cluster_data, cluster_num, device_type, day_of_week):
	
	clus_ratio = pd.DataFrame()

	clus = cluster_data[cluster_data['Cluster'] == cluster_num]
	clus_device = clus.loc[(clus['impression_device'] == device_type)]
	clus_day_week = clus_device.loc[clus_device['day_of_week'] == day_of_week]

	if len(clus)>1:
		clus_ratio = clus_day_week.groupby(['hour_of_day']).sum().reset_index()
		clus_ratio['ratio'] = clus_ratio['unique_clicks']/clus_ratio['spend']
		#print(clus_ratio)
	
	return clus_ratio
	



	# clus4_sun = clus4_sun.groupby(['hour_of_day']).sum().reset_index()
	# clus4_sun['ratio'] = clus4_sun['unique_clicks']/clus4_sun['spend']
	# display(clus4_sun)


	# clus4_mon = clus4_mon.groupby(['hour_of_day']).sum().reset_index()
	# clus4_mon['ratio'] = clus4_mon['unique_clicks']/clus4_mon['spend']
	# display(clus4_mon)


	# clus4_tue = clus4_tue.groupby(['hour_of_day']).sum().reset_index()
	# clus4_tue['ratio'] = clus4_tue['unique_clicks']/clus4_tue['spend']
	# display(clus4_tue)

def call_hour_of_day_process(cluster_data, device_type, day_of_week):
	#impression_device = 'android_smartphone'
	for i in range(0, 7):
		clus = cluster_data[cluster_data['Cluster'] == i]
		print(clus)
		hour_of_day_process(clus, device_type, day_of_week)


#call_hour_of_day_process("android_smartphone", "Sunday")



