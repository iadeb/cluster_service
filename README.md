# cluster_service

"NOTE: DEVICE_TYPE can easily be replaced with/ extended to Age_group, Gender, Country, Region, function flow doesnt change!"










# SELECT CAMPAIGN:

'/select_camp/<string:camp_type>/<string:camp_name>/'
'/select_camp/<string:camp_type>/<string:camp_name>/<string:objective>/

Example:
http://localhost:5000/select_camp/new_device/Edcil/
http://localhost:5000/select_camp/new_device/Edcil/LEAD_GENERATION/ 
Response {json: Success}

camp_type: By "impression_device, age_group, gender, country, region",
will change to database names on connection, for now, its csv names "new_device.csv".



DISPLAY CLUSTER:
http://localhost:5000/cluster

Response {cluster img}









# query DEVICE_TYPE BY CLUSTER:

Visualize:
'/device_graph/<int:clusid>/'

Example:
http://localhost:5000/device_graph/1/


DataFrame:
'/device_data/<int:clusid>/'


Example:
http://localhost:5000/device_data/1/










# query DAY_OF_WEEK by DEVICE_TYPE

Visualize:
'/day_graph/<int:clusid>/<string:device_type>'


Example:
http://localhost:5000/day_graph/1/android_smartphone




Dataframe:
'/day_data/<int:clusid>/<string:device_type>'

Example:
http://localhost:5000/day_data/1/android_smartphone








# query HOUR_OF_DAY by DEVICE_TYPE

Visualize:
'/hour_graph/<int:clusid>/<string:device_type>/<string:week_day>'

Example:
http://localhost:5000/hour_graph/1/android_smartphone/Thursday


Dataframe:
'/hour_data/<int:clusid>/<string:device_type>/<string:week_day>'

Example:
http://localhost:5000/hour_data/1/android_smartphone/Thursday










