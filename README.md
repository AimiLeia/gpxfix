# Trail elevation fix 
#### This project was created to address the problem of false elevation data in gpx files. When recording a trail/route on a mobile phone, the GPS accuracy depends on many factors, and sometimes there might be misrecordings in the final track. 

#### For that purpose, I created a simple web application using Streamlit, to upload the original gpx file, fix the elevation data and return the correct file back to the user.

#### The approach is quite straightforward. First, the original data is resampled to 30sec segments, then for each pair of lat-lon, a request is created through an Elevation API (https://www.gpxz.io) to get the respective elevation for each location. Then, the new data is transformed back to gpx file and the user can download it to their computer. 
#### Note: The user must posess an API key from the aforementioned API in order to request the elevation data.

# Usage
#### The web app opens if you run the following command in terminal:
```Python
streamlit run fix_elevation.py 
```
#### The app prompts the user to upload first an API key file, then a false-elevation gpx file. A plot of the original elevation data is created, and then after waiting for a few seconds, the correct data is plotted on a new figure for comparison. Finally, the user can download the new gpx file locally.

# Demo run
#### If you want to see a demonstration of the app, you can run demo.py instead of fix_elevation.py. A test gpx file ('test_erimanthos.gpx') is immediately utilized as test case to demonstrate the functionality of the app with the respective figures before and after the elevation fix.