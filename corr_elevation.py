import gpxpy
import gpxpy.gpx
import pandas as pd
import requests
import json
import numpy as np
import time
import matplotlib.pyplot as plt
import streamlit as st
from io import StringIO 


def read_gpx(filename):
    '''Parse gpx file to df'''

    gpx = gpxpy.parse(filename)
    
    points = []
    for segment in gpx.tracks[0].segments:
        for p in segment.points:
            points.append({
                'time': p.time,
                'latitude': p.latitude,
                'longitude': p.longitude,
                'elevation': p.elevation,
            })
    df = pd.DataFrame.from_records(points)

    return df 


def resample_df(df):
    '''Resample initial file to 30sec segmented data'''
    
    df['date'] = pd.to_datetime(df['time'])
    df.set_index('date', inplace=True, drop=True)

    tmp = pd.DataFrame([])
    tmp = df.copy()

    df.drop(['time','elevation'], axis=1, inplace=True)

    df = df.resample('30S').max()
    df.reset_index(inplace=True)
    df = df.dropna()
    df.rename(columns={'date':'time'}, inplace=True)
    return df,tmp 


def openstreet_elevation(df):
    '''Alternate elevation correction function through openstreetmap. Not working well though'''

    # convert lat lon to dicts
    tmp = df[['latitude','longitude']].to_dict('index') 
    data = {"locations":[v for k,v in tmp.items()]}

    url =  'https://api.open-elevation.com/'
    r = requests.post(url+'api/v1/lookup?', data = json.dumps(data) ,headers = {'Content-type': 'application/json', 'Accept': 'application/json'}).json()
    res = r['results']
    elev = np.array([r['elevation'] for r in res])
    df['elevation'] = elev

    return df


def gpxz_elevation(df,api_key):
    '''Basic function to correct elevation data, through gpxz package'''

    BATCH_SIZE = 50  # 

    lats = df.latitude
    lons = df.longitude

    elevations = []
    n_chunks = int(len(lats) // BATCH_SIZE)  + 1
    lat_chunks = np.array_split(lats, n_chunks) 
    lon_chunks = np.array_split(lons, n_chunks)
    
    with st.spinner('Wait for elevation data...'):
        for lat_chunk, lon_chunk in zip(lat_chunks, lon_chunks):
            latlons = '|'.join(f'{lat},{lon}' for lat, lon in zip(lat_chunk, lon_chunk))
            data = {
                'latlons': latlons,
            }
            response = requests.post(
                'https://api.gpxz.io/v1/elevation/points', 
                headers={'x-api-key': api_key},
                data=data,
            )
            response.raise_for_status()
            elevations += [r['elevation'] for r in response.json()['results']]
            time.sleep(1.2)
    st.success('Done!')
    df['elevation'] = elevations
    return df


def convert_df_to_gpx(df):
    '''Convert corrected data to gpx file and save locally'''

    gpx = gpxpy.gpx.GPX()
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(gpx_track)
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)

    for idx in df.index:
        gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(df.loc[idx, 'latitude'], df.loc[idx, 'longitude'], df.loc[idx, 'elevation']))
    with open('new_file.gpx', 'w') as f:
        f.write(gpx.to_xml())

    
def main():
    '''Take a gpx file with wrong elevation data and reform it with correct data from gpxz API'''

    # website title and basic formatting
    html_temp = """ 
    <div style ="background-color:blue;padding:12px"> 
    <h1 style ="color:black;text-align:center;">GPX elevation fix web app</h1> 
    </div> 
    """
    st.markdown(html_temp, unsafe_allow_html = True)

    
    apikeyfile = st.file_uploader("Select the API key file:", type=["txt"])
    if apikeyfile is not None:
        with open(apikeyfile.name, 'rt') as apkf:
            api_key = apkf.read()

        file = st.file_uploader("Please choose a corrupted GPX file:", type=["gpx"])

        if file is not None:
            initial_name = file.name
            df = read_gpx(file)    

            [df, tmp] = resample_df(df)

            arr = np.array(tmp['elevation'])
            ndf = pd.DataFrame([])
            ndf['elevation'] = arr

            # plot initial gpx data
            fig = plt.figure(figsize=[12,7])
            plt.plot(ndf.index, ndf['elevation'])
            plt.title('Raw data of GPX file')
            plt.ylabel('Elevation')
            st.pyplot(fig)

            df = gpxz_elevation(df,api_key)
            # df = openstreet_elevation(df)
            
            # plot corrected gpx data
            fig = plt.figure(figsize=[12,7])
            plt.plot(df.index, df['elevation'], color='tab:purple')
            plt.title('Corrected of GPX file')
            plt.ylabel('Elevation')
            st.pyplot(fig)

            convert_df_to_gpx(df)

            # prompt user to download corrected file
            with open('new_file.gpx') as f:
                st.download_button('Download corrected gpx file', f, file_name='corrected_'+initial_name)
                st.write('File correction completed!')

if __name__=='__main__':
    main()
