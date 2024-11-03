# Importing Needed Libraries
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import seaborn as sns
import plotly
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.io as pio

# Setting Default Theme for plotly graphs
pio.templates.default = "simple_white"

# Reading data from GitHub repository
df = pd.read_csv("price.csv")

# Setting page width to wide
st.set_page_config(layout="wide")

# Sidebar Menu
with st.sidebar:
    selected = option_menu(menu_title=None,
                           options=["Home", "Data Overview", "Exploratory Data Analysis", "In-depth Analysis", "Price Check"],
                           icons=["house", "bar-chart", "book", "bell", "bag-check"],
                           menu_icon="cast",
                           default_index=0,
                           styles={
                               "container": {"padding": "0!important"},
                               "icon": {"color": "#grey"},
                               "nav-link": {
                                   "font-size": "15px",
                                   "text-align": "left",
                                   "margin": "0px", "--hover-color": "#eee"},
                               "nav-link-selected": {"background-color": "#fed8b1"},
                           })

# Home Page
if selected == "Home":
    st.markdown("<h1 style='text-align: center; color: #fccc9a;'>Skyline Bookings</h1>", unsafe_allow_html=True)
    st.image("https://makeflycheap.in/wp-content/uploads/2020/02/dd914c6cca076f8cebb463a81e73e7e5.jpg")
    st.subheader("Book Your Tickets Across India with the Best Prices!")

# Data Overview Page
if selected == "Data Overview":
    uploaded_file = st.file_uploader('', type=["csv"])
    if uploaded_file is not None:
        @st.cache
        def load_csv():
            csv = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
            return csv
        df = load_csv()
    st.header('Input DataFrame')
    st.write(df)
    st.write('---')

    with st.expander('Know more about the data!'):
        st.write("""
            The dataframe above shows a flight booking dataset collected from "Ease My Trip" website.  \n
            The flight travel data is between the top 6 metro cities in India.  \n
            It includes 300261 datapoints and 11 features.  \n
            Duration: A continuous feature that displays the overall amount of time it takes to travel between cities in hours.  \n
            Days Left: This is a derived characteristic that is calculated by subtracting the trip date by the booking date.  \n
        """)

# Exploratory Page
if selected == "Exploratory Data Analysis":
    # To set each KPI in different column
    col1, col2, col3 = st.columns(3)
    # KPIs
    col1.metric("Number of Studied Flights", "300261")
    col2.metric("Highest Ticket Price", "123071 INR")
    col3.metric("Unique Airlines", "6")
    # To count the number of flights per Airline
    col1, col2 = st.columns(2)
    with col1:
        dff = df.groupby("Airline").count()
        df_groupby = dff.reset_index()
        figure6 = px.bar(df_groupby, x="Airline", y="SourceCity", color_continuous_scale='RdBu', title="Amount of Flights per Airline")
        figure6.update_layout(xaxis_title=None, yaxis_title=None)
        figure6.update_xaxes(showgrid=False, zeroline=False)
        figure6.update_yaxes(showgrid=False, showticklabels=True)
        st.plotly_chart(figure6, use_container_width=True)
    with col2:
        # Checking if Price changes by number of Stops
        dfff = df.groupby("Stops")["Price"].mean()
        df__groupby = dfff.reset_index()
        figure7 = px.bar(df__groupby, x="Stops", y="Price", color="Price", color_continuous_scale='RdBu', title="Average Price according to Stops")
        figure7.update_layout(xaxis_title=None, yaxis_title=None)
        figure7.update_xaxes(showgrid=False, zeroline=False)
        figure7.update_yaxes(showgrid=False, showticklabels=True)
        st.plotly_chart(figure7, use_container_width=True)

    # Checking if Price changes with duration Left
    dfairlines = df.groupby('DaysLeft')['Price'].mean()
    df___groupby = dfairlines.reset_index()
    figure8 = px.bar(df___groupby, x="DaysLeft", y="Price", color="Price", color_continuous_scale='RdBu', title="Average Price according to Days Left")
    figure8.update_layout(xaxis_title=None, yaxis_title=None)
    figure8.update_xaxes(showgrid=False, zeroline=False)
    figure8.update_yaxes(showgrid=False, showticklabels=True)
    st.plotly_chart(figure8, use_container_width=True)
    # To check the distribution of flights between classes and airlines
    figure2 = px.treemap(df, path=['Airline', 'Class'], values=df.nunique(axis=1), color='Price', color_continuous_scale='RdBu', title="Ticket Type per Airline")
    figure2.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    st.plotly_chart(figure2, use_container_width=True)

# Analysis Page
if selected == "In-depth Analysis":
    col1, col2 = st.columns(2)
    figure3, ax = plt.subplots()

    # To check the prices across airlines
    figure3 = sns.catplot(y="Price", x="Airline", data=df.sort_values("Price", ascending=False), kind="boxen", height=6, aspect=3, ax=ax)
    st.pyplot(figure3)

    with col1:
        # To check the prices across destination cities
        figure5, axx = plt.subplots()
        figure5 = sns.catplot(y="Price", x="DestinationCity", data=df.sort_values("Price", ascending=False), kind="boxen", height=4, aspect=3, ax=axx)
        st.pyplot(figure5)
    # To check prices across Departure times
    figure10 = px.box(df, x="DepartureTime", y="Price", title="Price according to Departure Time")
    figure10.update_layout(xaxis_title=None, yaxis_title=None)
    figure10.update_xaxes(showgrid=False, zeroline=False)
    figure10.update_yaxes(showgrid=False, showticklabels=True)
    st.plotly_chart(figure10, use_container_width=True)

    with col2:
        # To check the prices across source cities
        figure4, axx = plt.subplots()
        figure4 = sns.catplot(y="Price", x="SourceCity", data=df.sort_values("Price", ascending=False), kind="boxen", height=4, aspect=3, ax=axx)
        st.pyplot(figure4)

# ML Page
if selected == "Price Check":
    # Linear Regression Model to predict approximate price of tickets based on variables that the user inputs
    def main():
        # Source City
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.subheader("Source City")
            source = st.selectbox("from India", ['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', "Mumbai"])
            st.write("You are traveling from ", source)
        if source == "Bangalore":
            source_inp = 0
        elif source == "Chennai":
            source_inp = 1
        elif source == "Delhi":
            source_inp = 2
        elif source == "Hyderabad":
            source_inp = 3
        elif source == "Kolkata":
            source_inp = 4
        elif source == "Mumbai":
            source_inp = 5

        # Destination City
        with col2:
            st.subheader("Destination City")
            dest = st.selectbox("to India", ['Bangalore', "Chennai", 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai'])
            st.write("You are traveling towards ", dest)
        if dest == "Bangalore":
            dest_inp = 0
        elif dest == "Chennai":
            dest_inp = 1
        elif dest == "Hyderabad":
            dest_inp = 2
        elif dest == "Delhi":
            dest_inp = 3
        elif dest == "Kolkata":
            dest_inp = 4
        elif dest == "Mumbai":
            dest_inp = 5

        # Airline
        with col3:
            st.subheader("Select Airline")
            airline = st.selectbox("", ["AirAsia", "GOFIRST", "Air India", "Indigo", "SpiceJet", "Vistara"])
            st.write("You chose ", airline, " Airline")
        if airline == "AirAsia":
            air_inp = 1
        elif airline == "GOFIRST":
            air_inp = 2
        elif airline == "Air India":
            air_inp = 3
        elif airline == "Indigo":
            air_inp = 4
        elif airline == "SpiceJet":
            air_inp = 5
        elif airline == "Vistara":
            air_inp = 6

        # Arrival Time
        st.subheader("Arrival Time")
        arrival = st.selectbox("", ['Morning', 'Afternoon', 'Evening', 'Night'])
        st.write("You will arrive in the ", arrival)

        # Departure Time
        st.subheader("Departure Time")
        departure = st.selectbox("Departure Time", ['Morning', 'Afternoon', 'Evening'], key="departure_time")

        st.write("You will leave in the ", departure)

        # Ticket Class
        st.subheader("Class of Ticket")
        ticketclass = st.selectbox("", ["Economy", "Business", "First"])
        st.write("You chose ", ticketclass)

        # Number of Stops
        st.subheader("Number of Stops")
        stop = st.selectbox("", [0, 1, 2, 3, 4])
        st.write("You chose ", stop, " stops")

        # Duration of Flight
        st.subheader("Duration of Flight (hours)")
        dur = st.number_input("", min_value=1, max_value=24, value=1, step=1)
        st.write("The flight duration is ", dur, " hours")

        # Days Until Flight
        st.subheader("Days Until Flight")
        day = st.number_input("", min_value=1, max_value=90, value=1, step=1)
        st.write("You are booking the flight for ", day, " days from now")

        # Load the model
        with open('linearmodel.pkl', 'rb') as f:
            lrmodel = pickle.load(f)

        # Button to Check Price
        if st.checkbox("Check Your Ticket Price"):
            par = [[source_inp, dest_inp, air_inp, arrival, departure, ticketclass, stop, dur, day]]
            pred = lrmodel.predict(par)
            for i in pred:
                st.write("Your Fare Price is : ", round(i, 3), "INR")
                st.write("*Happy and Safe Journey ...*")

            # Function to save user inputs to a text file
            def save_inputs_to_file(inputs):
                with open("user_inputs.txt", "a") as file:  # Open the file in append mode
                    for key, value in inputs.items():
                        file.write(f"{key}: {value}\n")
                    file.write("\n")  # Add a newline for separation between entries

            # Store user inputs in a dictionary
            user_inputs = {
                "Source City": source,
                "Destination City": dest,
                "Selected Airline": airline,
                "Arrival Time": arrival,
                "Departure Time": departure,
                "Class of Ticket": ticketclass,
                "Number of Stops": stop,
                "Duration of Flight (hours)": dur,
                "Days Until Flight": day,
            }

            # Save inputs to file
            save_inputs_to_file(user_inputs)

    if __name__ == '__main__':
        main()

