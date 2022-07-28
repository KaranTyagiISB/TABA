import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Uber Reveiw Analysis App!")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    Uber Inc in the US wants to know:

    1. The major complaints premium users have about their cab services,
    2. How these impact service ratings.
    3. The data are API collected from itunes for iOS users. The dataset uber_reviews_itune.csv is small, containing a mere 490 records.
    
    
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)

