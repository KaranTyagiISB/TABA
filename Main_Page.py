import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome !")
st.write("# Uber Voice of Customer Analysis (Reviews)!")
st.subheader("Namaste, salaam, satsriakal.")
         
st.sidebar.success("Select a demo above.")

st.markdown(
    """
    Uber Inc in the US wants to know:

    1. The major complaints premium users have about their cab services,
    2. How these impact service ratings.
     - **Data Source** - The data are API collected from itunes for iOS users. The dataset uber_reviews_itune.csv is small, containing a mere 490 records.
    
    
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

df = pd.read_csv('https://raw.githubusercontent.com/KaranTyagiISB/TABA/main/uber_reviews_itune.csv', encoding = 'latin1')
st.write(df.head(5))

st.text("")
st.text("")
st.text("")

uber_review_clean = []

for i in uber_reviews :
    clean = re.sub(".\w+[+]+\d+\w+\d+[<>]","", i)
    clean = re.sub("\n\n"," ",clean)
    clean = re.sub("\n"," ",clean)
    clean = clean.lower()
    
    
    uber_review_clean.append(clean)
    
    
rating = pd.DataFrame(uber_df["Rating"].value_counts().reset_index())
rating.columns = ["Rating","Count"]

plt.figure(figsize = (10,4))
 
ax = sns.barplot(x="Rating", y="Count", data=rating)    

for container in ax.containers:
    ax.bar_label(container)


plt.xlabel("Ratings", fontsize = 15)
plt.ylabel("Counts", fontsize = 15)

plt.yticks(fontsize = 12)
plt.title("Ratings", fontsize = 18, pad = 20)

plt.text(3,300,"Rating 1 - Extremely Low", fontsize = 12)
plt.text(3,270,"Rating 5 - Extremely High", fontsize = 12)

# Show Plot
plt.show()
