import streamlit as st
from sklearn import linear_model
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from streamlit_option_menu import option_menu

import plotly.express as px

st.set_page_config(layout="wide")

select = option_menu(
    menu_title=None,
    options=["Home","Predict Price","About Tesla Y","About Dev"],
    orientation="horizontal"
)


df=pd.read_csv("teslay.csv")

if select == "Home":

    st.title("Tesla Y Model Price Analysis for 2024 and 2025")

    #col3, col4 = st.columns(2)
    #with col3:
    col5,col6,col7,col8 = st.columns(4)
    with col5:
        st.metric("Total colors of cars",df["color"].nunique())
    with col6:
        st.metric("Average Prices of Tesla Y Model",round(df["price"].mean(),2))
    with col7:
        st.metric("Maximum Price of Tesla Y Model",round(df["price"].max(),2))
    with col8:
        st.metric("Minimum Price of Tesla Y Model",round(df["price"].min(),2))


    col3,col4=st.columns(2)
    with col3:
        st.title("Total Value Count Color wise!")
        df2=df["color"].value_counts()
        st.dataframe(df2)

    with col4:

        st.title("Year wise (model) price trend!")

        date_year = df.groupby("year")["price"].mean().reset_index()

        fig=px.line(
            date_year,
            x="year",
            y="price",
            title="Price Trend"
        )

        st.plotly_chart(fig)

    col1,col2=st.columns(2)

    with col1:
        st.subheader("Raw Data set")
        st.dataframe(df)

    with col2:
        st.subheader("Color Wise Dataset")

        fig_price = px.bar(
            df,
            x="color",
            y="price",
            text="color",
            color="color"
        )

        st.plotly_chart(fig_price)

if select== "Predict Price":

    st.title("Predict the price of Tesla Y Model")
   

    le=LabelEncoder()

    df['color']=le.fit_transform(df['color'])

    x=df[["year","km","color"]]
    y=df.price

    # st.dataframe(x)
    # st.dataframe(y)

    model=linear_model.LinearRegression()
    model.fit(x,y)

    col9,col10=st.columns(2)
    with col9:
        st.info("by Selecting year, color, and km's")
        year=st.selectbox("Enter year:",[2020,2021,2022,2023,2024,2025,2026,2027,2028])
        km=int(st.number_input("Enter Kilometers:",min_value=6000))
        color=st.selectbox("select color",le.classes_)
        pred=st.button("Predict")


    with col10:
        col_ch=le.transform([color])[0]
        predicted_price=model.predict([[year,km,col_ch]])
        score=model.score(x,y)
        acuracy=int(score*100)
        st.metric("Accuracy of Current Model as per recent data!",acuracy,"%")
        if pred:
           
            if color:
                col_ch=le.transform([color])[0]
                predicted_price=model.predict([[year,km,col_ch]])
                st.balloons()
                score=model.score(x,y)
                acuracy=int(score*100)
                st.subheader("How accurate the predicted price is!")
                st.info(f"{acuracy}%")
                st.subheader("Here is your Predicted Price for Tesla Y")
                predic=int(predicted_price[0])
                st.info(predic)

            existing_data=pd.read_csv("teslay.csv")
            new_data=pd.DataFrame({"year":[year],"km":[km],"color":[color],"price":[predic]})
            updated_data=pd.concat([existing_data,new_data])
            updated_data.to_csv("teslay.csv",index=False)

if select == "About Tesla Y":
    st.title("Here is Some information About Tesla Y") 
    col11,col12=st.columns(2)
     
    with col11:
        st.image("Tesla model Y-16.jpg" ,width=600 )
    with col12:        
        st.write("Tesla, Inc. (TSLA) is an American electric vehicle and clean energy company founded in 2003. The company designs and manufactures electric cars, battery energy storage systems, and solar energy products.Tesla is known for popular vehicles such as the Model S, Model 3, Model X, and Model Y. The company aims to accelerate the world's transition to sustainable energy through innovation and advanced technology.CEO: Elon Musk Founded: 2003 Headquarters: Austin, Texas, USA Industry: Electric Vehicles and Clean Energy")
        st.subheader("Key Specifications & Features")
        st.markdown("""Range & Efficiency: Depending on the configuration, you can get anywhere from 294 to over 320 miles per charge on the standard/Long Range models.Performance: The AWD model goes from \(0\) to \(60\text{ mph}\) in just \(4.1\) seconds, while the Performance trim achieves the same in an blistering \(3.5\) seconds.Interior & Storage: Features a massive \(16\)-inch center touchscreen and a spacious cabin that offers an impressive \(74.8\) cubic feet of max cargo spac""")
        
    st.subheader("Standout Features") 
    st.markdown("""Charging Network: Access to Tesla's massive Supercharger network, offering fast charging speeds (adding up to 152 miles in just 15 minutes on base models).Comfort & Tech: Recent refreshes have significantly reduced cabin noise by over 20% using acoustic glass. The interior relies on a 16-inch center touchscreen to control nearly all vehicle functions, complemented by an 8-inch rear touchscreen for backseat passengers.Safety: Awarded top safety ratings and equipped with standard Full Self-Driving (Supervised) hardware.""")
   
    #col13,col14=st.columns(2)

    st.image("road_img.jpg" )
    st.subheader("All the Range You Need")
    st.markdown("Go anywhere with up to 321 miles of EPA estimated range on a single charge. Navigate on longer trips with Trip Planner and your Tesla will add Supercharging stops along the way.")
    
    col13,col14,col15=st.columns(3)
    with col13:
        st.metric(label="Recharge Time", value="15 min")
        st.write("Recharge up to 160 miles")

    with col14:
        st.metric(label="Global Superchargers", value="80,000+")

    with col15:
        st.metric(label="Supercharger Uptime", value="99%")

    st.image("Capture.PNG")

    col16, col17, col18 = st.columns(3)

    with col16:
        with st.container(border=True):
            st.markdown("⚡")
            st.subheader("Charging Costs Less")
            st.write("Electricity typically costs less than gas. When you plug in at home overnight, you can charge using lower-cost electricity and then start your day with plenty of range.")

    with col17:
        with st.container(border=True):
            st.markdown("🔧")
            st.subheader("Virtually No Maintenance")
            st.write("No oil changes, tune-ups or emissions inspections required. Simply keep your washer fluid topped up and rotate your tires to ensure smooth performance")

    with col18:
        with st.container(border=True):
            st.markdown("⚙️")
            st.subheader("Fewer Parts to Replace")
            st.write("A gas car engine has over 1,000 moving parts—Tesla drivetrains have around 20. Fewer moving parts means fewer repairs over time and greater reliability.")

        st.divider()

    st.subheader("🌟 Why Choose Tesla Model Y?")

    col19, col20, col21 = st.columns(3)

    with col19:
        with st.container(border=True):
            st.markdown("🔋")
            st.subheader("Long Electric Range")
            st.write(
                "Tesla Model Y offers impressive driving range, allowing you to travel long distances with fewer charging stops."
            )

    with col20:
        with st.container(border=True):
            st.markdown("🚀")
            st.subheader("High Performance")
            st.write(
                "Experience quick acceleration, smooth handling, and advanced technology for a modern driving experience."
            )

    with col21:
        with st.container(border=True):
            st.markdown("🛡️")
            st.subheader("Top Safety Features")
            st.write(
                "Tesla Model Y is designed with advanced safety systems and has received top safety ratings."
            )


    st.divider()

    st.info(
        "Tesla Model Y represents the future of sustainable transportation, "
        "combining innovation, safety, performance, and efficiency in one vehicle."
    )

    






if select == "About Dev":

    st.title("👨‍💻 About Developer")

    st.subheader("Shereen Aijaz")

    st.write("""
    Aspiring Machine Learning Engineer and Web Developer.

    Passionate about Artificial Intelligence, Machine Learning,
    Data Analysis, and Web Development.

    Currently building real-world projects using Python,
    Streamlit, and Machine Learning to gain practical experience.
    """)

    st.markdown("### 🔗 Connect With Me")

    st.markdown(
        "[LinkedIn](https://www.linkedin.com/in/shereen-aijaz-b2997430a)"
    )

    st.markdown(
        "[GitHub](https://github.com/Shireen123764)"
    )

    st.divider()

    st.subheader("🛠 Technologies Used In This Project")

    st.markdown("""
    - Python
    - Streamlit
    - Pandas
    - Scikit-Learn
    - Plotly Express
    - Linear Regression
    - Label Encoding
    - Data Visualization
    """)

    st.divider()

    st.subheader("📌 Project Purpose")

    st.write("""
    This project was developed to analyze Tesla Model Y pricing data
    and predict vehicle prices using Machine Learning.

    Users can enter:
    - Year
    - Kilometers Driven
    - Color

    and receive an estimated Tesla Model Y price instantly.
    """)

    st.subheader("✨ Project Features")

    st.markdown("""
    ✅ Interactive Dashboard

    ✅ Price Prediction

    ✅ Year-wise Price Analysis

    ✅ Color-wise Analysis

    ✅ Machine Learning Model

    ✅ Data Visualization

    ✅ User-Friendly Interface
    """)

    st.success("Thank you for visiting my project! 🚀")