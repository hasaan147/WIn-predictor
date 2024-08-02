import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import streamlit as st

# Load and prepare the dataset
def prepare_data():
    df = pd.read_csv("task.csv")
    df = df.drop("Date", axis=1)

    # Encode categorical columns
    label_encoder_venue = LabelEncoder()
    label_encoder_bat_first = LabelEncoder()
    label_encoder_bat_second = LabelEncoder()
    label_encoder_winner = LabelEncoder()

    df['Venue'] = label_encoder_venue.fit_transform(df['Venue'])
    df['Bat First'] = label_encoder_bat_first.fit_transform(df['Bat First'])
    df['Bat Second'] = label_encoder_bat_second.fit_transform(df['Bat Second'])
    df['Winner'] = label_encoder_winner.fit_transform(df['Winner'])

    # Save label encoders
    with open('label_encoder_venue.pkl', 'wb') as f:
        pickle.dump(label_encoder_venue, f)
    with open('label_encoder_bat_first.pkl', 'wb') as f:
        pickle.dump(label_encoder_bat_first, f)
    with open('label_encoder_bat_second.pkl', 'wb') as f:
        pickle.dump(label_encoder_bat_second, f)
    with open('label_encoder_winner.pkl', 'wb') as f:
        pickle.dump(label_encoder_winner, f)

    # Prepare data for training
    X = df[['Venue', 'Bat First', 'Bat Second']]
    y = df['Winner']
    return X, y, label_encoder_venue, label_encoder_bat_first, label_encoder_bat_second, label_encoder_winner

# Train and save the model
def train_and_save_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

# Load model and label encoders
def load_resources():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('label_encoder_venue.pkl', 'rb') as f:
            label_encoder_venue = pickle.load(f)
        with open('label_encoder_bat_first.pkl', 'rb') as f:
            label_encoder_bat_first = pickle.load(f)
        with open('label_encoder_bat_second.pkl', 'rb') as f:
            label_encoder_bat_second = pickle.load(f)
        with open('label_encoder_winner.pkl', 'rb') as f:
            label_encoder_winner = pickle.load(f)
        return model, label_encoder_venue, label_encoder_bat_first, label_encoder_bat_second, label_encoder_winner
    except FileNotFoundError as e:
        st.error(f"Error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

# Predict the winner
def predict_winner(model, label_encoder_venue, label_encoder_bat_first, label_encoder_bat_second, label_encoder_winner, venue, bat_first, bat_second):
    try:
        venue_encoded = label_encoder_venue.transform([venue])
        bat_first_encoded = label_encoder_bat_first.transform([bat_first])
        bat_second_encoded = label_encoder_bat_second.transform([bat_second])
    except ValueError as e:
        return f"Error: One or more input values are not recognized. {e}"

    input_data = {'Venue': [venue_encoded[0]], 'Bat First': [bat_first_encoded[0]], 'Bat Second': [bat_second_encoded[0]]}
    input_df = pd.DataFrame(input_data)

    try:
        prediction = model.predict(input_df)
        winner = label_encoder_winner.inverse_transform(prediction)
        return winner[0]
    except Exception as e:
        return f"Error: Prediction failed. {e}"

# Streamlit application
def main():
    # Define the teams and venues
    teams = ["India", "Pakistan", "Afghanistan", "Bangladesh", "Australia", "Ireland",
             "New Zealand", "South Africa", "Sri Lanka", "West Indies", "Zimbabwe",
             "Uganda", "Iceland"]

    venues = ['The Rose Bowl', 'Eden Park', 'New Wanderers Stadium', 'County Ground',
              'Gahanga International Cricket Stadium', 'GB Oval', 'Sportpark Het Schootsveld',
              'Malahide', 'Amini Park', 'Gymkhana Club Ground', 'Sylhet International Cricket Stadium',
              'Providence Stadium', 'Scott Page Field', 'JSCA International Stadium Complex',
              'Queens Sports Club']

    # Load or train the model
    try:
        X, y, label_encoder_venue, label_encoder_bat_first, label_encoder_bat_second, label_encoder_winner = prepare_data()
        train_and_save_model(X, y)
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    model, label_encoder_venue, label_encoder_bat_first, label_encoder_bat_second, label_encoder_winner = load_resources()

    # Streamlit UI
    st.title("Cricket Match Prediction")

    venue = st.selectbox("Select Venue:", venues)
    bat_first = st.selectbox("Select Team Batting First:", teams)
    bat_second = st.selectbox("Select Team Batting Second:", teams)

    if st.button("Predict Winner"):
        winner = predict_winner(model, label_encoder_venue, label_encoder_bat_first, label_encoder_bat_second, label_encoder_winner, venue, bat_first, bat_second)
        st.write(f"**Predicted Winner:** {winner}")

if __name__ == "__main__":
    main()
