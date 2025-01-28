import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load(r"C:\Users\aadip\JupyterPython\cricket_win_predictor_model.pkl")

# Predefined team strengths (hidden from users)
team_strengths = {
    "India": 0.85, "Australia": 0.80, "England": 0.78, "New Zealand": 0.75,
    "Pakistan": 0.72, "South Africa": 0.70, "Sri Lanka": 0.68, "West Indies": 0.65,
    "Bangladesh": 0.60, "Afghanistan": 0.58, "Zimbabwe": 0.50, "Ireland": 0.48,
    "Netherlands": 0.45, "Scotland": 0.42, "Namibia": 0.40, "UAE": 0.38,
    "Nepal": 0.36, "Oman": 0.34, "USA": 0.30
}

# Predefined venue averages
venue_avg_1st_innings = {
    "Eden Gardens, Kolkata": 260,
    "Wankhede Stadium, Mumbai": 280,
    "M. Chinnaswamy Stadium, Bangalore": 270,
    "Narendra Modi Stadium, Ahmedabad": 250,
    "Sydney Cricket Ground, Sydney": 270,
    "Lord's, London": 240,
    "Melbourne Cricket Ground": 270,
    "Sharjah Cricket Stadium": 240,
    "Trent Bridge, Nottingham": 270,
    "The Oval, London": 260,
    "Old Trafford, Manchester": 250,
    "R Premadasa Stadium, Colombo": 260,
    "The Wanderers Stadium, Johannesburg": 290,
    "SuperSport Park, Centurion": 280,
    "MA Chidambaram Stadium, Chennai": 250,
    "Feroz Shah Kotla, Delhi": 240,
    "Greenfield International Stadium, Thiruvananthapuram": 260,
    "Sardar Patel Stadium, Ahmedabad": 270,
    "Arun Jaitley Stadium, Delhi": 250,
    "Holkar Cricket Stadium, Indore": 280,
    "Rajiv Gandhi International Stadium, Hyderabad": 260,
    "Punjab Cricket Association Stadium, Mohali": 270,
    "Barsapara Cricket Stadium, Guwahati": 250,
    "Hagley Oval, Christchurch": 250,
    "Bay Oval, Mount Maunganui": 260,
    "Kensington Oval, Bridgetown": 260,
    "Zahur Ahmed Chowdhury Stadium, Chattogram": 250,
    "Sher-e-Bangla National Cricket Stadium, Mirpur": 260,
    "Pallekele International Cricket Stadium": 250,
    "Queens Sports Club, Bulawayo": 230,
    "Harare Sports Club": 230,
    "Perth Stadium, Perth": 280,
    "Adelaide Oval, Adelaide": 270,
    "Brisbane Cricket Ground, Woolloongabba": 280,
    "Manuka Oval, Canberra": 250,
    "Bellerive Oval, Hobart": 260,
    "Sophia Gardens, Cardiff": 250,
    "Headingley, Leeds": 270,
    "Riverside Ground, Chester-le-Street": 260,
    "National Stadium, Karachi": 250,
    "Gaddafi Stadium, Lahore": 250,
    "Multan Cricket Stadium": 240,
    "ICC Academy, Dubai": 240,
    "Dubai International Cricket Stadium": 240,
    "Darren Sammy National Cricket Stadium, Gros Islet": 240,
}

venue_avg_2nd_innings = {
    "Eden Gardens, Kolkata": 240, "Wankhede Stadium, Mumbai": 260, "M. Chinnaswamy Stadium, Bangalore": 250,
    "Narendra Modi Stadium, Ahmedabad": 230, "Sydney Cricket Ground, Sydney": 250,
    "Lord's, London": 220, "Melbourne Cricket Ground, Melbourne": 240
}

# Streamlit app
st.title("Cricket Match Win Predictor")

# Team selection
team1 = st.selectbox("Select Team 1", options=list(team_strengths.keys()))
team2 = st.selectbox("Select Team 2", options=list(team_strengths.keys()))

# Venue selection
venue = st.selectbox("Select the Venue", options=list(venue_avg_1st_innings.keys()))

# Toss selection
toss_winner = st.selectbox("Which team won the toss?", [team1, team2])
toss_decision = st.radio(f"What did {toss_winner} decide to do?", ["Bat", "Bowl"])

# Determine batting and bowling teams
if toss_winner == team1:
    if toss_decision == "Bat":
        batting_team1, bowling_team2 = team1, team2
    else:
        batting_team1, bowling_team2 = team2, team1
else:
    if toss_decision == "Bat":
        batting_team1, bowling_team2 = team2, team1
    else:
        batting_team1, bowling_team2 = team1, team2

st.write(f"**Batting Team (1st Innings):** {batting_team1}")
st.write(f"**Bowling Team (1st Innings):** {bowling_team2}")

# Inputs for the first innings
team1_score = st.number_input(f"Enter the total score for {batting_team1} (Team 1)", min_value=0)
team1_wickets = st.number_input(f"How many wickets did {batting_team1} lose?", min_value=0, max_value=10)
team1_overs = st.number_input(f"How many overs did {batting_team1} play?", min_value=0.0, max_value=50.0)

# Inputs for the second innings
team2_score = st.number_input(f"Enter the current score for {bowling_team2} (Team 2)", min_value=0)
team2_wickets = st.number_input(f"How many wickets has {bowling_team2} lost?", min_value=0, max_value=10)
team2_overs = st.number_input(f"How many overs has {bowling_team2} played?", min_value=0.0, max_value=50.0)

# Derived features
current_run_rate = team2_score / team2_overs if team2_overs > 0 else 0
required_run_rate = (team1_score - team2_score) / (50 - team2_overs) if (50 - team2_overs) > 0 else 0

# Assign team strengths
team1_strength = team_strengths[batting_team1]
team2_strength = team_strengths[bowling_team2]

# Venue averages
venue_avg_1st = venue_avg_1st_innings.get(venue, 250)
venue_avg_2nd = venue_avg_2nd_innings.get(venue, 230)

# Prepare input data
input_data = pd.DataFrame([{
    'team1_strength': team1_strength,
    'team2_strength': team2_strength,
    'venue_avg_1st': venue_avg_1st,
    'venue_avg_2nd': venue_avg_2nd,
    'team1_score': team1_score,
    'team1_wickets': team1_wickets,
    'team2_score': team2_score,
    'team2_overs': team2_overs,
    'team2_wickets': team2_wickets,
    'current_run_rate': current_run_rate,
    'required_run_rate': required_run_rate
}])

# Predict probabilities
if st.button("Predict Match Outcome"):
    win_probabilities = model.predict_proba(input_data)
    team1_win_prob = win_probabilities[0][1] * 100
    team2_win_prob = win_probabilities[0][0] * 100

    st.write(f"**{batting_team1} Win Probability:** {team1_win_prob:.2f}%")
    st.write(f"**{bowling_team2} Win Probability:** {team2_win_prob:.2f}%")
