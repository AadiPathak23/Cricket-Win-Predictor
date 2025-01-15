# Cricket-Win-Predictor
This project is a Cricket Match Win Predictor application built using Python and Streamlit, leveraging historical One-Day International (ODI) cricket data to predict match outcomes. The model incorporates machine learning algorithms to provide real-time win probability for ongoing matches based on various inputs.
#Key Features
1) Dynamic Predictions:
Provides real-time win probabilities for two teams based on live match inputs.

User-Friendly Interface:
Built with Streamlit for an intuitive, interactive web interface.

Data-Driven Insights:
Analyzes historical data of teams and venues to calculate strengths and averages.

Comprehensive Inputs:
Factors in toss decisions, venue, team strengths, innings scores, overs, and wickets to predict outcomes.

Customizable:
Supports adding new teams, venues, and data to improve predictions over time.

2) Inputs Required

->Team 1 and Team 2 (dropdown with all international teams).
->Venue (dropdown with all stadiums used for international cricket).
->Toss winner and their decision (bat or bowl).
->First innings details:
  Score, overs played, and wickets lost.
->Second innings details:
  Current score, overs played, and wickets lost.

3) Output

Predicts the probability of each team winning, displayed as a percentage.

4) How It Works

Historical match data is preprocessed to calculate team and venue strengths.
A Random Forest Classifier is trained on features such as team strengths, venue averages, and match progress.
User inputs are fed into the model to generate real-time predictions.

5) Tech Stack

-> Programming Language: Python

-> Framework: Streamlit

-> Libraries: pandas, scikit-learn, matplotlib, seaborn, joblib

-> Model: Random Forest Classifier 
