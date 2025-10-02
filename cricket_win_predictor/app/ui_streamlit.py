"""
Streamlit UI for Cricket Win Predictor.
Refactored from Basic_cricket_win_predictor.py with improved structure.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cwp.models import CricketWinPredictor
from cwp.data_loader import CricketDataLoader


class CricketWinPredictorUI:
    """Streamlit UI for Cricket Win Predictor."""
    
    def __init__(self):
        """Initialize the UI."""
        self.model = None
        self.data_loader = CricketDataLoader()
        self.team_strengths = {}
        self.venue_averages = {}
        
    def load_model(self, model_path: str = "models/cricket_win_predictor_model.pkl"):
        """Load the trained model."""
        try:
            self.model = CricketWinPredictor()
            self.model.load_model(model_path)
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def get_team_strengths(self):
        """Get predefined team strengths."""
        return {
            "India": 0.85, "Australia": 0.80, "England": 0.78, "New Zealand": 0.75,
            "Pakistan": 0.72, "South Africa": 0.70, "Sri Lanka": 0.68, "West Indies": 0.65,
            "Bangladesh": 0.60, "Afghanistan": 0.58, "Zimbabwe": 0.50, "Ireland": 0.48,
            "Netherlands": 0.45, "Scotland": 0.42, "Namibia": 0.40, "UAE": 0.38,
            "Nepal": 0.36, "Oman": 0.34, "USA": 0.30
        }
    
    def get_venue_averages(self):
        """Get predefined venue averages."""
        return {
            "Eden Gardens, Kolkata": {"1st": 260, "2nd": 240},
            "Wankhede Stadium, Mumbai": {"1st": 280, "2nd": 260},
            "M. Chinnaswamy Stadium, Bangalore": {"1st": 270, "2nd": 250},
            "Narendra Modi Stadium, Ahmedabad": {"1st": 250, "2nd": 230},
            "Sydney Cricket Ground, Sydney": {"1st": 270, "2nd": 250},
            "Lord's, London": {"1st": 240, "2nd": 220},
            "Melbourne Cricket Ground": {"1st": 270, "2nd": 240},
            "Sharjah Cricket Stadium": {"1st": 240, "2nd": 220},
            "Trent Bridge, Nottingham": {"1st": 270, "2nd": 250},
            "The Oval, London": {"1st": 260, "2nd": 240},
            "Old Trafford, Manchester": {"1st": 250, "2nd": 230},
            "R Premadasa Stadium, Colombo": {"1st": 260, "2nd": 240},
            "The Wanderers Stadium, Johannesburg": {"1st": 290, "2nd": 270},
            "SuperSport Park, Centurion": {"1st": 280, "2nd": 260},
            "MA Chidambaram Stadium, Chennai": {"1st": 250, "2nd": 230},
            "Feroz Shah Kotla, Delhi": {"1st": 240, "2nd": 220},
            "Greenfield International Stadium, Thiruvananthapuram": {"1st": 260, "2nd": 240},
            "Sardar Patel Stadium, Ahmedabad": {"1st": 270, "2nd": 250},
            "Arun Jaitley Stadium, Delhi": {"1st": 250, "2nd": 230},
            "Holkar Cricket Stadium, Indore": {"1st": 280, "2nd": 260},
            "Rajiv Gandhi International Stadium, Hyderabad": {"1st": 260, "2nd": 240},
            "Punjab Cricket Association Stadium, Mohali": {"1st": 270, "2nd": 250},
            "Barsapara Cricket Stadium, Guwahati": {"1st": 250, "2nd": 230},
            "Hagley Oval, Christchurch": {"1st": 250, "2nd": 230},
            "Bay Oval, Mount Maunganui": {"1st": 260, "2nd": 240},
            "Kensington Oval, Bridgetown": {"1st": 260, "2nd": 240},
            "Zahur Ahmed Chowdhury Stadium, Chattogram": {"1st": 250, "2nd": 230},
            "Sher-e-Bangla National Cricket Stadium, Mirpur": {"1st": 260, "2nd": 240},
            "Pallekele International Cricket Stadium": {"1st": 250, "2nd": 230},
            "Queens Sports Club, Bulawayo": {"1st": 230, "2nd": 210},
            "Harare Sports Club": {"1st": 230, "2nd": 210},
            "Perth Stadium, Perth": {"1st": 280, "2nd": 260},
            "Adelaide Oval, Adelaide": {"1st": 270, "2nd": 250},
            "Brisbane Cricket Ground, Woolloongabba": {"1st": 280, "2nd": 260},
            "Manuka Oval, Canberra": {"1st": 250, "2nd": 230},
            "Bellerive Oval, Hobart": {"1st": 260, "2nd": 240},
            "Sophia Gardens, Cardiff": {"1st": 250, "2nd": 230},
            "Headingley, Leeds": {"1st": 270, "2nd": 250},
            "Riverside Ground, Chester-le-Street": {"1st": 260, "2nd": 240},
            "National Stadium, Karachi": {"1st": 250, "2nd": 230},
            "Gaddafi Stadium, Lahore": {"1st": 250, "2nd": 230},
            "Multan Cricket Stadium": {"1st": 240, "2nd": 220},
            "ICC Academy, Dubai": {"1st": 240, "2nd": 220},
            "Dubai International Cricket Stadium": {"1st": 240, "2nd": 220},
            "Darren Sammy National Cricket Stadium, Gros Islet": {"1st": 240, "2nd": 220}
        }
    
    def render_sidebar(self):
        """Render the sidebar with match inputs."""
        st.sidebar.header("🏏 Match Configuration")
        
        # Team selection
        team_strengths = self.get_team_strengths()
        team1 = st.sidebar.selectbox("Select Team 1", options=list(team_strengths.keys()))
        team2 = st.sidebar.selectbox("Select Team 2", options=list(team_strengths.keys()))
        
        # Venue selection
        venue_averages = self.get_venue_averages()
        venue = st.sidebar.selectbox("Select the Venue", options=list(venue_averages.keys()))
        
        # Toss selection - using session state to ensure refresh when teams change
        # Initialize session state for toss if teams change
        if 'last_teams' not in st.session_state or st.session_state.last_teams != f"{team1}_{team2}":
            st.session_state.last_teams = f"{team1}_{team2}"
            st.session_state.toss_winner = team1  # Reset to first team
            st.session_state.toss_decision = "Bat"  # Reset decision
        
        toss_winner = st.sidebar.selectbox("Which team won the toss?", [team1, team2], key="toss_winner_select")
        toss_decision = st.sidebar.radio(f"What did {toss_winner} decide to do?", ["Bat", "Bowl"], key="toss_decision_select")
        
        return {
            'team1': team1,
            'team2': team2,
            'venue': venue,
            'toss_winner': toss_winner,
            'toss_decision': toss_decision
        }
    
    def render_match_inputs(self, match_config):
        """Render the main match input section."""
        st.header("📊 Match Progress Input")
        
        # Determine batting and bowling teams
        team1, team2 = match_config['team1'], match_config['team2']
        toss_winner = match_config['toss_winner']
        toss_decision = match_config['toss_decision']
        
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
        
        # Create two columns for inputs
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏏 First Innings")
            team1_score = st.number_input(f"Enter the total score for {batting_team1}", min_value=0, value=250)
            team1_wickets = st.number_input(f"How many wickets did {batting_team1} lose?", min_value=0, max_value=10, value=7)
            team1_overs = st.number_input(f"How many overs did {batting_team1} play?", min_value=0.0, max_value=50.0, value=50.0)
        
        with col2:
            st.subheader("🎯 Second Innings")
            team2_score = st.number_input(f"Enter the current score for {bowling_team2}", min_value=0, value=150)
            team2_wickets = st.number_input(f"How many wickets has {bowling_team2} lost?", min_value=0, max_value=10, value=4)
            team2_overs = st.number_input(f"How many overs has {bowling_team2} played?", min_value=0.0, max_value=50.0, value=30.0)
        
        return {
            'batting_team1': batting_team1,
            'bowling_team2': bowling_team2,
            'team1_score': team1_score,
            'team1_wickets': team1_wickets,
            'team1_overs': team1_overs,
            'team2_score': team2_score,
            'team2_wickets': team2_wickets,
            'team2_overs': team2_overs
        }
    
    def calculate_derived_features(self, match_inputs):
        """Calculate derived features from match inputs."""
        team2_score = match_inputs['team2_score']
        team2_overs = match_inputs['team2_overs']
        team1_score = match_inputs['team1_score']
        
        # Calculate run rates
        current_run_rate = team2_score / team2_overs if team2_overs > 0 else 0
        required_run_rate = (team1_score - team2_score) / (50 - team2_overs) if (50 - team2_overs) > 0 else 0
        
        return {
            'current_run_rate': current_run_rate,
            'required_run_rate': required_run_rate
        }
    
    def make_prediction(self, match_config, match_inputs, derived_features):
        """Make prediction using the loaded model."""
        if self.model is None:
            st.error("Model not loaded. Please ensure the model file exists.")
            return None
        
        try:
            # Get team strengths
            team_strengths = self.get_team_strengths()
            team1_strength = team_strengths[match_config['team1']]
            team2_strength = team_strengths[match_config['team2']]
            
            # Get venue averages
            venue_averages = self.get_venue_averages()
            venue = match_config['venue']
            venue_avg_1st = venue_averages[venue]['1st']
            venue_avg_2nd = venue_averages[venue]['2nd']
            
            # Make prediction
            prediction = self.model.predict_match(
                team1_strength=team1_strength,
                team2_strength=team2_strength,
                venue_avg_1st=venue_avg_1st,
                venue_avg_2nd=venue_avg_2nd,
                team1_wickets=match_inputs['team1_wickets'],
                team2_wickets=match_inputs['team2_wickets'],
                current_run_rate=derived_features['current_run_rate'],
                required_run_rate=derived_features['required_run_rate']
            )
            
            return prediction
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            return None
    
    def display_prediction_results(self, prediction, match_config, match_inputs):
        """Display prediction results."""
        if prediction is None:
            return
        
        st.header("🎯 Prediction Results")
        
        # Create columns for results
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label=f"{match_config['team1']} Win Probability",
                value=f"{prediction['team1_win_probability']*100:.1f}%"
            )
        
        with col2:
            st.metric(
                label=f"{match_config['team2']} Win Probability", 
                value=f"{prediction['team2_win_probability']*100:.1f}%"
            )
        
        # Display prediction bar chart
        st.subheader("📊 Win Probability Visualization")
        
        import plotly.express as px
        
        prob_data = pd.DataFrame({
            'Team': [match_config['team1'], match_config['team2']],
            'Win Probability': [
                prediction['team1_win_probability']*100,
                prediction['team2_win_probability']*100
            ]
        })
        
        fig = px.bar(prob_data, x='Team', y='Win Probability', 
                    title='Team Win Probabilities',
                    color='Win Probability',
                    color_continuous_scale='RdYlGn')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display match summary
        st.subheader("📋 Match Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Run Rate", f"{match_inputs['team2_score']/match_inputs['team2_overs']:.2f}")
        
        with col2:
            st.metric("Required Run Rate", f"{(match_inputs['team1_score']-match_inputs['team2_score'])/(50-match_inputs['team2_overs']):.2f}")
        
        with col3:
            st.metric("Wickets in Hand", f"{10-match_inputs['team2_wickets']}")
    
    def run(self):
        """Run the Streamlit application."""
        # Page configuration
        st.set_page_config(
            page_title="Cricket Win Predictor",
            page_icon="🏏",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Title and description
        st.title("🏏 Cricket Match Win Predictor")
        st.markdown("Predict the outcome of cricket matches using machine learning!")
        
        # Load model
        if not self.load_model():
            st.error("Failed to load model. Please check if the model file exists.")
            st.stop()
        
        # Render sidebar
        match_config = self.render_sidebar()
        
        # Render match inputs
        match_inputs = self.render_match_inputs(match_config)
        
        # Calculate derived features
        derived_features = self.calculate_derived_features(match_inputs)
        
        # Prediction button
        if st.button("🔮 Predict Match Outcome", type="primary"):
            with st.spinner("Making prediction..."):
                prediction = self.make_prediction(match_config, match_inputs, derived_features)
                
                if prediction:
                    self.display_prediction_results(prediction, match_config, match_inputs)
        
        # Display additional information
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ℹ️ About")
        st.sidebar.markdown("This predictor uses machine learning to analyze match conditions and predict outcomes.")
        
        # Footer
        st.markdown("---")
        st.markdown("Built with ❤️ using Streamlit and scikit-learn")


def main():
    """Main function to run the application."""
    app = CricketWinPredictorUI()
    app.run()


if __name__ == "__main__":
    main()

