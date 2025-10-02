"""
Enhanced Cricket Win Predictor
With progressive field enabling and full team/venue lists
"""

import streamlit as st
import pandas as pd

# Page config - Restore wide layout
st.set_page_config(
    page_title="Cricket Predictor", 
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🏏 Cricket Match Win Predictor")
st.markdown("Predict the outcome of cricket matches using advanced analytics!")

# Full team list
ALL_TEAMS = [
    "India", "Australia", "England", "New Zealand", "Pakistan", "South Africa", 
    "Sri Lanka", "West Indies", "Bangladesh", "Afghanistan", "Zimbabwe", 
    "Ireland", "Netherlands", "Scotland", "Namibia", "UAE", "Nepal", 
    "Oman", "USA", "Canada", "PNG", "Hong Kong", "Singapore"
]

# Full venue list
ALL_VENUES = [
    "Eden Gardens, Kolkata", "Wankhede Stadium, Mumbai", "M. Chinnaswamy Stadium, Bangalore",
    "Narendra Modi Stadium, Ahmedabad", "Sydney Cricket Ground, Sydney", "Lord's, London",
    "Melbourne Cricket Ground", "Sharjah Cricket Stadium", "Trent Bridge, Nottingham",
    "The Oval, London", "Old Trafford, Manchester", "R Premadasa Stadium, Colombo",
    "The Wanderers Stadium, Johannesburg", "SuperSport Park, Centurion",
    "MA Chidambaram Stadium, Chennai", "Feroz Shah Kotla, Delhi",
    "Greenfield International Stadium, Thiruvananthapuram", "Sardar Patel Stadium, Ahmedabad",
    "Arun Jaitley Stadium, Delhi", "Holkar Cricket Stadium, Indore",
    "Rajiv Gandhi International Stadium, Hyderabad", "Punjab Cricket Association Stadium, Mohali",
    "Barsapara Cricket Stadium, Guwahati", "Hagley Oval, Christchurch",
    "Bay Oval, Mount Maunganui", "Kensington Oval, Bridgetown",
    "Zahur Ahmed Chowdhury Stadium, Chattogram", "Sher-e-Bangla National Cricket Stadium, Mirpur",
    "Pallekele International Cricket Stadium", "Queens Sports Club, Bulawayo",
    "Harare Sports Club", "Perth Stadium, Perth", "Adelaide Oval, Adelaide",
    "Brisbane Cricket Ground, Woolloongabba", "Manuka Oval, Canberra",
    "Bellerive Oval, Hobart", "Sophia Gardens, Cardiff", "Headingley, Leeds",
    "Riverside Ground, Chester-le-Street", "National Stadium, Karachi",
    "Gaddafi Stadium, Lahore", "Multan Cricket Stadium", "ICC Academy, Dubai",
    "Dubai International Cricket Stadium", "Darren Sammy National Cricket Stadium, Gros Islet"
]

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1

# Create main layout with sidebar
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Match Setup")
    
    # Step 1: Team Selection
    st.subheader("Step 1: Select Teams")
    
    team_col1, team_col2 = st.columns(2)
    with team_col1:
        team1 = st.selectbox("Team 1", ["No selection"] + ALL_TEAMS, key="team1_select")
    with team_col2:
        team2 = st.selectbox("Team 2", ["No selection"] + ALL_TEAMS, key="team2_select")
    
    # Validate team selection
    teams_selected = (team1 != "No selection" and team2 != "No selection" and team1 != team2)
    
    if not teams_selected:
        if team1 == team2 and team1 != "No selection":
            st.error("⚠️ Please select different teams!")
        else:
            st.info("👆 Please select both teams to continue")
    else:
        st.success(f"✅ Teams selected: {team1} vs {team2}")
    
    # Step 2: Venue Selection (only if teams are selected)
    if teams_selected:
        st.subheader("Step 2: Select Venue")
        venue = st.selectbox("Venue", ["No selection"] + ALL_VENUES, key="venue_select")
        
        venue_selected = (venue != "No selection")
        
        if not venue_selected:
            st.info("👆 Please select a venue to continue")
        else:
            st.success(f"✅ Venue selected: {venue}")
    
    # Step 3: Toss Information (only if teams and venue are selected)
    if teams_selected and venue_selected:
        st.subheader("Step 3: Toss Information")
        
        # Initialize session state for toss if teams change
        if 'last_teams' not in st.session_state or st.session_state.last_teams != f"{team1}_{team2}":
            st.session_state.last_teams = f"{team1}_{team2}"
            st.session_state.toss_winner = "No selection"
            st.session_state.toss_decision = "No selection"
        
        toss_winner = st.selectbox("Who won the toss?", ["No selection", team1, team2], key="toss_winner_select")
        toss_decision = st.radio("What did they decide?", ["No selection", "Bat", "Bowl"], key="toss_decision_select")
        
        toss_complete = (toss_winner != "No selection" and toss_decision != "No selection")
        
        if not toss_complete:
            st.info("👆 Please complete toss information to continue")
        else:
            st.success(f"✅ Toss: {toss_winner} won and chose to {toss_decision.lower()}")
    
    # Step 4: Match Progress (only if all previous steps are complete)
    if teams_selected and venue_selected and toss_complete:
        st.subheader("Step 4: Match Progress")
        
        # Determine batting and bowling teams
        if toss_winner == team1:
            if toss_decision == "Bat":
                batting_team, bowling_team = team1, team2
            else:
                batting_team, bowling_team = team2, team1
        else:
            if toss_decision == "Bat":
                batting_team, bowling_team = team2, team1
            else:
                batting_team, bowling_team = team1, team2
        
        st.write(f"**🏏 Batting First:** {batting_team}")
        st.write(f"**🎯 Bowling First:** {bowling_team}")
        
        # Match progress inputs
        progress_col1, progress_col2 = st.columns(2)
        
        with progress_col1:
            st.subheader(f"First Innings - {batting_team}")
            first_runs = st.number_input("Total Runs", min_value=0, value=250, key="first_runs")
            first_wickets = st.number_input("Wickets Lost", min_value=0, max_value=10, value=7, key="first_wickets")
            first_overs = st.number_input("Overs Played", min_value=0.0, max_value=50.0, value=50.0, key="first_overs")
        
        with progress_col2:
            st.subheader(f"Second Innings - {bowling_team}")
            second_runs = st.number_input("Current Runs", min_value=0, value=150, key="second_runs")
            second_wickets = st.number_input("Wickets Lost", min_value=0, max_value=10, value=4, key="second_wickets")
            second_overs = st.number_input("Overs Played", min_value=0.0, max_value=50.0, value=30.0, key="second_overs")
        
        # Prediction button
        if st.button("🔮 Predict Match Outcome", type="primary", key="predict_btn"):
            # Calculate metrics with correct target (1st innings total + 1)
            target = first_runs + 1  # Target is always 1 more than 1st innings total
            current_rr = second_runs / second_overs if second_overs > 0 else 0
            required_rr = (target - second_runs) / (50 - second_overs) if (50 - second_overs) > 0 else 0
            runs_needed = target - second_runs
            
            # Display metrics
            st.header("🎯 Prediction Results")
            
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            with metrics_col1:
                st.metric("Current RR", f"{current_rr:.2f}")
            with metrics_col2:
                st.metric("Required RR", f"{required_rr:.2f}")
            with metrics_col3:
                st.metric("Target", f"{target}")
            with metrics_col4:
                st.metric("Runs Needed", runs_needed)
            
            # Enhanced prediction logic with multiple factors
            
            # 1. Base team strengths (overall ODI rankings)
            team_strengths = {
                "India": 0.85, "Australia": 0.80, "England": 0.78, "New Zealand": 0.75,
                "Pakistan": 0.72, "South Africa": 0.70, "Sri Lanka": 0.68, "West Indies": 0.65,
                "Bangladesh": 0.60, "Afghanistan": 0.58, "Zimbabwe": 0.50, "Ireland": 0.48,
                "Netherlands": 0.45, "Scotland": 0.42, "Namibia": 0.40, "UAE": 0.38,
                "Nepal": 0.36, "Oman": 0.34, "USA": 0.30, "Canada": 0.28,
                "PNG": 0.25, "Hong Kong": 0.23, "Singapore": 0.20
            }
            
            # 2. Head-to-head analysis (historical win percentages)
            head_to_head = {
                # India vs others
                ("India", "Australia"): 0.52, ("India", "England"): 0.55, ("India", "New Zealand"): 0.58,
                ("India", "Pakistan"): 0.60, ("India", "South Africa"): 0.48, ("India", "Sri Lanka"): 0.65,
                ("India", "West Indies"): 0.62, ("India", "Bangladesh"): 0.78, ("India", "Afghanistan"): 0.85,
                # Australia vs others
                ("Australia", "England"): 0.58, ("Australia", "New Zealand"): 0.55, ("Australia", "Pakistan"): 0.52,
                ("Australia", "South Africa"): 0.48, ("Australia", "Sri Lanka"): 0.60, ("Australia", "West Indies"): 0.65,
                ("Australia", "Bangladesh"): 0.75, ("Australia", "Afghanistan"): 0.82,
                # England vs others
                ("England", "New Zealand"): 0.52, ("England", "Pakistan"): 0.55, ("England", "South Africa"): 0.50,
                ("England", "Sri Lanka"): 0.58, ("England", "West Indies"): 0.62, ("England", "Bangladesh"): 0.72,
                ("England", "Afghanistan"): 0.78,
                # Add more combinations as needed...
            }
            
            # 3. Ground-specific win statistics
            ground_advantages = {
                "Eden Gardens, Kolkata": {"India": 0.75, "Australia": 0.45, "England": 0.40, "Pakistan": 0.35},
                "Wankhede Stadium, Mumbai": {"India": 0.72, "Australia": 0.48, "England": 0.42, "Pakistan": 0.38},
                "Lord's, London": {"England": 0.70, "Australia": 0.55, "India": 0.45, "Pakistan": 0.40},
                "Melbourne Cricket Ground": {"Australia": 0.75, "England": 0.48, "India": 0.42, "New Zealand": 0.38},
                "Sydney Cricket Ground": {"Australia": 0.72, "England": 0.45, "India": 0.40, "New Zealand": 0.35},
                # Default neutral ground advantage
                "default": 0.50
            }
            
            # Get base strengths
            team1_strength = team_strengths.get(team1, 0.50)
            team2_strength = team_strengths.get(team2, 0.50)
            
            # Calculate head-to-head factor
            h2h_key = (team1, team2) if (team1, team2) in head_to_head else (team2, team1)
            if h2h_key in head_to_head:
                h2h_team1_advantage = head_to_head[h2h_key] if h2h_key == (team1, team2) else 1 - head_to_head[h2h_key]
            else:
                h2h_team1_advantage = 0.50  # Neutral if no historical data
            
            # Calculate ground advantage
            if venue in ground_advantages:
                team1_ground_adv = ground_advantages[venue].get(team1, 0.50)
                team2_ground_adv = ground_advantages[venue].get(team2, 0.50)
            else:
                team1_ground_adv = 0.50
                team2_ground_adv = 0.50
            
            # Calculate match situation factors
            overs_left = 50 - second_overs
            wickets_left = 10 - second_wickets
            
            # Run rate analysis (most important factor for chasing)
            if required_rr > 0:
                rr_factor = current_rr / required_rr  # >1 is good for chasing team
                if rr_factor > 1.3:
                    chase_advantage = 0.8  # Excellent chasing position
                elif rr_factor > 1.1:
                    chase_advantage = 0.65  # Strong chasing position
                elif rr_factor > 1.0:
                    chase_advantage = 0.55  # Good chasing position
                elif rr_factor > 0.9:
                    chase_advantage = 0.4  # Moderate chasing position
                elif rr_factor > 0.8:
                    chase_advantage = 0.25  # Poor chasing position
                else:
                    chase_advantage = 0.1  # Very poor chasing position
            else:
                chase_advantage = 0.5  # Neutral
            
            # Enhanced wickets factor with realistic penalty/bonus system
            if wickets_left <= 5:
                # Heavy penalty for losing wickets after 5th wicket
                # Each wicket lost after 5th has exponential impact
                wickets_lost_after_5 = 5 - wickets_left
                if wickets_lost_after_5 > 0:
                    # Exponential penalty: 0.7^wickets_lost_after_5
                    wicket_factor = 0.7 ** wickets_lost_after_5
                else:
                    wicket_factor = 1.0  # 5 wickets in hand = neutral
            else:
                # Bonus for having more than 5 wickets in hand
                extra_wickets = wickets_left - 5
                # Bonus factor: 1.2^extra_wickets (up to 3 extra wickets)
                bonus_factor = min(1.2 ** extra_wickets, 1.5)  # Cap at 50% bonus
                wicket_factor = bonus_factor
            
            # Overs factor
            overs_factor = overs_left / 50
            
            # Calculate final probabilities with realistic logic
            if runs_needed <= 0:
                # Match already won by chasing team
                if batting_team == team1:
                    team1_prob, team2_prob = 100, 0
                else:
                    team1_prob, team2_prob = 0, 100
            else:
                # Determine which team is chasing
                chasing_team = bowling_team  # The team batting second is chasing
                
                # Base probabilities from team strengths and historical factors
                base_team1_prob = team1_strength / (team1_strength + team2_strength)
                
                # Apply head-to-head advantage (30% weight)
                h2h_adjusted_prob = base_team1_prob * 0.7 + h2h_team1_advantage * 0.3
                
                # Apply ground advantage (25% weight)
                ground_adjusted_prob = h2h_adjusted_prob * 0.75 + (team1_ground_adv / (team1_ground_adv + team2_ground_adv)) * 0.25
                
                # Now apply match situation factors (45% weight) - this is the most important
                if chasing_team == team1:
                    # Team1 is chasing, Team2 has set the target
                    # Start with Team2 having advantage (they set the target)
                    base_chase_prob = 0.3  # Team1 starts with 30% chance when chasing
                    
                    # Apply run rate advantage (most critical factor)
                    base_chase_prob += chase_advantage * 0.4  # Up to 40% bonus from run rate
                    
                    # Apply wicket factor (critical for chasing)
                    base_chase_prob *= wicket_factor  # Can reduce to 10% if few wickets
                    
                    # Apply overs factor (time pressure)
                    base_chase_prob *= (0.7 + overs_factor * 0.3)  # Time pressure factor
                    
                    # Combine with ground/team factors
                    final_team1_prob = base_chase_prob * 0.6 + ground_adjusted_prob * 0.4
                    
                else:
                    # Team2 is chasing, Team1 has set the target
                    # Start with Team1 having advantage (they set the target)
                    base_defend_prob = 0.7  # Team1 starts with 70% chance when defending
                    
                    # Reduce by chasing team's advantage
                    base_defend_prob -= chase_advantage * 0.3  # Lose up to 30% to chase advantage
                    
                    # Reduce by wicket factor (chasing team's wickets help them)
                    base_defend_prob *= (1.5 - wicket_factor * 0.5)  # Can lose up to 50% if chasing team has many wickets
                    
                    # Reduce by overs factor (more time = better for chasing team)
                    base_defend_prob *= (1.2 - overs_factor * 0.2)  # Time pressure helps chasing team
                    
                    # Combine with ground/team factors
                    final_team1_prob = base_defend_prob * 0.6 + ground_adjusted_prob * 0.4
                
                # Ensure probabilities are within reasonable bounds
                team1_prob = min(95, max(5, final_team1_prob * 100))
                team2_prob = 100 - team1_prob
            
            # Display results
            results_col1, results_col2 = st.columns(2)
            with results_col1:
                st.metric(f"{team1} Win Probability", f"{team1_prob:.1f}%")
            with results_col2:
                st.metric(f"{team2} Win Probability", f"{team2_prob:.1f}%")
            
            # Visualization
            st.subheader("📊 Win Probability Chart")
            chart_data = pd.DataFrame({
                'Team': [team1, team2],
                'Win Probability': [team1_prob, team2_prob]
            })
            st.bar_chart(chart_data.set_index('Team'))
            
            # Enhanced insights
            st.subheader("💡 Detailed Match Analysis")
            
            if runs_needed <= 0:
                st.success(f"🎉 {batting_team} has already won the match!")
            else:
                # Show analysis factors
                analysis_col1, analysis_col2 = st.columns(2)
                
                with analysis_col1:
                    st.write("**📊 Key Factors:**")
                    st.write(f"• **Head-to-Head:** {team1} has {h2h_team1_advantage*100:.1f}% historical advantage")
                    st.write(f"• **Ground Advantage:** {team1} {team1_ground_adv*100:.1f}% vs {team2} {team2_ground_adv*100:.1f}%")
                    st.write(f"• **Run Rate Factor:** {rr_factor:.2f} (>{1.0} favors chasing team)")
                    
                with analysis_col2:
                    st.write("**🎯 Match Situation:**")
                    
                    # Enhanced wicket analysis display
                    if wickets_left <= 5:
                        wickets_lost_after_5 = 5 - wickets_left
                        if wickets_lost_after_5 > 0:
                            st.write(f"• **Wickets:** {wickets_left}/10 ⚠️ **{wickets_lost_after_5} lost after 5th**")
                        else:
                            st.write(f"• **Wickets:** {wickets_left}/10 ✅ **Critical threshold**")
                    else:
                        extra_wickets = wickets_left - 5
                        st.write(f"• **Wickets:** {wickets_left}/10 🔥 **+{extra_wickets} bonus wickets**")
                    
                    st.write(f"• **Overs Remaining:** {overs_left:.1f}/50 ({overs_left*2:.0f}%)")
                    st.write(f"• **Chase Position:** {chase_advantage*100:.1f}% favorable")
                    st.write(f"• **Wicket Impact:** {wicket_factor:.2f}x factor")
                
                # Overall prediction
                if team1_prob > 70:
                    st.info(f"💪 **{team1}** is strongly favored to win ({team1_prob:.1f}%)")
                elif team2_prob > 70:
                    st.info(f"💪 **{team2}** is strongly favored to win ({team2_prob:.1f}%)")
                elif team1_prob > 60:
                    st.warning(f"⚖️ **{team1}** has a slight advantage ({team1_prob:.1f}%)")
                elif team2_prob > 60:
                    st.warning(f"⚖️ **{team2}** has a slight advantage ({team2_prob:.1f}%)")
                else:
                    st.info("⚖️ The match is very closely balanced")
                
                # Specific insights with enhanced wicket analysis
                if wickets_left <= 3:
                    st.error(f"🚨 **Critical Situation!** Only {wickets_left} wickets left - very difficult to win!")
                elif wickets_left == 4:
                    st.error("⚠️ **Danger Zone!** 4 wickets lost - need to be very careful!")
                elif wickets_left == 5:
                    st.warning("⚠️ **Critical Threshold!** At 5 wickets - each wicket lost now hurts badly!")
                elif wickets_left >= 8:
                    st.success("🔥 **Excellent Position!** Plenty of wickets in hand - can play aggressively!")
                elif rr_factor > 1.2 and wickets_left >= 6:
                    st.success("🔥 **Perfect Chase!** Strong run rate with plenty of wickets!")
                elif rr_factor < 0.8 and wickets_left <= 5:
                    st.error("⚠️ **Tough Chase!** Poor run rate + few wickets = very difficult!")
                elif overs_left < 10 and runs_needed > 50:
                    st.warning("⏰ **Time Pressure!** Need big hitting in final overs!")
                elif wickets_left <= 5 and runs_needed > 100:
                    st.error("💀 **Mission Impossible!** Too many runs needed with few wickets!")

with col2:
    st.header("📋 Progress")
    
    # Progress indicators
    if teams_selected:
        st.success("✅ Teams Selected")
    else:
        st.info("⏳ Select Teams")
    
    if teams_selected and venue_selected:
        st.success("✅ Venue Selected")
    else:
        st.info("⏳ Select Venue")
    
    if teams_selected and venue_selected and toss_complete:
        st.success("✅ Toss Complete")
    else:
        st.info("⏳ Complete Toss")
    
    if teams_selected and venue_selected and toss_complete:
        st.success("✅ Ready to Predict")
    else:
        st.info("⏳ Complete Setup")
    
    # Reset button
    if st.button("🔄 Reset All", key="reset_btn"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Footer
st.markdown("---")
st.markdown("**🏏 Cricket Win Predictor** - Advanced Analytics & Progressive Interface")