"""
Data loading and preprocessing module for Cricket Win Predictor.
Handles loading, cleaning, and basic preprocessing of cricket match data.
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, Optional


class CricketDataLoader:
    """Handles loading and preprocessing of cricket match data."""
    
    def __init__(self, data_dir: str = "datasets"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing the dataset files
        """
        self.data_dir = data_dir
        self.match_data = None
        self.match_info = None
        self.processed_data = None
        
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load raw cricket match data from CSV files.
        
        Returns:
            Tuple of (match_data, match_info) DataFrames
        """
        try:
            # Load ball-by-ball match data
            match_data_path = os.path.join(self.data_dir, "ODI_Match_Data.csv")
            self.match_data = pd.read_csv(match_data_path)
            
            # Load match information
            match_info_path = os.path.join(self.data_dir, "ODI_Match_info.csv")
            self.match_info = pd.read_csv(match_info_path)
            
            print(f"Loaded {len(self.match_data)} ball-by-ball records")
            print(f"Loaded {len(self.match_info)} match records")
            
            return self.match_data, self.match_info
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            raise
    
    def clean_match_data(self) -> pd.DataFrame:
        """
        Clean and preprocess the match data.
        
        Returns:
            Cleaned match data DataFrame
        """
        if self.match_data is None:
            raise ValueError("Match data not loaded. Call load_raw_data() first.")
        
        # Create a copy to avoid modifying original data
        data = self.match_data.copy()
        
        # Fill missing values in extras columns
        data['wides'] = data['wides'].fillna(0)
        data['noballs'] = data['noballs'].fillna(0)
        data['byes'] = data['byes'].fillna(0)
        data['legbyes'] = data['legbyes'].fillna(0)
        data['penalty'] = data['penalty'].fillna(0)
        
        # Calculate total runs for each ball
        data['total_runs'] = (
            data['runs_off_bat'] +
            data['wides'] +
            data['noballs'] +
            data['byes'] +
            data['legbyes'] +
            data['penalty']
        )
        
        # Calculate cumulative score for each innings
        data['cumulative_score'] = (
            data.groupby(['match_id', 'innings'])['total_runs'].cumsum()
        )
        
        # Calculate overs bowled
        data['overs_bowled'] = data['ball'] / 6
        
        # Calculate run rate
        data['run_rate'] = data['cumulative_score'] / data['overs_bowled']
        data['run_rate'] = data['run_rate'].replace([np.inf, -np.inf], 0)
        
        print("Match data cleaned successfully")
        return data
    
    def clean_match_info(self) -> pd.DataFrame:
        """
        Clean and preprocess the match info data.
        
        Returns:
            Cleaned match info DataFrame
        """
        if self.match_info is None:
            raise ValueError("Match info not loaded. Call load_raw_data() first.")
        
        # Create a copy to avoid modifying original data
        info = self.match_info.copy()
        
        # Drop rows with missing winners
        info = info.dropna(subset=['winner'])
        
        # Rename 'id' to 'match_id' for consistency
        info.rename(columns={'id': 'match_id'}, inplace=True)
        
        print(f"Cleaned match info: {len(info)} matches")
        return info
    
    def calculate_team_strengths(self, match_info: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate team strength based on historical win percentage.
        
        Args:
            match_info: Cleaned match info DataFrame
            
        Returns:
            Dictionary mapping team names to their strength scores
        """
        team_stats = {}
        
        # Get unique teams
        teams = pd.concat([match_info['team1'], match_info['team2']]).unique()
        
        for team in teams:
            # Count total matches for this team
            total_matches = len(match_info[
                (match_info['team1'] == team) | (match_info['team2'] == team)
            ])
            
            # Count wins for this team
            wins = len(match_info[match_info['winner'] == team])
            
            # Calculate win percentage
            win_percentage = (wins / total_matches) * 100 if total_matches > 0 else 0
            team_stats[team] = win_percentage
        
        return team_stats
    
    def calculate_venue_averages(self, match_info: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calculate venue-specific scoring averages.
        
        Args:
            match_info: Cleaned match info DataFrame
            
        Returns:
            Dictionary mapping venues to their scoring averages
        """
        venue_data = {}
        unique_venues = match_info['venue'].unique()
        
        # Default averages for venues
        default_avg = {"avg_1st_innings": 250, "avg_2nd_innings": 220}
        
        for venue in unique_venues:
            venue_data[venue] = default_avg.copy()
        
        return venue_data
    
    def create_processed_dataset(self) -> pd.DataFrame:
        """
        Create the final processed dataset for model training.
        
        Returns:
            Processed dataset ready for machine learning
        """
        if self.match_data is None or self.match_info is None:
            raise ValueError("Data not loaded. Call load_raw_data() first.")
        
        # Clean the data
        clean_match_data = self.clean_match_data()
        clean_match_info = self.clean_match_info()
        
        # Calculate team strengths
        team_strengths = self.calculate_team_strengths(clean_match_info)
        
        # Add team strength to match info
        clean_match_info['team1_strength'] = clean_match_info['team1'].map(team_strengths)
        clean_match_info['team2_strength'] = clean_match_info['team2'].map(team_strengths)
        
        # Calculate venue averages
        venue_averages = self.calculate_venue_averages(clean_match_info)
        
        # Add venue averages to match info
        clean_match_info['venue_avg_1st'] = clean_match_info['venue'].map(
            lambda x: venue_averages.get(x, {}).get('avg_1st_innings', 250)
        )
        clean_match_info['venue_avg_2nd'] = clean_match_info['venue'].map(
            lambda x: venue_averages.get(x, {}).get('avg_2nd_innings', 220)
        )
        
        # Calculate match results (1 if team1 wins, 0 if team2 wins)
        clean_match_info['match_result'] = (
            clean_match_info['winner'] == clean_match_info['team1']
        ).astype(int)
        
        # Add basic match statistics
        clean_match_info['team1_wickets'] = 10 - clean_match_info['win_by_wickets'].fillna(0)
        clean_match_info['team2_wickets'] = 10  # Default assumption
        
        # Calculate run rates (simplified)
        clean_match_info['current_run_rate'] = 5.0  # Default
        clean_match_info['required_run_rate'] = 5.0  # Default
        
        self.processed_data = clean_match_info
        
        print(f"Processed dataset created with {len(clean_match_info)} matches")
        return clean_match_info
    
    def get_feature_columns(self) -> list:
        """
        Get the list of feature columns for model training.
        
        Returns:
            List of feature column names
        """
        return [
            'team1_strength', 'team2_strength',
            'venue_avg_1st', 'venue_avg_2nd',
            'team1_wickets', 'team2_wickets',
            'current_run_rate', 'required_run_rate'
        ]
    
    def get_target_column(self) -> str:
        """
        Get the target column name for model training.
        
        Returns:
            Target column name
        """
        return 'match_result'
    
    def load_match_data(self, path: str) -> pd.DataFrame:
        """
        Load match data from CSV or Excel files with standardized column names.
        Handles both .csv and .xlsx files from the datasets/ folder.
        Standardizes all column names (lowercase, underscores).
        Merges venue info from missing_venues.csv if venue is missing.
        
        Args:
            path: Path to the dataset file (relative to data_dir)
            
        Returns:
            Cleaned Pandas DataFrame with standardized column names
        """
        # Construct full path
        full_path = os.path.join(self.data_dir, path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Dataset file not found: {full_path}")
        
        # Load data based on file extension
        file_extension = os.path.splitext(path)[1].lower()
        
        try:
            if file_extension == '.csv':
                df = pd.read_csv(full_path)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(full_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            print(f"Loaded {len(df)} records from {path}")
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            raise
        
        # Standardize column names (lowercase, underscores)
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
        
        print(f"Standardized column names: {list(df.columns)}")
        
        # Handle missing venues if venue column exists
        if 'venue' in df.columns:
            missing_venues_path = os.path.join(self.data_dir, 'missing_venues.csv')
            
            if os.path.exists(missing_venues_path):
                try:
                    # Load missing venues data
                    missing_venues_df = pd.read_csv(missing_venues_path)
                    
                    # Standardize missing venues column names
                    missing_venues_df.columns = missing_venues_df.columns.str.lower().str.replace(' ', '_')
                    
                    # Check if we have venue information to merge
                    if 'missing_venues' in missing_venues_df.columns:
                        # Create a mapping of missing venues to default values
                        venue_mapping = {}
                        for venue in missing_venues_df['missing_venues'].dropna():
                            # Assign default venue characteristics
                            venue_mapping[venue] = {
                                'venue_avg_1st_innings': 250,  # Default first innings average
                                'venue_avg_2nd_innings': 230,  # Default second innings average
                                'venue_type': 'unknown',
                                'venue_capacity': 0,
                                'venue_country': 'unknown'
                            }
                        
                        # Apply venue mapping to fill missing venue data
                        for venue, venue_info in venue_mapping.items():
                            mask = df['venue'] == venue
                            if mask.any():
                                for key, value in venue_info.items():
                                    if key not in df.columns:
                                        df[key] = None
                                    df.loc[mask, key] = value
                        
                        print(f"Applied venue mapping for {len(venue_mapping)} missing venues")
                    
                except Exception as e:
                    print(f"Warning: Could not load missing venues data: {e}")
            else:
                print("No missing_venues.csv found, skipping venue enhancement")
        
        # Clean the data
        df = self._clean_dataframe(df)
        
        print(f"Cleaned dataset: {len(df)} records, {len(df.columns)} columns")
        
        return df
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataframe by handling missing values and data types.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Create a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # Handle missing values in numeric columns
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        cleaned_df[numeric_columns] = cleaned_df[numeric_columns].fillna(0)
        
        # Handle missing values in categorical columns
        categorical_columns = cleaned_df.select_dtypes(include=['object']).columns
        cleaned_df[categorical_columns] = cleaned_df[categorical_columns].fillna('unknown')
        
        # Convert date columns if they exist
        date_columns = [col for col in cleaned_df.columns if 'date' in col.lower()]
        for col in date_columns:
            try:
                cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
            except:
                pass  # Skip if conversion fails
        
        # Remove completely empty rows
        cleaned_df = cleaned_df.dropna(how='all')
        
        # Remove duplicate rows
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        if len(cleaned_df) < initial_rows:
            print(f"Removed {initial_rows - len(cleaned_df)} duplicate rows")
        
        return cleaned_df

