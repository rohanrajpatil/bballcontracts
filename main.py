import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# STEP 1: LOAD AND CLEAN SALARY CAP DATA
# ==========================================
# Load the file (assuming no header based on your file)
cap_df = pd.read_csv('Salarycap.csv', header=None, names=['Year_Str', 'Cap', 'Inflation_Cap'])

def clean_year(year_str):
    # Converts "1984-85" -> 1985
    try:
        return int(year_str.split('-')[0]) + 1
    except:
        return None

def clean_money(money_str):
    # Converts "$3,600,000 " -> 3600000.0
    if isinstance(money_str, str):
        return float(money_str.replace('$', '').replace(',', '').strip())
    return money_str

cap_df['Season'] = cap_df['Year_Str'].apply(clean_year)
cap_df['Cap_Value'] = cap_df['Cap'].apply(clean_money)

# Create a lookup dictionary for fast mapping (Year -> Cap)
cap_lookup = dict(zip(cap_df['Season'], cap_df['Cap_Value']))

# ==========================================
# STEP 2: LOAD AND CLEAN PLAYER SALARIES
# ==========================================
salary_df = pd.read_csv('salaries.csv')

# Handle duplicate rows (e.g., if a player was waived and re-signed, sum the salary)
salary_df = salary_df.groupby(['Player', 'Season'])['Salary'].sum().reset_index()

# Map the Salary Cap to the salary year
salary_df['Salary_Cap'] = salary_df['Season'].map(cap_lookup)

# DROP rows where we don't have cap data (e.g., very old years)
salary_df = salary_df.dropna(subset=['Salary_Cap'])

# CALCULATE THE TARGET: % of Cap
salary_df['Cap_Percent'] = salary_df['Salary'] / salary_df['Salary_Cap']

# ==========================================
# STEP 3: LOAD AND CLEAN STATISTICS
# ==========================================
stats_df = pd.read_csv('stats.csv')

# Handle Traded Players:
# If a player has a 'TOT' (Total) row, keep that. If not, keep the row with most games.
stats_df['is_TOT'] = stats_df['tm'] == 'TOT'
# Sort so TOT and high games are at the top
stats_df = stats_df.sort_values(by=['player', 'season', 'is_TOT', 'g'], ascending=[True, True, False, False])
# Drop duplicates, keeping the top entry (the TOT row)
stats_cleaned = stats_df.drop_duplicates(subset=['player', 'season'], keep='first')

# ==========================================
# STEP 4: MERGE (THE TIME SHIFT)
# ==========================================
# Logic: Stats from 2023 (season) predict Salary in 2024 (Season)
stats_cleaned['Next_Season'] = stats_cleaned['season'] + 1

merged_df = pd.merge(
    stats_cleaned,
    salary_df,
    left_on=['player', 'Next_Season'],
    right_on=['Player', 'Season'],
    how='inner'
)

print(f"Dataset ready! Successfully merged {len(merged_df)} player-seasons.")

# ==========================================
# STEP 5: TRAIN THE MODEL
# ==========================================
# Define the features (X) and target (y)
features = ['age', 'g', 'mp', 'per', 'ts_percent', 'usg_percent', 'ws', 'bpm', 'vorp']
target = 'Cap_Percent'

# Fill missing stats with 0 (e.g., if they didn't play enough to have a stat)
X = merged_df[features].fillna(0)
y = merged_df[target]

# Split into Training (80%) and Testing (20%) data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==========================================
# STEP 6: EVALUATE PERFORMANCE
# ==========================================
predictions = model.predict(X_test)
error = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Convert error to 2025 dollars for context
current_cap = 140588000
dollar_error = error * current_cap

print("-" * 30)
print(f"Model Accuracy (R²): {r2:.3f} (1.0 is perfect)")
print(f"Average Error (Cap %): {error:.2%}")
print(f"Average Error ($): ${dollar_error:,.0f} (in today's money)")
print("-" * 30)

# Check Feature Importance (What matters most?)
importances = pd.DataFrame({'Stat': features, 'Weight': model.feature_importances_})
print(importances.sort_values(by='Weight', ascending=False))