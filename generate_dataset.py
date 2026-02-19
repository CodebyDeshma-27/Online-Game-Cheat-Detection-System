"""
generate_dataset.py
Run this FIRST to create gaming_dataset.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)

def generate_dataset():
    # ---- LEGITIMATE PLAYERS (800) ----
    n_legit = 800
    legit = pd.DataFrame({
        'PlayerID':              [f'P{str(i).zfill(4)}' for i in range(n_legit)],
        'Age':                   np.random.randint(13, 45, n_legit),
        'Gender':                np.random.choice(['Male', 'Female', 'Other'], n_legit),
        'Location':              np.random.choice(['USA', 'India', 'UK', 'Germany', 'Brazil'], n_legit),
        'GameGenre':             np.random.choice(['FPS', 'MOBA', 'Battle Royale', 'RPG', 'Sports'], n_legit),
        'PlayTimeHours':         np.random.normal(25, 10, n_legit).clip(1, 60),
        'SessionDurationMinutes':np.random.normal(45, 15, n_legit).clip(5, 120),
        'KillDeathRatio':        np.random.normal(1.2, 0.5, n_legit).clip(0.1, 4),
        'HeadshotPercentage':    np.random.normal(30, 10, n_legit).clip(0, 60),
        'MovementSpeed':         np.random.normal(5.5, 1.0, n_legit).clip(2, 9),
        'ReactionTimeMs':        np.random.normal(220, 50, n_legit).clip(100, 500),
        'AimAccuracy':           np.random.normal(55, 15, n_legit).clip(10, 85),
        'ActionsPerMinute':      np.random.normal(80, 20, n_legit).clip(10, 150),
        'NetworkLatencyMs':      np.random.normal(60, 20, n_legit).clip(10, 200),
        'PacketLossPercentage':  np.random.normal(0.5, 0.5, n_legit).clip(0, 3),
        'InGamePurchases':       np.random.choice([0, 1], n_legit, p=[0.4, 0.6]),
        'AchievementRate':       np.random.normal(45, 20, n_legit).clip(0, 90),
        'EngagementLevel':       np.random.choice(['Low', 'Medium', 'High'], n_legit, p=[0.2, 0.5, 0.3]),
        'Label':                 'Legitimate'
    })

    # ---- CHEATING PLAYERS (200) ----
    n_cheat = 200
    cheaters = pd.DataFrame({
        'PlayerID':              [f'P{str(i).zfill(4)}' for i in range(n_legit, n_legit + n_cheat)],
        'Age':                   np.random.randint(13, 35, n_cheat),
        'Gender':                np.random.choice(['Male', 'Female', 'Other'], n_cheat),
        'Location':              np.random.choice(['USA', 'India', 'UK', 'Germany', 'Brazil'], n_cheat),
        'GameGenre':             np.random.choice(['FPS', 'MOBA', 'Battle Royale', 'RPG', 'Sports'], n_cheat),
        'PlayTimeHours':         np.random.normal(20, 8, n_cheat).clip(1, 60),
        'SessionDurationMinutes':np.random.normal(90, 20, n_cheat).clip(30, 180),
        'KillDeathRatio':        np.random.normal(8.0, 2.0, n_cheat).clip(4, 20),   # suspiciously high
        'HeadshotPercentage':    np.random.normal(85, 8, n_cheat).clip(70, 100),    # near perfect
        'MovementSpeed':         np.random.normal(12.0, 2.0, n_cheat).clip(8, 20), # speed hack
        'ReactionTimeMs':        np.random.normal(30, 10, n_cheat).clip(5, 80),    # inhuman
        'AimAccuracy':           np.random.normal(95, 3, n_cheat).clip(85, 100),   # aimbot
        'ActionsPerMinute':      np.random.normal(350, 50, n_cheat).clip(200, 500),# macro
        'NetworkLatencyMs':      np.random.normal(200, 80, n_cheat).clip(50, 500),
        'PacketLossPercentage':  np.random.normal(3.0, 1.5, n_cheat).clip(0, 10),
        'InGamePurchases':       np.random.choice([0, 1], n_cheat, p=[0.7, 0.3]),
        'AchievementRate':       np.random.normal(95, 3, n_cheat).clip(85, 100),
        'EngagementLevel':       np.random.choice(['Low', 'Medium', 'High'], n_cheat, p=[0.1, 0.3, 0.6]),
        'Label':                 'Cheating'
    })

    df = pd.concat([legit, cheaters], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv('gaming_dataset.csv', index=False)

    print("=" * 50)
    print("  Dataset Created: gaming_dataset.csv")
    print("=" * 50)
    print(f"  Total players : {len(df)}")
    print(f"  Legitimate    : {n_legit}")
    print(f"  Cheating      : {n_cheat}")
    print("  Run train_model.py next!")

if __name__ == "__main__":
    generate_dataset()
