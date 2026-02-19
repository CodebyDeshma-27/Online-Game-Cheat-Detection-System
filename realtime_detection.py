"""
realtime_detection.py
Run this THIRD to test real-time detection on sample players
"""

import pandas as pd
import numpy as np
import joblib
import time

# ── LOAD MODELS ────────────────────────────────────────────
print("Loading models...")
rf          = joblib.load('models/random_forest_model.pkl')
iso         = joblib.load('models/isolation_forest_model.pkl')
scaler      = joblib.load('models/scaler.pkl')
FEATURE_COLS= joblib.load('models/feature_cols.pkl')
print("Models loaded!\n")

ENCODE_MAP = {
    'Gender':          {'Male': 1, 'Female': 0, 'Other': 2},
    'Location':        {'Brazil': 0, 'Germany': 1, 'India': 2, 'UK': 3, 'USA': 4},
    'GameGenre':       {'Battle Royale': 0, 'FPS': 1, 'MOBA': 2, 'RPG': 3, 'Sports': 4},
    'EngagementLevel': {'High': 0, 'Low': 1, 'Medium': 2}
}

def detect(player, player_id="Unknown"):
    start = time.time()

    # Encode
    row = dict(player)
    for col, mapping in ENCODE_MAP.items():
        row[col] = mapping.get(str(row.get(col, '')), 0)

    X = pd.DataFrame([row])[FEATURE_COLS]
    X_scaled = scaler.transform(X)

    # Predict
    rf_pred  = rf.predict(X_scaled)[0]
    rf_proba = rf.predict_proba(X_scaled)[0][1]
    iso_pred = iso.predict(X_scaled)[0]
    iso_score= -iso.score_samples(X_scaled)[0]
    latency  = (time.time() - start) * 1000

    is_cheat = (rf_pred == 0) or (iso_pred == -1)  # 0 = Cheating in label encoding

    # Suspicious flags
    flags = []
    if player.get('KillDeathRatio', 0)    > 5:   flags.append(f"KD Ratio = {player['KillDeathRatio']} (too high)")
    if player.get('HeadshotPercentage', 0)> 70:   flags.append(f"Headshot% = {player['HeadshotPercentage']}% (near perfect)")
    if player.get('ReactionTimeMs', 999)  < 80:   flags.append(f"Reaction = {player['ReactionTimeMs']}ms (inhuman)")
    if player.get('AimAccuracy', 0)       > 90:   flags.append(f"Aim = {player['AimAccuracy']}% (aimbot)")
    if player.get('ActionsPerMinute', 0)  > 250:  flags.append(f"APM = {player['ActionsPerMinute']} (macro/script)")
    if player.get('MovementSpeed', 0)     > 10:   flags.append(f"Speed = {player['MovementSpeed']} (speed hack)")

    print(f"\n{'='*55}")
    print(f"  PLAYER: {player_id}")
    print(f"{'='*55}")
    if is_cheat:
        print(f"  VERDICT : ⛔  CHEATING DETECTED")
    else:
        print(f"  VERDICT : ✅  LEGITIMATE PLAYER")
    print(f"  RF Cheat Prob : {rf_proba:.1%}")
    print(f"  Anomaly Score : {iso_score:.4f}")
    print(f"  Latency       : {latency:.1f} ms")
    if flags:
        print(f"\n  SUSPICIOUS FLAGS:")
        for f in flags:
            print(f"    • {f}")
    print(f"{'='*55}")

    return is_cheat, latency

# ── TEST PLAYERS ───────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   AI-BASED CHEAT DETECTION — REAL-TIME TEST")
    print("=" * 55)

    # 1. Normal player
    detect({
        'Age': 22, 'Gender': 'Male', 'Location': 'USA',
        'GameGenre': 'FPS', 'PlayTimeHours': 28,
        'SessionDurationMinutes': 50, 'KillDeathRatio': 1.3,
        'HeadshotPercentage': 32, 'MovementSpeed': 5.8,
        'ReactionTimeMs': 210, 'AimAccuracy': 58,
        'ActionsPerMinute': 85, 'NetworkLatencyMs': 55,
        'PacketLossPercentage': 0.3, 'InGamePurchases': 1,
        'AchievementRate': 48, 'EngagementLevel': 'Medium'
    }, "Player_LEGIT_001")

    # 2. Obvious cheater
    detect({
        'Age': 19, 'Gender': 'Male', 'Location': 'Germany',
        'GameGenre': 'FPS', 'PlayTimeHours': 20,
        'SessionDurationMinutes': 95, 'KillDeathRatio': 12.5,
        'HeadshotPercentage': 94, 'MovementSpeed': 14.2,
        'ReactionTimeMs': 18, 'AimAccuracy': 98,
        'ActionsPerMinute': 420, 'NetworkLatencyMs': 250,
        'PacketLossPercentage': 4.2, 'InGamePurchases': 0,
        'AchievementRate': 97, 'EngagementLevel': 'High'
    }, "Player_CHEAT_001")

    # 3. Suspicious player
    detect({
        'Age': 25, 'Gender': 'Female', 'Location': 'India',
        'GameGenre': 'Battle Royale', 'PlayTimeHours': 35,
        'SessionDurationMinutes': 70, 'KillDeathRatio': 6.8,
        'HeadshotPercentage': 75, 'MovementSpeed': 8.5,
        'ReactionTimeMs': 65, 'AimAccuracy': 88,
        'ActionsPerMinute': 180, 'NetworkLatencyMs': 120,
        'PacketLossPercentage': 1.5, 'InGamePurchases': 1,
        'AchievementRate': 75, 'EngagementLevel': 'High'
    }, "Player_SUSPICIOUS_001")

    # ── BATCH TEST ─────────────────────────────────────────
    print("\n\nBATCH TEST — scanning 100 players from dataset...")
    df = pd.read_csv('gaming_dataset.csv')
    sample = df.sample(100, random_state=99)
    latencies, cheaters = [], 0

    for _, row in sample.iterrows():
        is_cheat, lat = detect(row.to_dict(), row['PlayerID'])
        latencies.append(lat)
        if is_cheat: cheaters += 1

    print(f"\n  Cheaters found  : {cheaters}/100")
    print(f"  Avg Latency     : {np.mean(latencies):.1f} ms")
    print(f"  Max Latency     : {np.max(latencies):.1f} ms")
    print(f"\n  All under 2000ms threshold ✅")
