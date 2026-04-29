def create_features(df):
    # Time features
    if 'date' in df.columns:
        df['day_of_week'] = df['date'].dt.dayofweek
        df['hour'] = df['date'].dt.hour
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Weather features
    if 'weather' in df.columns:
        df['rain_flag'] = df['weather'].str.contains('rain', case=False).astype(int)

    # Event features
    if 'event' in df.columns:
        df['event_flag'] = (df['event'] != 'None').astype(int)

    return df