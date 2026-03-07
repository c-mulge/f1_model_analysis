import requests
import pandas as pd

BASE_URL='https://api.jolpi.ca/ergast/f1/'


def _safe_int(value, default=None):
    try:
        if value is None:
            return default
        return int(value)
    except (ValueError, TypeError):
        return default


def _safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

def fetch_race_result(year):
    url=f"{BASE_URL}/{year}/results.json?limit=1000"
    try:
        res = requests.get(url, timeout=20)
        res.raise_for_status()
        data = res.json()
    except Exception:
        return pd.DataFrame()

    races = data.get('MRData', {}).get('RaceTable', {}).get('Races', [])
    records = []

    for race in races:
        race_name = race.get('raceName')
        circuit = race.get('Circuit', {}).get('circuitName')

        for result in race.get('Results', []):
            driver = result.get('Driver', {}).get('familyName')
            constructor = result.get('Constructor', {}).get('name')

            grid = _safe_int(result.get('grid'))
            position = _safe_int(result.get('position'))
            points = _safe_float(result.get('points'), 0.0)
            laps = _safe_int(result.get('laps'))

            records.append({
                "season": year,
                "race": race_name,
                "circuit": circuit,
                "driver": driver,
                "constructor": constructor,
                "grid": grid,
                "position": position,
                "points": points,
                "laps": laps
            })

    return pd.DataFrame(records)



def collect_mulit_season(start= 2015, end=2024):
    df_list=[]

    for year in range(start, end+1):
        print(f"Fetching {year}")
        try:
            df = fetch_race_result(year)
        except Exception:
            df = pd.DataFrame()

        if df is None or df.empty:
            # skip years where fetch failed/returned nothing
            continue

        df_list.append(df)

    if not df_list:
        return pd.DataFrame()

    return pd.concat(df_list, ignore_index=True)


# backwards-compatible alias (typo preserved)
def collect_multi_season(start=2015, end=2024):
    return collect_mulit_season(start=start, end=end)