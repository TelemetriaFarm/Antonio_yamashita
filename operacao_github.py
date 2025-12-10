import pandas as pd
import numpy as np
import json
import requests
import time
from datetime import datetime, timedelta
from pytz import timezone
import math
from collections import Counter
import locale
import os
import sys

# --- CONFIGURAÇÕES FIXAS (PARA O GITHUB) ---
GROWER_ID_FIXO = 1139788
# Dias para analisar para trás (ex: 5 dias)
DIAS_ANALISE = 7 

# Dados da Estação (Hardcoded para não precisar subir Excel)
ESTACOES_FIXAS = [
    {'id_grower': 1139788, 'name': 'Fazenda Guará', 'id_estacao': 52944, 'latitude': -21.6533, 'longitude': -55.4610}
]

# --- IMPORTAÇÃO DA AUTENTICAÇÃO ---
try:
    from farm_auth import get_authenticated_session
except ImportError:
    print("❌ ERRO CRÍTICO: Não foi possível encontrar o arquivo 'farm_auth.py'.")
    sys.exit(1)

# --- CONFIGURAÇÃO DE LOCALE ---
try:
    locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_TIME, 'Portuguese_Brazil.1252')
    except locale.Error:
        print("AVISO: Locale português não disponível. Usando padrão.")

# --- BIBLIOTECAS GEOGRÁFICAS ---
try:
    from shapely.geometry import LineString
    from shapely.ops import unary_union
    from pyproj import Transformer, CRS
    from sklearn.cluster import DBSCAN
except ImportError:
    print("❌ ERRO: Bibliotecas geográficas não encontradas. Verifique requirements.txt")
    sys.exit(1)

# --- CONSTANTES GLOBAIS ---
VELOCIDADE_LIMITE_PARADO = 0.1
DURACAO_MINIMA_PARADA_SEG = 90
MAX_PULSE_GAP_SECONDS = 180
AREA_MINIMA_BLOCO_HA = 4.0

class AnalisadorTelemetriaClima:
    def __init__(self, session: requests.Session):
        self.session = session
        
        # URLs da API
        self.assets_url = "https://admin.farmcommand.com/asset/?season=1083" 
        self.field_border_url = "https://admin.farmcommand.com/fieldborder/?assetID={}&format=json"
        self.canplug_url = "https://admin.farmcommand.com/canplug/?growerID={}"
        self.canplug_iot_url = "https://admin.farmcommand.com/canplug/iot/{}/?format=json"
        self.weather_url_base = "https://admin.farmcommand.com/weather/{}/historical-summary-hourly/"

        # Cache e Dados
        self.cache_limites_talhoes = {}
        self.machine_types_map = {}
        self.machine_names_map = {}
        self.fuso_horario_cuiaba = timezone('America/Cuiaba')
        self.operacoes_definidas = []
        
        # Carrega estações fixas (substitui o Excel)
        self.estacoes_climaticas = pd.DataFrame(ESTACOES_FIXAS)

    def _get_estacoes_para_produtor(self, grower_id: int) -> pd.DataFrame:
        if self.estacoes_climaticas.empty: return pd.DataFrame()
        return self.estacoes_climaticas[self.estacoes_climaticas['id_grower'] == grower_id].copy()

    def _buscar_dados_climaticos_para_produtor(self, grower_id: int, start_date: str, end_date: str) -> pd.DataFrame:
        estacoes_produtor = self._get_estacoes_para_produtor(grower_id)
        if estacoes_produtor.empty:
            print(f"AVISO: Nenhuma estação configurada.")
            return pd.DataFrame()

        all_dfs = []
        for _, station in estacoes_produtor.iterrows():
            station_id = int(station['id_estacao'])
            station_name = station['name']
            print(f"\n--- Buscando Clima: {station_name} (ID: {station_id}) ---")
            
            raw_data = self._buscar_dados_climaticos_por_estacao(str(station_id), start_date, end_date)
            if raw_data:
                df_station = self._processar_clima_para_dataframe(raw_data, station_id, station_name)
                all_dfs.append(df_station)
        
        if not all_dfs: return pd.DataFrame()
        return pd.concat(all_dfs, ignore_index=True)

    def _buscar_dados_climaticos_por_estacao(self, station_id: str, start_date: str, end_date: str) -> list:
        all_results = []
        current_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        final_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        while current_date <= final_date:
            api_start = current_date.strftime('%Y-%m-%dT00:00:00')
            api_end = current_date.strftime('%Y-%m-%dT23:59:59')
            url = self.weather_url_base.format(station_id)
            params = {'startDate': api_start, 'endDate': api_end, 'format': 'json'}
            json_data = self._fazer_requisicao(url, params=params)
            
            if json_data and 'results' in json_data:
                all_results.extend(json_data['results'])
            
            time.sleep(0.1)
            current_date += timedelta(days=1)
            
        return all_results

    def _processar_clima_para_dataframe(self, json_list: list, station_id: int, station_name: str) -> pd.DataFrame:
        if not json_list: return pd.DataFrame()

        records = [{
            'datetime_utc': r.get('local_time'),
            'station_id': station_id,
            'nome_estacao': station_name,
            'temp_c': r.get('avg_temp_c'),
            'umidade_relativa': r.get('avg_relative_humidity'),
            'vento_kph': r.get('avg_windspeed_kph'),
            'delta_t': r.get('avgDeltaT'),
            'rajada_vento_kph': r.get('avgWindGust')
        } for r in json_list]
        
        df = pd.DataFrame(records)
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], errors='coerce', utc=True)
        df = df.dropna(subset=['datetime_utc'])
        
        for col in ['temp_c', 'umidade_relativa', 'vento_kph', 'delta_t', 'rajada_vento_kph']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['datetime_local'] = df['datetime_utc'].dt.tz_convert(self.fuso_horario_cuiaba)
        df['merge_date'] = df['datetime_local'].dt.date
        df['merge_hour'] = df['datetime_local'].dt.hour
        return df

    def _fazer_requisicao(self, url: str, params: dict = None) -> dict | list | None:
        for tentativa in range(2):
            try:
                response = self.session.get(url, params=params, timeout=180)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code in [401, 403]:
                    print("Sessão expirada. Re-autenticando...")
                    self.session = get_authenticated_session()
                    if not self.session: return None
                    continue
                if tentativa == 0: time.sleep(5)
            except json.JSONDecodeError: return None
        return None

    def buscar_dados_telemetria(self, implement_id: int, data_inicio: str, data_fim: str) -> dict | None:
        url = "https://admin.farmcommand.com/canplug/historical-summary/"
        params = {'endDate': data_fim, 'format': 'json', 'implementID': implement_id, 'startDate': data_inicio}
        return self._fazer_requisicao(url, params)

    def _buscar_dados_telemetria_com_chunking(self, implement_id: int, start_date_str: str, end_date_str: str) -> pd.DataFrame:
        all_dfs = []
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        current_date = start_date
        
        print(f"  Buscando dados de hora em hora (ID: {implement_id})...")

        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            horas = range(24)
            for h in horas:
                start_time = f"{h:02d}:00:00"
                end_time = f"{h:02d}:59:59"
                api_start = f"{date_str}T{start_time}"
                api_end = f"{date_str}T{end_time}"
                
                sucesso = False
                tentativas = 0
                while not sucesso and tentativas < 3:
                    try:
                        if tentativas > 0: time.sleep(3)
                        json_data = self.buscar_dados_telemetria(implement_id, api_start, api_end)
                        if json_data:
                            df_chunk = self._analisar_json_telemetria_para_df(json_data, implement_id)
                            if not df_chunk.empty:
                                all_dfs.append(df_chunk)
                                sys.stdout.write(".") 
                            else:
                                sys.stdout.write("_")
                        else:
                            sys.stdout.write("x")
                        sys.stdout.flush()
                        sucesso = True
                    except Exception:
                        tentativas += 1
                        time.sleep(2)
            current_date += timedelta(days=1)
            
        if not all_dfs: return pd.DataFrame()
        print("\n  Consolidando dados...")
        df_combined = pd.concat(all_dfs, ignore_index=True)
        if 'Timestamp (sec)' in df_combined.columns:
            df_combined = df_combined.drop_duplicates(subset=['Timestamp (sec)', 'ImplementID'])
        df_combined['Date'] = pd.to_datetime(df_combined['Date'])
        return df_combined.sort_values(by='Timestamp (sec)').reset_index(drop=True)

    def obter_canplugs_por_produtor(self, grower_id: int) -> list:
        dados_canplug = self._fazer_requisicao(self.canplug_url.format(grower_id))
        if not dados_canplug: return []
        for canplug in dados_canplug:
            implements_raw = canplug.get('implements', [])
            implements = json.loads(implements_raw) if isinstance(implements_raw, str) else implements_raw
            if not isinstance(implements, list): implements = [implements]
            implements = [int(imp) for imp in implements if str(imp).isdigit()]
            if not implements: continue
            
            canplug_id = canplug.get('canplugID')
            machine_type, machine_name = 'Desconhecido', 'Desconhecido'
            if canplug_id:
                iot_data = self._fazer_requisicao(self.canplug_iot_url.format(canplug_id))
                if iot_data and isinstance(iot_data.get('installed_in'), dict):
                    installed_in = iot_data['installed_in']
                    machine_name = installed_in.get('name', 'Desconhecido')
                    if isinstance(installed_in.get('machine_type'), dict):
                        machine_type = installed_in['machine_type'].get('label', 'Desconhecido')
            for imp_id in implements:
                self.machine_types_map[imp_id], self.machine_names_map[imp_id] = machine_type, machine_name
        return list(self.machine_names_map.keys())

    def obter_talhoes_por_produtor(self, grower_id: int):
        data = self._fazer_requisicao(self.assets_url)
        if not data: return
        farms = [item["id"] for item in data if item.get("category") == "Farm" and item.get("parent") == grower_id]
        fields = [item for item in data if item.get("category") == "Field" and item.get("parent") in farms]
        for talhao in fields:
            if talhao['id'] not in self.cache_limites_talhoes:
                limite = self.obter_limite_talhao(talhao['id'])
                if limite:
                    limite.update({'field_id': talhao['id'], 'field_name': talhao['label']})
                    self.cache_limites_talhoes[talhao['id']] = limite
                    
    def obter_limite_talhao(self, field_id: int) -> dict | None:
        border_data = self._fazer_requisicao(self.field_border_url.format(field_id))
        if border_data and "shapeData" in border_data[0]:
            try:
                shape_data = json.loads(border_data[0]["shapeData"])
                coords = []
                if shape_data.get("type") == "FeatureCollection":
                    geom = shape_data["features"][0].get("geometry", {})
                    if geom.get("type") == "Polygon":
                        coords = [[c[1], c[0]] for c in geom["coordinates"][0]]
                if coords:
                    return {"coordinates": coords, "field_name": border_data[0].get("label")}
            except Exception: pass
        return None

    @staticmethod
    def _is_ponto_no_poligono(ponto_lat_lon: tuple, coordenadas_poligono: list) -> bool:
        p_lat, p_lon = ponto_lat_lon
        n, dentro = len(coordenadas_poligono), False
        if n < 3: return False
        p1_lat, p1_lon = coordenadas_poligono[0]
        for i in range(n + 1):
            p2_lat, p2_lon = coordenadas_poligono[i % n]
            if p_lat > min(p1_lat, p2_lat) and p_lat <= max(p1_lat, p2_lat) and p_lon <= max(p1_lon, p2_lon) and p1_lat != p2_lat:
                xinters = (p_lat - p1_lat) * (p2_lon - p1_lon) / (p2_lat - p1_lat) + p1_lon
                if p1_lon == p2_lon or p_lon <= xinters: dentro = not dentro
            p1_lat, p1_lon = p2_lat, p2_lon
        return dentro

    def _get_first_valid_value(self, data: dict, keys: list) -> any:
        for key in keys:
            value = data.get(key)
            if value is not None: return value
        return None
    
    def _analisar_json_telemetria_para_df(self, json_data: any, implement_id: int) -> pd.DataFrame:
        features_list = []
        if isinstance(json_data, dict) and 'results' in json_data:
            for resultado in json_data.get('results', []):
                if resultado.get('type') == 'FeatureCollection':
                    features_list.extend(resultado.get('features', []))
        elif isinstance(json_data, list):
            features_list = json_data
        
        if not features_list: return pd.DataFrame()

        registros = []
        for feature in features_list:
            if isinstance(feature, dict):
                coords = None
                if 'geometry' in feature and feature['geometry']:
                    coords = feature['geometry'].get('coordinates')
                
                if not coords or len(coords) < 2: continue
                
                props = feature.get('properties', {}).copy()
                props['Longitude'], props['Latitude'] = coords[0], coords[1]
                
                for key in list(props.keys()):
                    if isinstance(props[key], dict):
                        props[key] = props[key].get('value', props[key].get('status'))

                vel_raw = self._get_first_valid_value(props, ['Computed Velocity (miles/hour)', 'velocity', 'Ground Speed (miles/hour)'])
                vel_kmh = pd.to_numeric(vel_raw, errors='coerce')
                if vel_raw is not None: vel_kmh = float(vel_raw) * 1.60934
                props['velocity'] = vel_kmh if pd.notna(vel_kmh) and vel_kmh <= 60 else np.nan

                props['Fuel Rate (L/h)'] = self._get_first_valid_value(props, ['Fuel Rate (L/h)', 'Fuel Consumption (L/h)', 'Instantaneous Liquid Fuel Usage (L/hour)'])
                
                width_m = self._get_first_valid_value(props, ['machine width (meters)', 'Implement Width (meters)'])
                if width_m is None and 'Header width in use (mm)' in props:
                    width_m = props.get('Header width in use (mm)', 0) / 1000
                props['machine width (meters)'] = width_m
                
                registros.append(props)

        if not registros: return pd.DataFrame()
        df = pd.DataFrame(registros)

        df['Datetime'] = pd.to_datetime(df['Timestamp (sec)'], unit='s', utc=True).dt.tz_convert(self.fuso_horario_cuiaba)
        df['Date'] = df['Datetime'].dt.date
        df['ImplementID'] = implement_id
        df['MachineName'] = self.machine_names_map.get(implement_id, f"ID {implement_id}")
        
        cols_num = ['velocity', 'Fuel Rate (L/h)', 'Engine RPM', 'machine width (meters)']
        for col in cols_num:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['Inside_Field_Name'] = pd.NA
        if self.cache_limites_talhoes:
            df_valid = df.dropna(subset=['Latitude', 'Longitude'])
            for idx, row in df_valid.iterrows():
                for border in self.cache_limites_talhoes.values():
                    if border.get('coordinates') and self._is_ponto_no_poligono((row['Latitude'], row['Longitude']), border['coordinates']):
                        df.loc[idx, 'Inside_Field_Name'] = border['field_name']
                        break
        return df

    @staticmethod
    def _haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371
        lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
        lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def _encontrar_estacao_mais_proxima(self, lat, lon, estacoes_df):
        if estacoes_df.empty or pd.isna(lat) or pd.isna(lon): return None
        distancias = estacoes_df.apply(lambda row: self._haversine_distance(lat, lon, row['latitude'], row['longitude']), axis=1)
        return estacoes_df.loc[distancias.idxmin()]['id_estacao']
        
    def _merge_telemetria_clima(self, df_telemetria: pd.DataFrame, df_clima: pd.DataFrame, estacoes_produtor: pd.DataFrame) -> pd.DataFrame:
        print("\nCruzando dados...")
        if df_clima.empty or df_telemetria.empty:
            df_telemetria['temp_c'] = np.nan
            df_telemetria['umidade_relativa'] = np.nan
            df_telemetria['vento_kph'] = np.nan
            df_telemetria['delta_t'] = np.nan
            df_telemetria['rajada_vento_kph'] = np.nan
            df_telemetria['CondicaoAplicacao'] = 'Sem Dados'
            return df_telemetria
        
        df_telemetria['nearest_station_id'] = df_telemetria.apply(
            lambda row: self._encontrar_estacao_mais_proxima(row['Latitude'], row['Longitude'], estacoes_produtor),
            axis=1
        ).astype('float64')

        df_telemetria['merge_date'] = df_telemetria['Datetime'].dt.date
        df_telemetria['merge_hour'] = df_telemetria['Datetime'].dt.hour
        
        df_merged = pd.merge(
            df_telemetria,
            df_clima[['station_id', 'merge_date', 'merge_hour', 'temp_c', 'umidade_relativa', 'vento_kph', 'delta_t', 'rajada_vento_kph']],
            how='left',
            left_on=['nearest_station_id', 'merge_date', 'merge_hour'],
            right_on=['station_id', 'merge_date', 'merge_hour']
        )
        
        def get_delta_t_state(delta_t_val):
            if pd.isna(delta_t_val): return 'Sem Dados'
            if delta_t_val >= 9: return 'Vermelho'
            if delta_t_val < 2 or (delta_t_val > 8 and delta_t_val < 9): return 'Amarelo'
            if delta_t_val >= 2 and delta_t_val <= 8: return 'Verde'
            return 'Sem Dados'

        def get_vento_state(vento_val):
            if pd.isna(vento_val): return 'Sem Dados'
            if (vento_val >= 0 and vento_val <= 1) or (vento_val > 10): return 'Vermelho'
            if (vento_val > 1 and vento_val <= 2) or (vento_val > 8 and vento_val <= 10): return 'Amarelo'
            if vento_val > 2 and vento_val <= 8: return 'Verde'
            return 'Sem Dados'

        def get_condicao_aplicacao(row):
            vento_state = get_vento_state(row['vento_kph']) 
            delta_t_state = get_delta_t_state(row['delta_t'])
            if vento_state == 'Vermelho' or delta_t_state == 'Vermelho': return 'Evitar'
            if vento_state == 'Verde' and delta_t_state == 'Verde': return 'Ideal'
            if vento_state == 'Amarelo' or delta_t_state == 'Amarelo': return 'Atenção'
            return 'Sem Dados'
            
        df_merged['CondicaoAplicacao'] = df_merged.apply(get_condicao_aplicacao, axis=1)
        return df_merged.drop(columns=['merge_date', 'merge_hour', 'nearest_station_id', 'station_id'])

    def _get_utm_projection(self, df: pd.DataFrame) -> CRS | None:
        if df.empty or df['Longitude'].isnull().all() or df['Latitude'].isnull().all(): return None
        avg_lon = df['Longitude'].mean()
        avg_lat = df['Latitude'].mean()
        utm_zone = math.floor((avg_lon + 180) / 6) + 1
        return CRS(f"EPSG:327{utm_zone}") if avg_lat < 0 else CRS(f"EPSG:326{utm_zone}")

    def _estimar_largura_por_geometria(self, df_maquina: pd.DataFrame) -> float | None:
        df_trabalho = df_maquina[df_maquina['Operating_Mode'] == 'Trabalho Produtivo'].copy()
        if len(df_trabalho) < 50: return None
        try:
            crs_wgs84 = CRS("EPSG:4326")
            crs_utm = self._get_utm_projection(df_trabalho)
            if not crs_utm: return None
            transformer = Transformer.from_crs(crs_wgs84, crs_utm, always_xy=True)
            points_utm = [transformer.transform(lon, lat) for lon, lat in df_trabalho[['Longitude', 'Latitude']].values]
            df_trabalho[['utm_x', 'utm_y']] = points_utm
        except Exception: return None

        coords = df_trabalho[['utm_x', 'utm_y']].values
        df_trabalho['pass_label'] = DBSCAN(eps=10, min_samples=5).fit(coords).labels_
        pass_labels = [l for l in df_trabalho['pass_label'].unique() if l != -1]
        if len(pass_labels) < 2: return None

        pass_lines = {l: LineString(df_trabalho[df_trabalho['pass_label'] == l][['utm_x', 'utm_y']].values) for l in pass_labels if len(df_trabalho[df_trabalho['pass_label'] == l]) > 1}
        measured_widths = []
        labels_list = list(pass_lines.keys())
        for i in range(len(labels_list)):
            for j in range(i + 1, len(labels_list)):
                line1, line2 = pass_lines[labels_list[i]], pass_lines[labels_list[j]]
                for k in range(5):
                    dist = line1.interpolate(k/4, normalized=True).distance(line2)
                    if 5 < dist < 50: measured_widths.append(dist)

        if not measured_widths: return None
        hist, bin_edges = np.histogram(measured_widths, bins=np.arange(min(measured_widths), max(measured_widths) + 1, 0.5))
        if hist.any(): return bin_edges[np.argmax(hist)]
        return None

    def _calcular_area_e_sobreposicao(self, df_pontos_trabalho: pd.DataFrame, largura_maquina_m: float) -> dict:
        if df_pontos_trabalho.empty or pd.isna(largura_maquina_m) or largura_maquina_m <= 0:
            return {'area_ha': 0.0, 'sobreposicao_percent': 0.0, 'area_bruta_ha': 0.0}
        try:
            crs_wgs84 = CRS("EPSG:4326")
            crs_utm = self._get_utm_projection(df_pontos_trabalho)
            if not crs_utm: return {'area_ha': 0.0, 'sobreposicao_percent': 0.0, 'area_bruta_ha': 0.0}

            transformer = Transformer.from_crs(crs_wgs84, crs_utm, always_xy=True)
            all_polygons = []
            df_pontos_trabalho = df_pontos_trabalho.sort_values('Timestamp (sec)')
            df_pontos_trabalho['time_diff'] = df_pontos_trabalho.groupby('ImplementID')['Timestamp (sec)'].diff()
            df_pontos_trabalho['segment_id'] = (df_pontos_trabalho['time_diff'] > MAX_PULSE_GAP_SECONDS).cumsum()

            for _, segment_df in df_pontos_trabalho.groupby(['ImplementID', 'segment_id']):
                if len(segment_df) < 2: continue
                points_utm = [transformer.transform(lon, lat) for lon, lat in segment_df[['Longitude', 'Latitude']].values]
                line = LineString(points_utm).simplify(2.0, preserve_topology=False)
                if line.length > 0:
                    polygon = line.buffer(largura_maquina_m / 2, cap_style=2, join_style=2)
                    all_polygons.append(polygon)

            if not all_polygons: return {'area_ha': 0.0, 'sobreposicao_percent': 0.0, 'area_bruta_ha': 0.0}

            area_bruta_m2 = sum(p.area for p in all_polygons)
            uniao_total = unary_union(all_polygons)
            area_unica_m2 = uniao_total.area
            percentual_sobreposicao = ((area_bruta_m2 - area_unica_m2) / area_unica_m2) * 100 if area_unica_m2 > 0 else 0.0

            return {
                'area_ha': area_unica_m2 / 10000,
                'sobreposicao_percent': percentual_sobreposicao,
                'area_bruta_ha': area_bruta_m2 / 10000
            }
        except Exception:
            return {'area_ha': 0.0, 'sobreposicao_percent': 0.0, 'area_bruta_ha': 0.0}

    def _formatar_duracao(self, segundos: float) -> str:
        if pd.isna(segundos) or segundos < 0: return "0h 00min:00s"
        h, m, s = int(segundos // 3600), int((segundos % 3600) // 60), int(segundos % 60)
        return f"{h}h {m:02d}min:{s:02d}s"

    def _identificar_blocos_operacao(self, df: pd.DataFrame):
        df_trabalho = df[df['Operating_Mode'] == 'Trabalho Produtivo'].copy()
        if df_trabalho.empty: return

        df_trabalho['DateOnly'] = pd.to_datetime(df_trabalho['Date']).dt.date
        df_trabalho = df_trabalho.sort_values(by=['ImplementID', 'Inside_Field_Name', 'DateOnly'])
        df_trabalho['date_diff'] = df_trabalho.groupby(['ImplementID', 'Inside_Field_Name'])['DateOnly'].diff().apply(lambda x: x.days if pd.notna(x) else 0)
        df_trabalho['operation_block'] = (df_trabalho['date_diff'] > 1).cumsum()

        blocos = df_trabalho.groupby(['ImplementID', 'Inside_Field_Name', 'operation_block'])
        op_counter = 1
        for (imp_id, talhao, _), grupo in blocos:
            if pd.isna(talhao): continue
            start_date_str = grupo['DateOnly'].min().strftime('%d/%m')
            maquina = self.machine_names_map.get(imp_id, f"ID {imp_id}")
            nome_bloco = f"Bloco {op_counter}: {maquina.split()[0]} em {talhao} (a partir de {start_date_str})"
            
            self.operacoes_definidas.append({'id': op_counter, 'nome': nome_bloco, 'tipo': "Não definido", 'indices_df': grupo.index.tolist()})
            df.loc[grupo.index, 'OperationID'] = op_counter
            df.loc[grupo.index, 'OperationName'] = nome_bloco
            op_counter += 1

    def executar_relatorio(self):
        # CONFIGURAÇÃO AUTOMÁTICA (Sem input)
        grower_id = GROWER_ID_FIXO
        hoje = datetime.now()
        data_fim = hoje.strftime('%Y-%m-%d')
        data_inicio = (hoje - timedelta(days=DIAS_ANALISE)).strftime('%Y-%m-%d')
        
        print(f"\n--- EXECUÇÃO AUTOMÁTICA ---")
        print(f"Período: {data_inicio} a {data_fim}")

        self.obter_talhoes_por_produtor(grower_id)
        implement_ids = self.obter_canplugs_por_produtor(grower_id)
        
        # Filtra Sprayers
        implement_ids = [mid for mid, mtype in self.machine_types_map.items() if 'Sprayer' in mtype or 'Pulverizador' in mtype]
        if not implement_ids:
            print("Nenhum pulverizador encontrado.")
            return None
        
        print(f"Máquinas: {implement_ids}")

        df_clima_completo = self._buscar_dados_climaticos_para_produtor(grower_id, data_inicio, data_fim)
        estacoes_deste_produtor = self._get_estacoes_para_produtor(grower_id)

        dfs_telemetria = [df for df in [self._buscar_dados_telemetria_com_chunking(imp, data_inicio, data_fim) for imp in implement_ids] if not df.empty]
        if not dfs_telemetria:
            print("Sem dados de telemetria.")
            return None
            
        df_telemetria_completo = pd.concat(dfs_telemetria, ignore_index=True).sort_values('Timestamp (sec)').reset_index(drop=True)
        df_final_combinado = self._merge_telemetria_clima(df_telemetria_completo, df_clima_completo, estacoes_deste_produtor)

        df_final_combinado['Operating_Mode'] = np.select(
            [df_final_combinado['velocity'] <= VELOCIDADE_LIMITE_PARADO, df_final_combinado['Inside_Field_Name'].notna()],
            ['Parado', 'Trabalho Produtivo'], default='Deslocamento'
        )

        for imp_id in df_final_combinado['ImplementID'].unique():
            mask = df_final_combinado['ImplementID'] == imp_id
            larguras_api = df_final_combinado.loc[mask, 'machine width (meters)'].dropna()
            larguras_api = larguras_api[larguras_api > 0.1]
            largura_final = np.nan
            if not larguras_api.empty:
                largura_final = Counter(larguras_api).most_common(1)[0][0]
            else:
                largura_est = self._estimar_largura_por_geometria(df_final_combinado[mask])
                if largura_est: largura_final = largura_est
            df_final_combinado.loc[mask, 'machine width (meters)'] = largura_final

        df_final_combinado['OperationID'] = pd.NA
        df_final_combinado['OperationName'] = pd.NA
        self._identificar_blocos_operacao(df_final_combinado)

        print("\nGerando HTML bonito...")
        self.criar_relatorio_html_unificado(df_final_combinado, data_inicio, data_fim)
        return df_final_combinado

    def criar_relatorio_html_unificado(self, df: pd.DataFrame, data_inicio_str: str, data_fim_str: str):
        if df.empty: return

        df = df.sort_values('Timestamp (sec)')
        df['duration_sec'] = df.groupby('ImplementID')['Timestamp (sec)'].diff().shift(-1)
        med_dur = df.loc[df['duration_sec'] <= MAX_PULSE_GAP_SECONDS, 'duration_sec'].median()
        med_dur = med_dur if pd.notna(med_dur) and med_dur > 0 else 10
        df['duration_sec'] = df['duration_sec'].apply(lambda x: med_dur if pd.isna(x) or x > MAX_PULSE_GAP_SECONDS else x)

        eventos_calendario = []
        df['DateOnly'] = pd.to_datetime(df['Date']).dt.date
        
        for (data_obj, implement_id), df_dia in df.groupby(['DateOnly', 'ImplementID']):
            tempos_seg = {mode: df_dia[df_dia['Operating_Mode'] == mode]['duration_sec'].sum() for mode in ['Trabalho Produtivo', 'Deslocamento', 'Parado']}
            df_prod = df_dia[df_dia['Operating_Mode'] == 'Trabalho Produtivo']
            largura_maq = df_dia['machine width (meters)'].median()
            resultado_area = self._calcular_area_e_sobreposicao(df_prod, largura_maq)

            resumo_climatico = {'delta_t_medio': df_dia['delta_t'].mean(), 'vento_medio': df_dia['vento_kph'].mean()}
            vel_media = df_prod['velocity'].mean()
            vel_str = f"{vel_media:.1f} km/h" if pd.notna(vel_media) else "N/D"
            
            consumo = 0.0
            if 'Fuel Rate (L/h)' in df_dia.columns:
                valid_fuel = df_dia[df_dia['Fuel Rate (L/h)'].notna()]
                consumo = (valid_fuel['Fuel Rate (L/h)'] * (valid_fuel['duration_sec'] / 3600)).sum()
            cons_str = f"{consumo:.1f} L" if consumo > 0 else "N/D"

            dt_str = f"{resumo_climatico['delta_t_medio']:.1f} °C" if pd.notna(resumo_climatico['delta_t_medio']) else "N/D"
            vt_str = f"{resumo_climatico['vento_medio']:.1f} km/h" if pd.notna(resumo_climatico['vento_medio']) else "N/D"

            resumo_html = f"""
            <div class="summary-header">
                <h3>Resumo do Dia</h3>
                <p><strong>Máquina:</strong> {self.machine_names_map.get(implement_id, "N/A")} | <strong>Data:</strong> {data_obj.strftime('%d/%m/%Y')}</p>
                <hr>
                <div class="summary-grid">
                    <div>⏱️ <strong>Produtivo:</strong> {self._formatar_duracao(tempos_seg['Trabalho Produtivo'])}</div>
                    <div>🌱 <strong>Área:</strong> {resultado_area['area_ha']:.2f} ha</div>
                    <div>💨 <strong>Vel. Média:</strong> {vel_str}</div>
                    <div>⛽ <strong>Consumo:</strong> {cons_str}</div>
                </div>
                 <div class="summary-grid-clima">
                    <div>🌡️ <strong>ΔT Médio:</strong> {dt_str}</div>
                    <div>🌬️ <strong>Vento Médio:</strong> {vt_str}</div>
                </div>
            </div>"""

            metric_map = {
                'velocity': {'label': 'Velocidade', 'unit': 'km/h'},
                'Fuel Rate (L/h)': {'label': 'Consumo', 'unit': 'L/h'},
                'delta_t': {'label': 'Delta T', 'unit': '°C'},
                'vento_kph': {'label': 'Vento', 'unit': 'km/h'},
                'CondicaoAplicacao': {'label': 'Condição Aplicação', 'unit': ''},
            }
            available_metrics = {k: v for k, v in metric_map.items() if k in df_dia.columns and df_dia[k].notna().any()}
            
            df_mapa = df_dia[['Latitude', 'Longitude', 'Datetime', 'duration_sec', 'MachineName'] + list(available_metrics.keys())].dropna(subset=['Latitude', 'Longitude']).copy()
            segmentos_mapa = []
            if not df_mapa.empty:
                df_mapa['time_str'] = df_mapa['Datetime'].dt.strftime('%H:%M')
                df_mapa['hour_of_day'] = df_mapa['Datetime'].dt.hour
                for i in range(len(df_mapa) - 1):
                    p1, p2 = df_mapa.iloc[i], df_mapa.iloc[i+1]
                    props = {col: p1[col] for col in available_metrics if pd.notna(p1[col])}
                    props.update({'time': p1['time_str'], 'hour': int(p1['hour_of_day']), 'machine': p1['MachineName']})
                    segmentos_mapa.append({'coords': [[p1['Latitude'], p1['Longitude']], [p2['Latitude'], p2['Longitude']]], 'properties': props})
            
            dados_janela = df_dia[['Datetime', 'duration_sec', 'Operating_Mode', 'CondicaoAplicacao']].copy()
            
            eventos_calendario.append({
                'date': data_obj, 'title': self.machine_names_map.get(implement_id, f"ID {implement_id}"),
                'implement_id': implement_id,
                'summary_html': resumo_html,
                'all_day_segments_json': json.dumps(segmentos_mapa),
                'available_metrics_json': json.dumps(available_metrics),
                'borders_json': json.dumps([{"name": b.get("field_name"), "coords": b.get("coordinates")} for b in self.cache_limites_talhoes.values() if b]),
                'dados_janela_json': dados_janela.to_json(orient='records', date_format='iso')
            })

        self._gerar_html_final_unificado(eventos_calendario, data_inicio_str, data_fim_str)

    def _gerar_html_final_unificado(self, eventos, data_inicio_str, data_fim_str):
        eventos_por_data = {}
        for ev in eventos:
            data_str = ev['date'].strftime('%Y-%m-%d')
            if data_str not in eventos_por_data: eventos_por_data[data_str] = []
            eventos_por_data[data_str].append(ev)

        html_calendario = ""
        data_inicio = datetime.strptime(data_inicio_str, '%Y-%m-%d').date()
        data_fim = datetime.strptime(data_fim_str, '%Y-%m-%d').date()
        data_atual = data_inicio.replace(day=1)

        while data_atual <= data_fim:
            mes_ano_str = data_atual.strftime('%B %Y').capitalize()
            html_calendario += f"<h3>{mes_ano_str}</h3><div class='calendar-grid'>"
            html_calendario += "".join([f"<div class='day-header'>{dia}</div>" for dia in ['Dom', 'Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb']])
            primeiro_dia_semana = (data_atual.weekday() + 1) % 7
            html_calendario += "<div class='day-cell empty'></div>" * primeiro_dia_semana
            mes_atual = data_atual.month
            while data_atual.month == mes_atual and data_atual <= data_fim:
                data_str = data_atual.strftime('%Y-%m-%d')
                eventos_do_dia = eventos_por_data.get(data_str, [])
                html_calendario += f"<div class='day-cell'><div class='day-number'>{data_atual.day}</div>"
                for ev in eventos_do_dia:
                    unique_id = f"{data_str.replace('-','')}_{ev['implement_id']}"
                    html_calendario += f"<div class='event' onclick='showDetails(\"{unique_id}\", this)'>{ev['title']}</div>"
                html_calendario += "</div>"
                data_atual += timedelta(days=1)
            html_calendario += "</div>"

        html_detalhes_eventos = ""
        for data_str, eventos_do_dia in eventos_por_data.items():
            for ev in eventos_do_dia:
                id_evento = f"{data_str.replace('-','')}_{ev['implement_id']}"
                html_detalhes_eventos += f"""
                <div id="{id_evento}" class="event-details-container" style="display:none;"
                             data-all-segments='{ev['all_day_segments_json'].replace("'", "&apos;")}'
                             data-available-metrics='{ev['available_metrics_json'].replace("'", "&apos;")}'
                             data-borders='{ev['borders_json'].replace("'", "&apos;")}'
                             data-janela='{ev['dados_janela_json'].replace("'", "&apos;")}'>
                    {ev['summary_html']}
                    <div class="content-wrapper">
                        <div class="map-section">
                            <div class="map-controls">
                                <div class="filter-group">
                                    <label>Colorir Rastro por:</label>
                                    <select id="metric_selector_{id_evento}" onchange="applyMapFilters('{id_evento}', false)"></select>
                                </div>
                                <div class="filter-group">
                                    <label>Filtrar Hora:</label>
                                    <select id="hour_start_{id_evento}" onchange="applyMapFilters('{id_evento}', true)"></select>
                                    <span>até</span>
                                    <select id="hour_end_{id_evento}" onchange="applyMapFilters('{id_evento}', true)"></select>
                                </div>
                            </div>
                            <div class="legend" id="legend_{id_evento}"></div>
                            <div class="map-container" id="map-container-{id_evento}"></div>
                        </div>
                        <div class="chart-section">
                            <h3>Janela de Pulverização vs. Horas Trabalhadas</h3>
                            <div class="chart-canvas-wrapper"><canvas id="chart-janela-{id_evento}"></canvas></div>
                        </div>
                    </div>
                </div>"""

        # CSS E JS EMBUTIDOS (O mesmo do seu código original)
        css_styles = """
        body { font-family: sans-serif; margin: 20px; background-color: #f4f5f7; }
        .calendar-grid { display: grid; grid-template-columns: repeat(7, 1fr); gap: 5px; }
        .day-cell { border: 1px solid #dfe1e6; min-height: 80px; padding: 5px; background-color: #fff; }
        .day-header { text-align: center; font-weight: bold; padding: 8px; }
        .event { background-color: #e9f2ff; border-left: 3px solid #0052cc; padding: 3px 6px; margin-top: 3px; cursor: pointer; font-size: 0.8em; }
        .event-details-container { border-top: 3px solid #0052cc; padding: 15px; margin-top: 20px; background-color: #fff; }
        .content-wrapper { display: flex; gap: 20px; margin-top: 15px; flex-wrap: wrap; }
        .map-section { flex: 2; min-width: 500px; }
        .chart-section { flex: 1; min-width: 400px; }
        .map-container { height: 500px; width: 100%; border: 1px solid #ccc; }
        .map-controls { background-color: #f8f9fa; padding: 10px; border-bottom: 1px solid #ccc; }
        .legend { padding: 10px 5px 5px 5px; font-size: 0.9em; }
        .legend span { display: inline-block; width: 15px; height: 15px; margin-right: 5px; border: 1px solid #ccc; vertical-align: middle; }
        .summary-header { background-color: #fafafa; padding: 10px; border-radius: 5px; border: 1px solid #eee; }
        .summary-grid, .summary-grid-clima { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; margin-top: 10px; }
        .chart-canvas-wrapper { position: relative; height: 450px; }
        .filter-group { margin-top: 8px; display: flex; align-items: center; }
        .filter-group label { font-weight: bold; margin-right: 5px; font-size: 0.9em; }
        .filter-group select { margin: 0 5px; padding: 2px; }
        """
        
        # JAVASCRIPT GIGANTE (Resumido aqui, ele usa o mesmo do seu código)
        js_script = """
        let maps = {}; let trailLayerGroup = {}; let charts = {}; let activeDetailContainer = null;
        const SPRAY_CONDITIONS_COLOR = { 'Ideal': '#28a745', 'Atenção': '#ffc107', 'Evitar': '#dc3545', 'Sem Dados': '#6c757d' };
        function showDetails(id, clickedElement) {
            const containerToShow = document.getElementById(id);
            if (!containerToShow) return;
            if (activeDetailContainer && activeDetailContainer !== containerToShow) activeDetailContainer.style.display = 'none';
            const isVisible = containerToShow.style.display !== 'none';
            containerToShow.style.display = isVisible ? 'none' : 'block';
            activeDetailContainer = isVisible ? null : containerToShow;
            if (!isVisible) {
                const mapContainerId = `map-container-${id}`;
                if (!maps[id]) {
                    maps[id] = L.map(mapContainerId, { preferCanvas: true });
                    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(maps[id]);
                    trailLayerGroup[id] = L.layerGroup().addTo(maps[id]);
                    drawBorders(id);
                    setupDynamicControls(id); 
                }
                setTimeout(() => { if (maps[id]) { maps[id].invalidateSize(); applyMapFilters(id, true); renderSprayingWindowChart(id); } }, 10);
            }
        }
        function populateHourFilters(id) {
            const startSelect = document.getElementById(`hour_start_${id}`);
            const endSelect = document.getElementById(`hour_end_${id}`);
            startSelect.innerHTML = ''; endSelect.innerHTML = '';
            for (let i = 0; i < 24; i++) {
                const optS = document.createElement('option'); optS.value = i; optS.textContent = `${String(i).padStart(2, '0')}:00`; startSelect.appendChild(optS);
                const optE = document.createElement('option'); optE.value = i; optE.textContent = `${String(i).padStart(2, '0')}:59`; endSelect.appendChild(optE);
            }
            startSelect.value = 0; endSelect.value = 23;
        }
        function setupDynamicControls(id) {
            const detailEl = document.getElementById(id);
            const metrics = JSON.parse(detailEl.getAttribute('data-available-metrics'));
            const metricSelector = document.getElementById(`metric_selector_${id}`);
            metricSelector.innerHTML = '';
            const condOption = document.createElement('option'); condOption.value = 'CondicaoAplicacao'; condOption.textContent = 'Condição de Aplicação'; metricSelector.appendChild(condOption);
            for (const key in metrics) { if (key !== 'CondicaoAplicacao') { const option = document.createElement('option'); option.value = key; option.textContent = metrics[key].label; metricSelector.appendChild(option); } }
            populateHourFilters(id);
        }
        function applyMapFilters(id, fitBounds = true) {
            const selector = document.getElementById(`metric_selector_${id}`);
            const selectedMetric = selector.value;
            const startHour = parseInt(document.getElementById(`hour_start_${id}`).value, 10);
            const endHour = parseInt(document.getElementById(`hour_end_${id}`).value, 10);
            updateLegend(id, selectedMetric);
            try {
                const detailEl = document.getElementById(id);
                const segments = JSON.parse(detailEl.getAttribute('data-all-segments'));
                drawTrails(id, segments, selectedMetric, startHour, endHour, fitBounds);
            } catch (e) { console.error("Erro ao filtrar:", e); }
        }
        function getSemaforicColor(metricKey, value) {
            if (value === null || value === undefined) return SPRAY_CONDITIONS_COLOR['Sem Dados'];
            if (metricKey === 'vento_kph') {
                if ((value >= 0 && value <= 1) || value > 10) return SPRAY_CONDITIONS_COLOR['Evitar'];
                if ((value > 1 && value <= 2) || (value > 8 && value <= 10)) return SPRAY_CONDITIONS_COLOR['Atenção'];
                if (value > 2 && value <= 8) return SPRAY_CONDITIONS_COLOR['Ideal'];
            }
            if (metricKey === 'delta_t') {
                if (value >= 9) return SPRAY_CONDITIONS_COLOR['Evitar'];
                if (value < 2 || (value > 8 && value < 9)) return SPRAY_CONDITIONS_COLOR['Atenção'];
                if (value >= 2 && value <= 8) return SPRAY_CONDITIONS_COLOR['Ideal'];
            }
            return SPRAY_CONDITIONS_COLOR['Sem Dados'];
        }
        function drawTrails(id, segments, metricKey, startHour, endHour, fitBounds = true) {
            const trails = trailLayerGroup[id];
            if (!trails) return;
            trails.clearLayers();
            const metrics = JSON.parse(document.getElementById(id).getAttribute('data-available-metrics'));
            const unit = metrics[metricKey]?.unit || '';
            let bounds = L.latLngBounds();
            let min, max;
            if (metricKey !== 'CondicaoAplicacao' && metricKey !== 'vento_kph' && metricKey !== 'delta_t') {
                const values = segments.filter(s => s.properties.hour >= startHour && s.properties.hour <= endHour).map(s => s.properties[metricKey]).filter(v => v !== undefined && v !== null);
                if (values.length > 0) { min = Math.min(...values); max = Math.max(...values); } else { min = 0; max = 0; }
            }
            segments.forEach(seg => {
                const segmentHour = seg.properties.hour;
                if (segmentHour < startHour || segmentHour > endHour) return;
                if (seg.coords && seg.coords.length > 1) {
                    const value = seg.properties[metricKey];
                    let color;
                    if (metricKey === 'vento_kph' || metricKey === 'delta_t') color = getSemaforicColor(metricKey, value);
                    else if (metricKey === 'CondicaoAplicacao') color = SPRAY_CONDITIONS_COLOR[value] || '#6c757d';
                    else color = getColorForValue(value, min, max);
                    const line = L.polyline(seg.coords, { color: color, weight: 5, opacity: 0.8 });
                    const tooltipText = `${seg.properties.machine} @ ${seg.properties.time} <br><b>${metrics[metricKey]?.label}:</b> ${value !== undefined ? (typeof value === 'number' ? value.toFixed(1) : value) : 'N/A'} ${unit}`;
                    line.bindTooltip(tooltipText);
                    trails.addLayer(line);
                    bounds.extend(line.getBounds());
                }
            });
            if (fitBounds && bounds.isValid()) maps[id].fitBounds(bounds, { padding: [50, 50] });
        }
        function drawBorders(id) {
            const borders = JSON.parse(document.getElementById(id).getAttribute('data-borders'));
            const borderGroup = L.layerGroup().addTo(maps[id]);
            borders.forEach(b => { if (b.coords && b.coords.length > 0) L.polygon(b.coords, { color: '#1a73e8', weight: 2, fillOpacity: 0.1 }).bindPopup(b.name).addTo(borderGroup); });
        }
        function updateLegend(id, metricKey) {
            const legendEl = document.getElementById(`legend_${id}`);
            const metrics = JSON.parse(document.getElementById(id).getAttribute('data-available-metrics'));
            const label = metrics[metricKey]?.label || '';
            if (metricKey === 'CondicaoAplicacao' || metricKey === 'vento_kph' || metricKey === 'delta_t') {
                legendEl.innerHTML = `<strong>${label}</strong>: <span style="background-color:${SPRAY_CONDITIONS_COLOR['Ideal']}"></span> Ideal <span style="background-color:${SPRAY_CONDITIONS_COLOR['Atenção']}"></span> Atenção <span style="background-color:${SPRAY_CONDITIONS_COLOR['Evitar']}"></span> Evitar `;
            } else {
                legendEl.innerHTML = `<strong>${label}</strong>: <span style="background: linear-gradient(to right, #440154, #2a788e, #7ad151, #fde725)"></span>`;
            }
        }
        function getColorForValue(value, min, max) {
            const COLOR_SCALE = ['#440154', '#414487', '#2a788e', '#22a884', '#7ad151', '#fde725'];
            if (value === undefined || value === null) return '#6c757d';
            if (value <= min) return COLOR_SCALE[0];
            if (value >= max) return COLOR_SCALE[COLOR_SCALE.length - 1];
            if (max - min === 0) return COLOR_SCALE[0];
            const ratio = (value - min) / (max - min);
            const index = Math.floor(ratio * (COLOR_SCALE.length - 1));
            return COLOR_SCALE[index];
        }
        function renderSprayingWindowChart(id) {
            const canvasId = `chart-janela-${id}`;
            if (charts[canvasId]) { charts[canvasId].destroy(); }
            const detailEl = document.getElementById(id);
            const janelaData = JSON.parse(detailEl.getAttribute('data-janela'));
            const dataByHour = Array(24).fill(null).map(() => ({ conditions: [], work_duration: 0 }));
            janelaData.forEach(d => {
                const hour = new Date(d.Datetime).getHours();
                if(d.CondicaoAplicacao) dataByHour[hour].conditions.push(d.CondicaoAplicacao);
                if(d.Operating_Mode === 'Trabalho Produtivo' && d.duration_sec) dataByHour[hour].work_duration += d.duration_sec;
            });
            const labels = Array.from({length: 24}, (_, i) => `${i}h`);
            const backgroundColors = dataByHour.map(h => {
                if (h.conditions.length === 0) return SPRAY_CONDITIONS_COLOR['Sem Dados'];
                const mode = h.conditions.reduce((a, b, i, arr) => (arr.filter(v => v === a).length >= arr.filter(v => v === b).length ? a : b));
                return SPRAY_CONDITIONS_COLOR[mode];
            });
            const workMinutes = dataByHour.map(h => h.work_duration / 60);
            const ctx = document.getElementById(canvasId).getContext('2d');
            charts[canvasId] = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [
                        { label: 'Minutos Trabalhados', data: workMinutes, backgroundColor: 'rgba(100, 100, 100, 0.7)', borderColor: 'rgba(200, 200, 200, 1)', borderWidth: 1, yAxisID: 'y_minutes', order: 1 },
                        { label: 'Condição Climática', data: Array(24).fill(1), backgroundColor: backgroundColors, yAxisID: 'y_condition', barPercentage: 1.0, categoryPercentage: 1.0, order: 2 }
                    ]
                },
                options: { responsive: true, maintainAspectRatio: false, plugins: { tooltip: { callbacks: { label: function(context) { if (context.dataset.label === 'Minutos Trabalhados') return `Trabalho: ${Math.round(context.raw)} min`; const hour = context.dataIndex; const conds = dataByHour[hour].conditions; if (conds.length === 0) return 'Condição: Sem Dados'; const mode = conds.reduce((a, b, i, arr) => (arr.filter(v => v === a).length >= arr.filter(v => v === b).length ? a : b)); return `Condição: ${mode}`; } } } }, scales: { x: { stacked: true }, y_minutes: { type: 'linear', position: 'left', title: { display: true, text: 'Minutos Trabalhados' }, max: 60, grid: { drawOnChartArea: false } }, y_condition: { display: false, stacked: true } } }
            });
        }
        """

        with open('index.html', 'w', encoding='utf-8') as f:
            f.write(f"""
<!DOCTYPE html><html lang="pt-BR"><head><meta charset="UTF-8"><title>Relatório de Telemetria e Clima</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css" /><script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>{css_styles}</style>
</head><body>
<h1>Calendário de Atividades Diárias</h1>
{html_calendario}
<div id="details-section">{html_detalhes_eventos}</div>
<script>{js_script}</script>
</body></html>""")

        print(f"\nRelatório 'index.html' gerado com sucesso!")
        # webbrowser removido para o GitHub

if __name__ == "__main__":
    print("Iniciando autenticação via farm_auth...")
    sessao_autenticada = get_authenticated_session()
    
    if not sessao_autenticada:
        print("❌ ERRO CRÍTICO: Falha na autenticação.")
        sys.exit(1)
    
    analisador = AnalisadorTelemetriaClima(session=sessao_autenticada)
    analisador.executar_relatorio()
