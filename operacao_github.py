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
import sys
import os

# --- CONFIGURAÇÕES FIXAS (Clima) ---
# Se quiser fixar um ID de produtor para pular o input, altere aqui. Se deixar None, ele pergunta.
GROWER_ID_FIXO =  1139788

# Dados da Estação (Hardcoded para não precisar subir Excel)
# Adicione mais estações aqui se necessário
ESTACOES_FIXAS = [
    {'id_grower': 1139788, 'name': 'Fazenda Guará', 'id_estacao': 52944, 'latitude': -21.6533, 'longitude': -55.4610}
]

# --- CONSTANTES GLOBAIS ---
VELOCIDADE_LIMITE_PARADO = 0.1
DURACAO_MINIMA_PARADA_SEG = 90
FATOR_KG_POR_SACO_PADRAO = 60
MAX_RENDIMENTO_REALISTA_SC_POR_HA = 200
MAX_PULSE_GAP_SECONDS = 180
AREA_MINIMA_BLOCO_HA = 4.0
VELOCIDADE_VARIACAO_LIMITE_PERCENT = 40.0

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
    print("❌ ERRO: Bibliotecas geográficas (shapely, pyproj, sklearn) não encontradas.")
    print("Instale: pip install shapely pyproj scikit-learn pandas requests")
    sys.exit(1)

class MasterFarmAnalyzer:
    def __init__(self):
        print("🔄 Conectando ao FarmCommand via farm_auth...")
        self.session = get_authenticated_session()
        
        if not self.session:
            print("❌ Falha crítica na autenticação. Encerrando script.")
            sys.exit(1)
            
        # URLs da API
        self.assets_url = "https://admin.farmcommand.com/asset/?season=1083"
        self.field_border_url = "https://admin.farmcommand.com/fieldborder/?assetID={}&format=json"
        self.canplug_url = "https://admin.farmcommand.com/canplug/?growerID={}"
        self.canplug_iot_url = "https://admin.farmcommand.com/canplug/iot/{}/?format=json"
        self.weather_url_base = "https://admin.farmcommand.com/weather/{}/historical-summary-hourly/"

        # Caches e Mapas
        self.cache_limites_talhoes = {}
        self.machine_types_map = {}
        self.machine_names_map = {}
        self.fuso_horario_cuiaba = timezone('America/Cuiaba')
        self.operacoes_definidas = []
        
        # Carrega estações fixas
        self.estacoes_climaticas = pd.DataFrame(ESTACOES_FIXAS)

    # =========================================================================
    #  MÉTODOS DE CLIMA (DO CÓDIGO 1)
    # =========================================================================
    def _get_estacoes_para_produtor(self, grower_id: int) -> pd.DataFrame:
        if self.estacoes_climaticas.empty: return pd.DataFrame()
        return self.estacoes_climaticas[self.estacoes_climaticas['id_grower'] == grower_id].copy()

    def _buscar_dados_climaticos_para_produtor(self, grower_id: int, start_date: str, end_date: str) -> pd.DataFrame:
        estacoes_produtor = self._get_estacoes_para_produtor(grower_id)
        if estacoes_produtor.empty:
            print(f"AVISO: Nenhuma estação climática configurada para este produtor (ID {grower_id}).")
            return pd.DataFrame()

        all_dfs = []
        for _, station in estacoes_produtor.iterrows():
            station_id = int(station['id_estacao'])
            station_name = station['name']
            print(f"   ☁️  Buscando Clima: {station_name} (ID: {station_id})...")
            
            raw_data = self._buscar_dados_climaticos_por_estacao(str(station_id), start_date, end_date)
            if raw_data:
                df_station = self._processar_clima_para_dataframe(raw_data, station_id, station_name)
                all_dfs.append(df_station)
        
        if not all_dfs: return pd.DataFrame()
        return pd.concat(all_dfs, ignore_index=True)

    def _buscar_dados_climaticos_por_estacao(self, station_id: str, start_date: str, end_date: str) -> list:
        all_results = []
        # Converte strings para datetime.date
        try:
            current_date = datetime.strptime(start_date, '%Y-%m-%d').date()
            final_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        except ValueError:
            current_date = pd.to_datetime(start_date).date()
            final_date = pd.to_datetime(end_date).date()
        
        while current_date <= final_date:
            api_start = current_date.strftime('%Y-%m-%dT00:00:00')
            api_end = current_date.strftime('%Y-%m-%dT23:59:59')
            url = self.weather_url_base.format(station_id)
            params = {'startDate': api_start, 'endDate': api_end, 'format': 'json'}
            json_data = self._fazer_requisicao(url, params=params)
            
            if json_data and 'results' in json_data:
                all_results.extend(json_data['results'])
            
            # Pequeno delay para não sobrecarregar
            time.sleep(0.05)
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

    @staticmethod
    def _haversine_distance(lat1, lon1, lat2, lon2):
        if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2): return 99999
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
        print("\n⚡ Cruzando dados de Telemetria com Clima...")
        if df_clima.empty or df_telemetria.empty:
            cols_clima = ['temp_c', 'umidade_relativa', 'vento_kph', 'delta_t', 'rajada_vento_kph']
            for c in cols_clima: df_telemetria[c] = np.nan
            df_telemetria['CondicaoAplicacao'] = 'Sem Dados'
            return df_telemetria
        
        # Encontra a estação mais próxima para cada ponto (pode ser lento se muitos dados, mas é preciso)
        # Otimização: Agrupar por dia/hora/talhão seria ideal, mas linha a linha garante precisão
        # Se estacoes_produtor tiver só 1 linha, é direto.
        if len(estacoes_produtor) == 1:
            station_id = estacoes_produtor.iloc[0]['id_estacao']
            df_telemetria['nearest_station_id'] = float(station_id)
        else:
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
        
        # Lógica semafórica para aplicação
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
        
        # Limpeza pós-merge
        cols_drop = ['merge_date', 'merge_hour', 'nearest_station_id', 'station_id']
        cols_drop = [c for c in cols_drop if c in df_merged.columns]
        return df_merged.drop(columns=cols_drop)

    # =========================================================================
    #  MÉTODOS DE TELEMETRIA E GEOMETRIA (DO CÓDIGO 2 - MAIS ROBUSTO)
    # =========================================================================
    def _fazer_requisicao(self, url: str, params: dict = None) -> dict | None:
        for tentativa in range(2):
            try:
                response = self.session.get(url, params=params, timeout=300)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code in [401, 403]:
                    print("Sessão expirada. Re-autenticando...")
                    self.session = get_authenticated_session()
                    if self.session: continue
                if tentativa == 0: time.sleep(3)
            except json.JSONDecodeError: return None
        return None

    def buscar_dados(self, implement_id: int, data_inicio: str, data_fim: str) -> dict | None:
        url = "https://admin.farmcommand.com/canplug/historical-summary/"
        params = {'endDate': data_fim, 'format': 'json', 'implementID': implement_id, 'startDate': data_inicio}
        return self._fazer_requisicao(url, params)

    def _get_first_valid_value(self, data: dict, keys: list) -> any:
        for key in keys:
            if data.get(key) is not None: return data.get(key)
        return None

    def analisar_json_telemetria(self, json_data: dict | list, implement_id: int) -> pd.DataFrame:
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
            if feature.get('type') == 'Feature' and feature.get('geometry'):
                coords = feature['geometry'].get('coordinates')
                if not coords or len(coords) < 2: continue
                
                props = feature.get('properties', {}).copy()
                props['Longitude'], props['Latitude'] = coords[0], coords[1]
                
                # Normaliza valores aninhados (ex: {value: 10, unit: 'km/h'})
                for key in list(props.keys()):
                    if isinstance(props[key], dict):
                        props[key] = props[key].get('value', props[key].get('status'))

                # Velocidade
                vel_mph = self._get_first_valid_value(props, ['Computed Velocity (miles/hour)', 'velocity'])
                vel_kmh = pd.to_numeric(vel_mph, errors='coerce') * 1.60934
                props['velocity'] = vel_kmh if pd.notna(vel_kmh) and vel_kmh <= 60 else np.nan
                
                # Combustível e Largura
                props['Fuel Rate (L/h)'] = self._get_first_valid_value(props, ['Fuel Rate (L/h)', 'Fuel Consumption (L/h)', 'Instantaneous Liquid Fuel Usage (L/hour)'])
                width_m = self._get_first_valid_value(props, ['machine width (meters)', 'Implement Width (meters)'])
                if width_m is None and 'Header width in use (mm)' in props:
                    width_m = props.get('Header width in use (mm)', 0) / 1000
                props['machine width (meters)'] = width_m
                
                # Colheita
                props['Mass flow value'] = self._get_first_valid_value(props, ['Mass flow value', 'Mass flow (kg/sec)', 'Grain Flow 1 (kg/s)'])
                props['Coolant Temp (C)'] = self._get_first_valid_value(props, ['Engine Coolant Temperature (C)', 'Coolant Temperature'])
                
                registros.append(props)

        if not registros: return pd.DataFrame()
        df = pd.DataFrame(registros)

        df['Datetime'] = pd.to_datetime(df['Timestamp (sec)'], unit='s', utc=True).dt.tz_convert(self.fuso_horario_cuiaba)
        df['Date'] = df['Datetime'].dt.date
        df['ImplementID'] = implement_id
        df['MachineName'] = self.machine_names_map.get(implement_id, f"ID {implement_id}")

        cols_num = ['velocity', 'Fuel Rate (L/h)', 'Engine RPM', 'machine width (meters)',
                    'Mass flow value', 'Grain Moisture (%)', 'Actual Engine - Percent Torque (%)',
                    'Coolant Temp (C)', 'Engine Oil Temperature (C)']
        for col in cols_num:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

        # Produtividade Estimada
        try:
            area_por_hora_ha = (df['velocity'] * 1000 * df['machine width (meters)']) / 10000
            kg_por_hora = df['Mass flow value'] * 3600
            kg_por_ha = kg_por_hora / area_por_hora_ha
            sc_por_ha = (kg_por_ha / FATOR_KG_POR_SACO_PADRAO)
            df['Produtividade (sc/ha)'] = np.where((sc_por_ha > 0) & (sc_por_ha < MAX_RENDIMENTO_REALISTA_SC_POR_HA), sc_por_ha, np.nan)
        except:
            df['Produtividade (sc/ha)'] = np.nan

        # Identificar Talhão
        df['Inside_Field_Name'] = pd.NA
        if self.cache_limites_talhoes:
            for idx, row in df.iterrows():
                if pd.notna(row['Latitude']):
                    for border in self.cache_limites_talhoes.values():
                        if border.get('coordinates') and self._is_ponto_no_poligono((row['Latitude'], row['Longitude']), border['coordinates']):
                            df.loc[idx, 'Inside_Field_Name'] = border['field_name']
                            break
        return df

    def _buscar_dados_com_chunking(self, implement_id: int, start_date_str: str, end_date_str: str) -> pd.DataFrame:
        all_dfs = []
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        except ValueError:
            start_date = pd.to_datetime(start_date_str).date()
            end_date = pd.to_datetime(end_date_str).date()

        current_date = start_date
        print(f"   🚜 Buscando Telemetria (Formiguinha): ID {implement_id}...")

        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            horas = range(24)
            sys.stdout.write(f"      Dia {date_str}: ")
            for h in horas:
                api_start = f"{date_str}T{h:02d}:00:00"
                api_end = f"{date_str}T{h:02d}:59:59"
                
                sucesso = False
                tentativas = 0
                while not sucesso and tentativas < 3:
                    try:
                        if tentativas > 0: time.sleep(1)
                        json_data = self.buscar_dados(implement_id, api_start, api_end)
                        if json_data:
                            df_chunk = self.analisar_json_telemetria(json_data, implement_id)
                            if not df_chunk.empty:
                                all_dfs.append(df_chunk)
                                sys.stdout.write(".") # Tem dados
                            else:
                                sys.stdout.write("_") # Vazio (sucesso)
                        else:
                            sys.stdout.write("x") # Erro req
                        sys.stdout.flush()
                        sucesso = True
                    except Exception:
                        tentativas += 1
                        time.sleep(1)
                if not sucesso: sys.stdout.write("!")
            print("") # Quebra linha
            current_date += timedelta(days=1)
            
        if not all_dfs: return pd.DataFrame()
        df_combined = pd.concat(all_dfs, ignore_index=True)
        if 'Timestamp (sec)' in df_combined.columns:
            df_combined = df_combined.drop_duplicates(subset=['Timestamp (sec)', 'ImplementID'])
        df_combined['Date'] = pd.to_datetime(df_combined['Date'])
        return df_combined.sort_values(by='Timestamp (sec)').reset_index(drop=True)

    # ... [Métodos de Geometria: _get_utm_projection, _estimar_largura_por_geometria, _calcular_area_e_sobreposicao] ...
    # (Mantidos iguais ao código 2, resumo aqui para brevidade)
    def _get_utm_projection(self, df):
        if df.empty or df['Longitude'].isnull().all(): return None
        avg_lon = df['Longitude'].mean()
        avg_lat = df['Latitude'].mean()
        utm_zone = math.floor((avg_lon + 180) / 6) + 1
        return CRS(f"EPSG:327{utm_zone}") if avg_lat < 0 else CRS(f"EPSG:326{utm_zone}")

    def _estimar_largura_por_geometria(self, df_maquina):
        # ... (Lógica do código 2) ...
        df_trabalho = df_maquina[df_maquina['Operating_Mode'] == 'Trabalho Produtivo'].copy()
        if len(df_trabalho) < 50: return None
        try:
            crs_wgs84 = CRS("EPSG:4326")
            crs_utm = self._get_utm_projection(df_trabalho)
            transformer = Transformer.from_crs(crs_wgs84, crs_utm, always_xy=True)
            points_utm = [transformer.transform(lon, lat) for lon, lat in df_trabalho[['Longitude', 'Latitude']].values]
            df_trabalho[['utm_x', 'utm_y']] = points_utm
            coords = df_trabalho[['utm_x', 'utm_y']].values
            df_trabalho['pass_label'] = DBSCAN(eps=10, min_samples=5).fit(coords).labels_
            pass_labels = [l for l in df_trabalho['pass_label'].unique() if l != -1]
            if len(pass_labels) < 2: return None
            pass_lines = {l: LineString(df_trabalho[df_trabalho['pass_label'] == l][['utm_x', 'utm_y']].values) for l in pass_labels}
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
            return bin_edges[np.argmax(hist)]
        except: return None

    def _calcular_area_e_sobreposicao(self, df_pontos_trabalho, largura_maquina_m):
        # ... (Lógica do código 2) ...
        if df_pontos_trabalho.empty or pd.isna(largura_maquina_m) or largura_maquina_m <= 0: return {'area_ha': 0.0, 'sobreposicao_percent': 0.0}
        try:
            crs_utm = self._get_utm_projection(df_pontos_trabalho)
            if not crs_utm: return {'area_ha': 0.0, 'sobreposicao_percent': 0.0}
            transformer = Transformer.from_crs(CRS("EPSG:4326"), crs_utm, always_xy=True)
            all_polygons = []
            df_pontos_trabalho = df_pontos_trabalho.sort_values('Timestamp (sec)')
            df_pontos_trabalho['segment_id'] = (df_pontos_trabalho['Timestamp (sec)'].diff() > MAX_PULSE_GAP_SECONDS).cumsum()
            for _, segment_df in df_pontos_trabalho.groupby(['ImplementID', 'segment_id']):
                if len(segment_df) < 2: continue
                points_utm = [transformer.transform(lon, lat) for lon, lat in segment_df[['Longitude', 'Latitude']].values]
                line = LineString(points_utm)
                all_polygons.append(line.buffer(largura_maquina_m / 2, cap_style=2, join_style=2))
            if not all_polygons: return {'area_ha': 0.0, 'sobreposicao_percent': 0.0}
            uniao_total = unary_union(all_polygons)
            area_bruta = sum(p.area for p in all_polygons)
            area_unica = uniao_total.area
            return {'area_ha': area_unica / 10000, 'sobreposicao_percent': ((area_bruta - area_unica) / area_unica * 100) if area_unica > 0 else 0}
        except: return {'area_ha': 0.0, 'sobreposicao_percent': 0.0}

    # ... [Helpers de Poligono e Outros] ...
    @staticmethod
    def _is_ponto_no_poligono(ponto, poligono):
        p_lat, p_lon = ponto
        n, dentro = len(poligono), False
        if n < 3: return False
        p1_lat, p1_lon = poligono[0]
        for i in range(n + 1):
            p2_lat, p2_lon = poligono[i % n]
            if p_lat > min(p1_lat, p2_lat) and p_lat <= max(p1_lat, p2_lat) and p_lon <= max(p1_lon, p2_lon) and p1_lat != p2_lat:
                xinters = (p_lat - p1_lat) * (p2_lon - p1_lon) / (p2_lat - p1_lat) + p1_lon
                if p1_lon == p2_lon or p_lon <= xinters: dentro = not dentro
            p1_lat, p1_lon = p2_lat, p2_lon
        return dentro

    def _formatar_duracao(self, segundos):
        if pd.isna(segundos) or segundos < 0: return "00:00"
        h, m = int(segundos // 3600), int((segundos % 3600) // 60)
        return f"{h}h {m:02d}m"

    # =========================================================================
    #  LÓGICA DE NEGÓCIO PRINCIPAL (Gerar Blocos e Relatório)
    # =========================================================================
    def _identificar_blocos_operacao(self, df):
        # Mesma lógica robusta do Código 2
        print("\n🏗️  Identificando Blocos de Operação...")
        df_trabalho = df[(df['Operating_Mode'] == 'Trabalho Produtivo') & (df['Inside_Field_Name'].notna())].copy()
        if df_trabalho.empty: return
        
        df_trabalho['DateOnly'] = pd.to_datetime(df_trabalho['Date']).dt.date
        micro_blocos = []
        for (imp_id, talhao, data_dia), grupo in df_trabalho.groupby(['ImplementID', 'Inside_Field_Name', 'DateOnly']):
            micro_blocos.append({
                'ImplementID': imp_id, 'Inside_Field_Name': talhao, 'DateOnly': data_dia,
                'avg_daily_speed': grupo['velocity'].mean(), 'indices_df': grupo.index.tolist()
            })
        
        df_blocos = pd.DataFrame(micro_blocos).sort_values(by=['ImplementID', 'Inside_Field_Name', 'DateOnly'])
        op_counter = 1
        current_block_id = -1
        last_row = None
        
        for index, row in df_blocos.iterrows():
            if last_row is None:
                current_block_id = op_counter
                op_counter += 1
            else:
                is_same = (row['ImplementID'] == last_row['ImplementID']) and (row['Inside_Field_Name'] == last_row['Inside_Field_Name'])
                date_diff = (row['DateOnly'] - last_row['DateOnly']).days
                speed_change = abs(row['avg_daily_speed'] - last_row['avg_daily_speed']) / last_row['avg_daily_speed'] * 100 if last_row['avg_daily_speed'] > 0 else 0
                
                if is_same and date_diff == 1 and speed_change < VELOCIDADE_VARIACAO_LIMITE_PERCENT:
                    pass # Mantém current_block_id
                else:
                    current_block_id = op_counter
                    op_counter += 1
            
            df_blocos.loc[index, 'final_block_id'] = current_block_id
            last_row = row

        for final_id, grupo in df_blocos.groupby('final_block_id'):
            todos_indices = [idx for sublist in grupo['indices_df'] for idx in sublist]
            primeiro = grupo.iloc[0]
            nome = f"Op {final_id}: {self.machine_names_map.get(primeiro['ImplementID'], 'Maq').split()[0]} em {primeiro['Inside_Field_Name']} ({primeiro['DateOnly'].strftime('%d/%m')})"
            
            self.operacoes_definidas.append({'id': final_id, 'nome': nome, 'tipo': "Não definido", 'indices_df': todos_indices})
            df.loc[todos_indices, 'OperationID'] = final_id
            df.loc[todos_indices, 'OperationName'] = nome

    def executar_relatorio(self):
        # 1. Configuração (Input ou Fixo)
        if GROWER_ID_FIXO:
            grower_id = GROWER_ID_FIXO
            print(f"Usando ID Produtor Fixo: {grower_id}")
            # Pega data de ontem e hoje para teste rápido, ou mude para input
            hoje = datetime.now()
            data_fim = hoje.strftime('%Y-%m-%d')
            data_inicio = (hoje - timedelta(days=5)).strftime('%Y-%m-%d') # Últimos 5 dias
            
            todos_implementos = self.obter_canplugs_por_produtor(grower_id)
            implement_ids = todos_implementos # Pega todos por padrão
            largura_manual = None
        else:
            # Lógica de input do Código 2
            try:
                grower_id = int(input("ID do Produtor: ").strip())
                self.obter_canplugs_por_produtor(grower_id)
                ids_str = input("IDs das Máquinas (separados por vírgula): ").strip()
                implement_ids = [int(x) for x in ids_str.split(',')]
                data_inicio = input("Data Início (YYYY-MM-DD): ").strip()
                data_fim = input("Data Fim (YYYY-MM-DD): ").strip()
                largura_manual = input("Largura manual (opcional, Enter para pular): ").strip()
                largura_manual = float(largura_manual) if largura_manual else None
            except:
                print("Entrada inválida."); return

        # 2. Obter Dados
        print("\n🚀 Iniciando Carga de Dados...")
        self.obter_talhoes_por_produtor(grower_id)
        
        # Telemetria
        dfs_telemetria = [df for df in [self._buscar_dados_com_chunking(mid, data_inicio, data_fim) for mid in implement_ids] if not df.empty]
        if not dfs_telemetria: print("Sem telemetria."); return
        df_telemetria = pd.concat(dfs_telemetria, ignore_index=True)
        
        # Clima
        df_clima = self._buscar_dados_climaticos_para_produtor(grower_id, data_inicio, data_fim)
        
        # 3. Processamento
        # Merge Telemetria + Clima
        df_final = self._merge_telemetria_clima(df_telemetria, df_clima, self._get_estacoes_para_produtor(grower_id))
        
        # Definição de Modos e Largura
        df_final['Operating_Mode'] = np.select(
            [df_final['velocity'] <= VELOCIDADE_LIMITE_PARADO, df_final['Inside_Field_Name'].notna()],
            ['Parado', 'Trabalho Produtivo'], default='Deslocamento'
        )
        
        print("📏 Calculando larguras...")
        for mid in df_final['ImplementID'].unique():
            mask = df_final['ImplementID'] == mid
            if largura_manual: largura = largura_manual
            else:
                # Tenta API, se não, tenta Geometria
                larguras_api = df_final.loc[mask, 'machine width (meters)'].dropna()
                largura = Counter(larguras_api).most_common(1)[0][0] if not larguras_api.empty else (self._estimar_largura_por_geometria(df_final[mask]) or 10.0)
            df_final.loc[mask, 'machine width (meters)'] = largura

        # Identificação de Blocos
        self._identificar_blocos_operacao(df_final)
        
        # 4. Geração do HTML Master
        print("\n🎨 Gerando Relatório Master HTML...")
        self.gerar_html_master(df_final, data_inicio, data_fim)

    def obter_canplugs_por_produtor(self, grower_id):
        # Helper simples para popular self.machine_names_map
        canplugs = self._fazer_requisicao(self.canplug_url.format(grower_id)) or []
        ids = []
        for c in canplugs:
            imps = c.get('implements', [])
            if isinstance(imps, str): imps = json.loads(imps)
            if not isinstance(imps, list): imps = [imps]
            
            name = "Desconhecido"
            if c.get('canplugID'):
                iot = self._fazer_requisicao(self.canplug_iot_url.format(c['canplugID']))
                if iot: name = iot.get('installed_in', {}).get('name', 'Maq')
            
            for i in imps:
                if str(i).isdigit():
                    self.machine_names_map[int(i)] = name
                    ids.append(int(i))
        return list(set(ids))

    def obter_talhoes_por_produtor(self, grower_id):
        # Popula cache_limites_talhoes
        assets = self._fazer_requisicao(self.assets_url) or []
        farms = [a['id'] for a in assets if a.get('category') == 'Farm' and a.get('parent') == grower_id]
        fields = [a for a in assets if a.get('category') == 'Field' and a.get('parent') in farms]
        for f in fields:
            if f['id'] not in self.cache_limites_talhoes:
                b = self._fazer_requisicao(self.field_border_url.format(f['id']))
                if b and 'shapeData' in b[0]:
                     geom = json.loads(b[0]['shapeData']).get('features', [{}])[0].get('geometry', {})
                     if geom.get('type') == 'Polygon':
                         coords = [[c[1], c[0]] for c in geom['coordinates'][0]]
                         self.cache_limites_talhoes[f['id']] = {'coordinates': coords, 'field_name': f['label']}

    # =========================================================================
    #  GERAÇÃO DE HTML (FUSÃO DAS DUAS VERSÕES)
    # =========================================================================
    def gerar_html_master(self, df, data_inicio, data_fim):
        # Prepara dados para o calendário diário (com clima e eficiência)
        eventos_calendario = []
        df['DateOnly'] = pd.to_datetime(df['Date']).dt.date
        
        for (dia, mid), subdf in df.groupby(['DateOnly', 'ImplementID']):
            produtivo = subdf[subdf['Operating_Mode'] == 'Trabalho Produtivo']
            area_calc = self._calcular_area_e_sobreposicao(produtivo, subdf['machine width (meters)'].median())
            
            # Métricas Climáticas (Médias do dia na operação)
            resumo_clima = {
                'delta_t': subdf['delta_t'].mean(),
                'vento': subdf['vento_kph'].mean(),
                'temp': subdf['temp_c'].mean()
            }
            
            # HTML Resumo do Dia (Card)
            resumo_html = f"""
            <div class='summary-card'>
                <h4>{self.machine_names_map.get(mid, str(mid))} - {dia.strftime('%d/%m')}</h4>
                <div class='grid-2'>
                    <div>🌱 Área: {area_calc['area_ha']:.1f} ha</div>
                    <div>⏱️ Produtivo: {self._formatar_duracao(produtivo['duration_sec'].sum())}</div>
                    <div>🌡️ Delta T Médio: {resumo_clima['delta_t']:.1f}</div>
                    <div>💨 Vento Médio: {resumo_clima['vento']:.1f}</div>
                </div>
            </div>
            """
            
            # Dados JSON para o mapa (Inclui clima)
            metricas_mapa = ['velocity', 'Fuel Rate (L/h)', 'delta_t', 'vento_kph', 'CondicaoAplicacao', 'Engine RPM']
            cols_existentes = [c for c in metricas_mapa if c in subdf.columns]
            
            # Segmentos para o mapa
            segmentos = []
            subdf_sorted = subdf.sort_values('Timestamp (sec)')
            # Simplificação: cria segmentos a cada ponto para colorir
            # (Para produção real, ideal reduzir resolução ou usar GeoJSON LineString por cor)
            points_json = subdf_sorted[['Latitude', 'Longitude'] + cols_existentes + ['Timestamp (sec)']].dropna(subset=['Latitude', 'Longitude']).to_dict(orient='records')
            
            # Prepara segmentos simples (ponto a ponto) para o Leaflet desenhar colorido
            # Estrutura otimizada: Lista de {coords: [[lat1,lon1],[lat2,lon2]], props: {val...}}
            for i in range(len(points_json)-1):
                p1 = points_json[i]
                p2 = points_json[i+1]
                if (p2['Timestamp (sec)'] - p1['Timestamp (sec)']) > MAX_PULSE_GAP_SECONDS: continue
                
                props = {k: p1[k] for k in cols_existentes}
                props['time'] = pd.to_datetime(p1['Timestamp (sec)'], unit='s').strftime('%H:%M')
                segmentos.append({'coords': [[p1['Latitude'], p1['Longitude']], [p2['Latitude'], p2['Longitude']]], 'properties': props})

            eventos_calendario.append({
                'date': dia,
                'implement_id': mid,
                'title': f"{self.machine_names_map.get(mid, str(mid))}",
                'summary_html': resumo_html,
                'segments_json': json.dumps(segmentos),
                'metrics_avail': json.dumps(cols_existentes),
                'borders': json.dumps([v for v in self.cache_limites_talhoes.values()])
            })

        # Prepara dados da tabela de operações (Código 2)
        # (Logica simplificada para brevidade - usa self.operacoes_definidas)
        # ... [Implementação da tabela similar ao codigo 2] ...
        
        # Gera o arquivo HTML
        with open('relatorio_master.html', 'w', encoding='utf-8') as f:
            f.write(self._get_html_template(eventos_calendario, data_inicio, data_fim))
        print("✅ Relatório 'relatorio_master.html' gerado com sucesso!")

    def _get_html_template(self, eventos, data_inicio, data_fim):
        # Aqui fundimos o CSS/JS.
        # Destaque: A função JS 'getColor' agora lida com o semáforo climático
        
        events_json_store = {}
        calendar_html = "<div class='calendar-container'>"
        current = datetime.strptime(data_inicio, '%Y-%m-%d').date()
        end = datetime.strptime(data_fim, '%Y-%m-%d').date()
        
        while current <= end:
            day_str = current.strftime('%Y-%m-%d')
            evs_dia = [e for e in eventos if e['date'] == current]
            
            calendar_html += f"<div class='day-column'><h4>{current.strftime('%d/%m')}</h4>"
            for ev in evs_dia:
                uid = f"{day_str}_{ev['implement_id']}"
                events_json_store[uid] = ev
                calendar_html += f"<div class='event-card' onclick=\"loadMap('{uid}')\">{ev['summary_html']}</div>"
            calendar_html += "</div>"
            current += timedelta(days=1)
        calendar_html += "</div>"

        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Relatório Master Farm</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
    <style>
        body {{ font-family: sans-serif; margin: 0; padding: 20px; background: #f0f2f5; }}
        .calendar-container {{ display: flex; overflow-x: auto; gap: 10px; padding-bottom: 20px; }}
        .day-column {{ min-width: 250px; background: #fff; padding: 10px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .event-card {{ background: #e6f7ff; border: 1px solid #91d5ff; border-radius: 5px; padding: 8px; margin-bottom: 8px; cursor: pointer; transition: transform 0.2s; }}
        .event-card:hover {{ transform: scale(1.02); background: #bae7ff; }}
        .summary-card h4 {{ margin: 0 0 5px 0; color: #0050b3; }}
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 5px; font-size: 0.85em; }}
        
        #map-area {{ display: none; margin-top: 20px; background: #fff; padding: 20px; border-radius: 8px; height: 600px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }}
        #map {{ height: 500px; width: 100%; }}
        .controls {{ margin-bottom: 10px; }}
        .legend {{ background: white; padding: 10px; border-radius: 4px; border: 1px solid #ccc; font-size: 0.8em; }}
        .dot {{ height: 10px; width: 10px; display: inline-block; border-radius: 50%; margin-right: 5px; }}
    </style>
</head>
<body>
    <h1>🚜 Relatório Master: Operações & Clima</h1>
    {calendar_html}
    
    <div id="map-area">
        <div class="controls">
            <label>Colorir Rastro por: </label>
            <select id="metricSelect" onchange="redrawMap()"></select>
            <button onclick="document.getElementById('map-area').style.display='none'">Fechar Mapa</button>
        </div>
        <div id="map"></div>
    </div>

    <script>
        const eventsData = {json.dumps(events_json_store)};
        let currentEventId = null;
        let map = null;
        let trailLayer = null;

        // CORES SEMAFÓRICAS (DO CÓDIGO 1)
        const CONDITIONS_COLOR = {{
            'Ideal': '#28a745', 'Verde': '#28a745',
            'Atenção': '#ffc107', 'Amarelo': '#ffc107',
            'Evitar': '#dc3545', 'Vermelho': '#dc3545',
            'Sem Dados': '#6c757d'
        }};
        const GRADIENT_SCALE = ['#440154', '#2a788e', '#7ad151', '#fde725']; // Roxo -> Amarelo

        function loadMap(uid) {{
            currentEventId = uid;
            document.getElementById('map-area').style.display = 'block';
            const data = eventsData[uid];
            const metrics = JSON.parse(data.metrics_avail);
            
            const select = document.getElementById('metricSelect');
            select.innerHTML = '';
            metrics.forEach(m => {{
                let opt = document.createElement('option');
                opt.value = m;
                opt.innerText = m;
                select.appendChild(opt);
            }});
            // Seleciona CondicaoAplicacao por padrão se existir
            if(metrics.includes('CondicaoAplicacao')) select.value = 'CondicaoAplicacao';

            if(!map) {{
                map = L.map('map').setView([0,0], 13);
                L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png').addTo(map);
                trailLayer = L.layerGroup().addTo(map);
            }}
            
            // Desenha bordas
            const borders = JSON.parse(data.borders);
            // (Logica simplificada de bordas aqui)
            
            setTimeout(() => {{ map.invalidateSize(); redrawMap(); }}, 100);
        }}

        function redrawMap() {{
            if(!currentEventId) return;
            trailLayer.clearLayers();
            
            const metric = document.getElementById('metricSelect').value;
            const segments = JSON.parse(eventsData[currentEventId].segments_json);
            
            // Calcula min/max para gradiente
            let min = Infinity, max = -Infinity;
            if(!['CondicaoAplicacao', 'CondiçãoAplicacao'].includes(metric)) {{
                segments.forEach(s => {{
                    const val = s.properties[metric];
                    if(val !== undefined && val !== null) {{
                        if(val < min) min = val;
                        if(val > max) max = val;
                    }}
                }});
            }}

            const bounds = L.latLngBounds();
            
            segments.forEach(seg => {{
                const val = seg.properties[metric];
                let color = '#ccc';
                
                // LÓGICA HÍBRIDA DE CORES
                if (metric === 'CondicaoAplicacao') {{
                    color = CONDITIONS_COLOR[val] || '#ccc';
                }} else if (metric === 'delta_t') {{
                    // Regra Semafórica Delta T
                    if (val >= 9) color = CONDITIONS_COLOR['Vermelho'];
                    else if (val >= 2 && val <= 8) color = CONDITIONS_COLOR['Verde'];
                    else color = CONDITIONS_COLOR['Amarelo'];
                }} else if (metric === 'vento_kph') {{
                    // Regra Semafórica Vento
                    if ((val >= 0 && val <= 1) || val > 10) color = CONDITIONS_COLOR['Vermelho'];
                    else if (val > 2 && val <= 8) color = CONDITIONS_COLOR['Verde'];
                    else color = CONDITIONS_COLOR['Amarelo'];
                }} else {{
                    // Gradiente Numérico Padrão (Velocidade, RPM, Consumo)
                    color = getGradientColor(val, min, max);
                }}

                const line = L.polyline(seg.coords, {{color: color, weight: 5, opacity: 0.8}});
                line.bindTooltip(`${{metric}}: ${{val}}`);
                trailLayer.addLayer(line);
                bounds.extend(line.getBounds());
            }});
            
            if(bounds.isValid()) map.fitBounds(bounds);
        }}

        function getGradientColor(value, min, max) {{
            if (value === null || value === undefined) return '#ccc';
            if (max === min) return GRADIENT_SCALE[0];
            const ratio = (value - min) / (max - min);
            const index = Math.min(Math.floor(ratio * GRADIENT_SCALE.length), GRADIENT_SCALE.length - 1);
            return GRADIENT_SCALE[index];
        }}
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    analyser = MasterFarmAnalyzer()
    analyser.executar_relatorio()
